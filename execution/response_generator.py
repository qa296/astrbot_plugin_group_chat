"""
回复生成器

负责生成回复内容：
- 调用 LLM 生成回复
- 人格注入
- 读空气功能
- 回复后处理
"""

import time
from dataclasses import dataclass

from astrbot.api import logger


@dataclass
class GeneratedResponse:
    """生成的回复"""

    content: str
    should_send: bool
    method: str  # llm, template, skip
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResponseGenerator:
    """
    回复生成器

    负责生成回复内容
    """

    def __init__(self, context, config):
        self.context = context
        self.config = config

        # 人格缓存
        self._persona_cache: dict[str, dict] = {}
        self._persona_cache_time: float = 0

        # 读空气配置
        air_config = config.get("air_reading", {})
        self.air_reading_enabled = air_config.get("enabled", True)
        self.no_reply_marker = air_config.get("no_reply_marker", "<NO_RESPONSE>")

        # 性能模式
        self.performance_mode = config.get("performance_mode", "balanced")

        logger.info("回复生成器初始化完成")

    async def generate(self, event, context, decision) -> str | None:
        """
        生成回复

        Args:
            event: 消息事件
            context: 分析后的上下文
            decision: 决策结果

        Returns:
            回复内容，None 表示不回复
        """
        if not decision.should_act:
            return None

        # 获取 LLM 提供商
        provider = self.context.get_using_provider()
        if not provider:
            logger.warning("无可用 LLM 提供商")
            return None

        try:
            # 构建提示词
            prompt = await self._build_prompt(event, context, decision)

            # 获取人格
            persona = await self._resolve_persona(event)

            # 构建系统提示
            system_prompt = self._build_system_prompt(persona, decision)

            # 调用 LLM
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                image_urls=[],
                system_prompt=system_prompt,
            )

            if not response or not response.completion_text:
                return None

            content = response.completion_text.strip()

            # 检查读空气标记
            if self.air_reading_enabled and self.no_reply_marker in content:
                logger.debug("LLM 决定不回复（读空气）")
                return None

            # 后处理
            content = self._post_process(content)

            return content

        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return None

    async def generate_proactive(self, group_id: str, context, decision) -> str | None:
        """
        生成主动消息

        Args:
            group_id: 群组ID
            context: 分析后的上下文
            decision: 决策结果

        Returns:
            回复内容
        """
        if not decision.should_act:
            return None

        provider = self.context.get_using_provider()
        if not provider:
            return None

        try:
            # 构建主动消息提示词
            prompt = await self._build_proactive_prompt(group_id, context, decision)

            # 获取人格
            persona = await self._resolve_persona_by_group(group_id)

            # 构建系统提示
            system_prompt = self._build_proactive_system_prompt(persona)

            # 调用 LLM
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                image_urls=[],
                system_prompt=system_prompt,
            )

            if not response or not response.completion_text:
                return None

            content = response.completion_text.strip()

            # 检查读空气标记
            if self.no_reply_marker in content:
                return None

            return self._post_process(content)

        except Exception as e:
            logger.error(f"生成主动消息失败: {e}")
            return None

    async def _build_prompt(self, event, context, decision) -> str:
        """构建提示词"""
        message_str = event.message_str
        user_name = event.get_sender_name()

        # 获取对话历史摘要
        history_summary = self._get_history_summary(context)

        # 构建提示
        prompt = f"""当前群聊场景：
- 群活跃度: {context.group_activity:.2f}
- 话题连贯性: {context.topic_coherence:.2f}
- 与你相关性: {context.relevance_to_bot:.2f}
- 你的能量: {context.last_bot_reply_ts:.2f}

最近的对话：
{history_summary}

用户 {user_name} 发送的消息：
{message_str}

请根据上下文，决定如何回复。如果不应该回复，请输出 {self.no_reply_marker}
如果应该回复，请直接输出回复内容（不要添加任何前缀或解释）。

回复："""

        return prompt

    async def _build_proactive_prompt(self, group_id: str, context, decision) -> str:
        """构建主动消息提示词"""
        # 获取最近的对话
        history_summary = self._get_history_summary(context)

        prompt = f"""当前群聊场景：
- 群活跃度: {context.group_activity:.2f}
- 话题连贯性: {context.topic_coherence:.2f}
- 最近消息数(1分钟): {context.message_count_1m}

最近的对话：
{history_summary}

你有一段时间没有发言了，现在群内有一些活动。请判断是否应该主动发言参与讨论。
如果不应该发言，请输出 {self.no_reply_marker}
如果应该发言，请直接输出你想说的内容。

发言："""

        return prompt

    def _build_system_prompt(self, persona: dict, decision) -> str:
        """构建系统提示"""
        base_prompt = """你是一个群聊中的机器人成员。你的回复应该：
1. 自然、友好，像真人一样
2. 考虑当前群聊的氛围和话题
3. 不要过于频繁地回复，避免刷屏
4. 适时保持沉默也是一种好的选择"""

        if persona and persona.get("enabled"):
            persona_prompt = persona.get("persona_prompt", "")
            if persona_prompt:
                return f"【人格设定】\n{persona_prompt}\n\n{base_prompt}"

        return base_prompt

    def _build_proactive_system_prompt(self, persona: dict) -> str:
        """构建主动消息系统提示"""
        base_prompt = """你是一个群聊中的机器人成员。你正在考虑是否主动发言。
主动发言应该：
1. 有明确的参与目的，不是为了说话而说话
2. 与当前话题相关或有新的有价值的内容
3. 自然融入群聊氛围
4. 如果没有合适的时机，选择沉默"""

        if persona and persona.get("enabled"):
            persona_prompt = persona.get("persona_prompt", "")
            if persona_prompt:
                return f"【人格设定】\n{persona_prompt}\n\n{base_prompt}"

        return base_prompt

    async def _resolve_persona(self, event) -> dict:
        """解析人格"""
        try:
            # 检查缓存
            current_time = time.time()
            if current_time - self._persona_cache_time < 300:
                cache_key = event.get_group_id() or "default"
                if cache_key in self._persona_cache:
                    return self._persona_cache[cache_key]

            # 从上下文获取人格
            pm = getattr(self.context, "provider_manager", None)
            if not pm:
                return {"enabled": False}

            # 获取对话的人格
            conversation = None
            try:
                uid = event.unified_msg_origin
                cid = await self.context.conversation_manager.get_curr_conversation_id(
                    uid
                )
                if cid:
                    conversation = (
                        await self.context.conversation_manager.get_conversation(
                            uid, cid
                        )
                    )
            except Exception:
                pass

            persona_id = (
                getattr(conversation, "persona_id", None) if conversation else None
            )

            # 显式取消人格
            if persona_id == "[%None]":
                return {"enabled": False}

            # 获取人格名称
            if persona_id:
                persona_name = persona_id
            else:
                selected = getattr(pm, "selected_default_persona", {}) or {}
                persona_name = selected.get("name", "")

            if not persona_name:
                return {"enabled": False}

            # 获取人格数据
            personas = getattr(pm, "personas", {})
            persona_data = None

            if isinstance(personas, dict):
                persona_data = personas.get(persona_name)
            elif isinstance(personas, list):
                for p in personas:
                    name = (
                        p.get("name")
                        if isinstance(p, dict)
                        else getattr(p, "name", None)
                    )
                    if name == persona_name:
                        persona_data = p
                        break

            if not persona_data:
                return {"enabled": False}

            # 提取 prompt
            if isinstance(persona_data, dict):
                prompt = (
                    persona_data.get("prompt") or persona_data.get("description") or ""
                )
            else:
                prompt = (
                    getattr(persona_data, "prompt", "")
                    or getattr(persona_data, "description", "")
                    or ""
                )

            result = {
                "enabled": True,
                "persona_name": persona_name,
                "persona_prompt": prompt,
            }

            # 缓存
            cache_key = event.get_group_id() or "default"
            self._persona_cache[cache_key] = result
            self._persona_cache_time = current_time

            return result

        except Exception as e:
            logger.debug(f"解析人格失败: {e}")
            return {"enabled": False}

    async def _resolve_persona_by_group(self, group_id: str) -> dict:
        """根据群组解析人格"""
        cache_key = group_id or "default"

        if cache_key in self._persona_cache:
            return self._persona_cache[cache_key]

        # 返回默认人格
        try:
            pm = getattr(self.context, "provider_manager", None)
            if pm:
                selected = getattr(pm, "selected_default_persona", {}) or {}
                persona_name = selected.get("name", "")

                if persona_name:
                    personas = getattr(pm, "personas", {})
                    persona_data = None

                    if isinstance(personas, dict):
                        persona_data = personas.get(persona_name)

                    if persona_data:
                        prompt = (
                            persona_data.get("prompt", "")
                            if isinstance(persona_data, dict)
                            else ""
                        )
                        return {
                            "enabled": True,
                            "persona_name": persona_name,
                            "persona_prompt": prompt,
                        }
        except Exception:
            pass

        return {"enabled": False}

    def _get_history_summary(self, context) -> str:
        """获取对话历史摘要"""
        history = getattr(context, "conversation_history", [])

        if not history:
            return "（无最近对话）"

        lines = []
        for msg in history[-5:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                lines.append(f"机器人: {content[:50]}...")
            else:
                lines.append(f"用户: {content[:50]}...")

        return "\n".join(lines)

    def _post_process(self, content: str) -> str:
        """后处理"""
        # 移除读空气标记
        content = content.replace(self.no_reply_marker, "")

        # 去除首尾空白
        content = content.strip()

        # 限制长度（可选）
        if len(content) > 500:
            content = content[:500] + "..."

        return content

    def clear_cache(self):
        """清除缓存"""
        self._persona_cache.clear()
        self._persona_cache_time = 0
