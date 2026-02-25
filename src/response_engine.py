import time
from typing import Any, Dict

from astrbot.api import logger
from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent

class ResponseEngine:
    """
    回复引擎：负责决定是否回复以及生成回复内容。
    核心功能包括传统的阈值判断和基于LLM的"读空气"判断。
    """

    def __init__(self, context: Context, config: Any):
        """
        初始化回复引擎。

        Args:
            context: AstrBot的上下文对象，提供访问核心组件的接口。
            config: 插件的配置对象，从_conf_schema.json加载。
        """
        self.context = context
        self.config = config
        # 人格缓存：{persona_name: {"prompt": str, "timestamp": float}}
        self._persona_cache = {}
    
    async def generate_response(self, event: Any, chat_context: Dict[str, Any], willingness_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成回复的主入口函数。
        根据意愿计算的结果，决定是使用传统阈值判断还是LLM“读空气”来判断是否回复。
        
        Args:
            event: AstrBot的消息事件对象。
            chat_context: 由ContextAnalyzer提供的完整聊天上下文。
            willingness_result: 由WillingnessCalculator提供的意愿计算结果。
            
        Returns:
            一个包含回复决策和内容的字典。
        """
        logger.debug(f"ResponseEngine: 开始生成回复。需要LLM决策: {willingness_result.get('requires_llm_decision')}")
        
        # 如果需要 LLM 决策，进行读空气
        if willingness_result.get("requires_llm_decision"):
            logger.info("ResponseEngine: 进入LLM读空气决策流程。")
            return await self._generate_with_air_reading(event, chat_context, willingness_result)
        else:
            # 传统回复生成
            logger.info("ResponseEngine: 进入传统阈值决策流程。")
            if willingness_result.get("should_respond"):
                response_content = await self._generate_normal_response(event, chat_context)
                return {
                    "should_reply": True,
                    "content": response_content,
                    "decision_method": "threshold",
                    "willingness_score": willingness_result.get("willingness_score")
                }
            else:
                logger.debug("ResponseEngine: 传统阈值判断为不回复。")
                return {
                    "should_reply": False,
                    "content": None,
                    "decision_method": "threshold",
                    "willingness_score": willingness_result.get("willingness_score")
                }
    
    async def _resolve_persona_text(self, event: Any) -> Dict[str, Any]:
        """
        解析当前会话的人格设定（若有），返回用于注入的提示词。
        加入缓存机制避免重复解析。
        返回: {"enabled": bool, "persona_name": str, "persona_prompt": str}
        """
        try:
            # 可通过配置关闭注入
            if isinstance(self.config, dict) and not self.config.get("enable_persona_injection", True):
                return {"enabled": False, "persona_name": "", "persona_prompt": ""}

            pm = getattr(self.context, "provider_manager", None)
            if not pm:
                return {"enabled": False, "persona_name": "", "persona_prompt": ""}

            uid = getattr(event, "unified_msg_origin", None)
            conversation = None
            if uid:
                try:
                    cid = await self.context.conversation_manager.get_curr_conversation_id(uid)
                    if cid:
                        conversation = await self.context.conversation_manager.get_conversation(uid, cid)
                except Exception:
                    conversation = None

            persona_id = getattr(conversation, "persona_id", None)
            # 显式取消人格
            if persona_id == "[%None]":
                return {"enabled": False, "persona_name": "", "persona_prompt": ""}

            # 选择 persona 名
            if persona_id:
                persona_name = persona_id
            else:
                selected = getattr(pm, "selected_default_persona", {}) or {}
                persona_name = selected.get("name", "")

            if not persona_name:
                return {"enabled": False, "persona_name": "", "persona_prompt": ""}

            # 检查缓存（学习Heartflow的缓存机制）
            current_time = time.time()
            cache_key = persona_name
            if cache_key in self._persona_cache:
                cached = self._persona_cache[cache_key]
                # 缓存5分钟内有效
                if current_time - cached.get("timestamp", 0) < 300:
                    logger.debug(f"使用人格缓存: {persona_name}")
                    return {
                        "enabled": True,
                        "persona_name": persona_name,
                        "persona_prompt": cached["prompt"]
                    }

            # 缓存未命中，正常解析
            personas = getattr(pm, "personas", {})
            persona_data = None
            # personas 可能是 dict 或 list
            if isinstance(personas, dict):
                persona_data = personas.get(persona_name)
            else:
                try:
                    for p in personas:
                        name = getattr(p, "name", None) if hasattr(p, "name") else (p.get("name") if isinstance(p, dict) else None)
                        if name == persona_name:
                            persona_data = p
                            break
                except Exception:
                    persona_data = None

            if not persona_data:
                return {"enabled": False, "persona_name": persona_name, "persona_prompt": ""}

            # 取 prompt 或 description
            if isinstance(persona_data, dict):
                prompt = persona_data.get("prompt") or persona_data.get("description") or ""
            else:
                prompt = getattr(persona_data, "prompt", "") or getattr(persona_data, "description", "") or ""

            if not prompt:
                return {"enabled": False, "persona_name": persona_name, "persona_prompt": ""}

            # 写入缓存
            self._persona_cache[cache_key] = {
                "prompt": prompt,
                "timestamp": current_time
            }

            return {"enabled": True, "persona_name": persona_name, "persona_prompt": prompt}
        except Exception:
            return {"enabled": False, "persona_name": "", "persona_prompt": ""}

    def _compose_system_prompt_with_persona(self, base_prompt: str, persona: Dict[str, Any]) -> str:
        """
        将人格前缀与基础 system prompt 组合。
        """
        if persona and persona.get("enabled") and persona.get("persona_prompt"):
            name = persona.get("persona_name", "")
            preface = f"【人格设定：{name}】\n{persona.get('persona_prompt','')}\n\n"
            return preface + base_prompt
        return base_prompt

    async def _generate_with_air_reading(self, event: Any, chat_context: Dict[str, Any], willingness_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        实现基于LLM的“读空气”功能，让LLM决定是否回复。
        
        Args:
            event: AstrBot的消息事件对象。
            chat_context: 完整的聊天上下文。
            willingness_result: 意愿计算结果。
            
        Returns:
            包含最终决策和回复内容的字典。
        """
        logger.debug("ResponseEngine: 构建读空气提示词。")
        # 构建读空气提示
        air_reading_prompt = await self._build_air_reading_prompt(event, chat_context, willingness_result)
        
        # 调用 LLM 进行读空气决策
        llm_response = await self._call_llm_for_air_reading(air_reading_prompt, event)
        
        # 检查LLM的回复是否是不回复的标记（兼容多种写法与配置）
        # 优先使用配置中的标记，其次兼容历史写法
        cfg_marker = getattr(self.config, 'air_reading_no_reply_marker', None)
        no_reply_markers = [m for m in [cfg_marker, "<NO_RESPONSE>", "[DO_NOT_REPLY]"] if m]
        resp_text = llm_response.strip()

        if any(m in resp_text for m in no_reply_markers):
            logger.info(f"ResponseEngine: LLM决定跳过回复。")
            return {
                "should_reply": False,
                "content": None,
                "decision_method": "air_reading",
                "llm_response": llm_response,
                "skip_reason": "LLM decided to skip",
                "willingness_score": willingness_result.get("willingness_score")
            }
        else:
            logger.info(f"ResponseEngine: LLM决定进行回复。")
            # LLM的回复就是直接要发送的内容
            return {
                "should_reply": True,
                "content": resp_text,
                "decision_method": "air_reading",
                "llm_response": llm_response,
                "willingness_score": willingness_result.get("willingness_score")
            }
    
    async def _build_air_reading_prompt(self, event: Any, chat_context: Dict[str, Any], willingness_result: Dict[str, Any]) -> str:
        """
        为“读空气”功能构建发送给LLM的提示词。
        
        Args:
            event: AstrBot的消息事件对象。
            chat_context: 完整的聊天上下文。
            willingness_result: 意愿计算结果。
            
        Returns:
            构建好的提示词字符串。
        """
        user_id = event.get_sender_id()
        message_content = event.message_str
        
        # 获取上下文信息，并提供默认值以防止KeyError
        decision_context = willingness_result.get("decision_context", {})
        user_impression = chat_context.get("user_impression", {})
        conversation_history = chat_context.get("conversation_history", [])
        relevant_memories = chat_context.get("relevant_memories", [])
        
        # 格式化上下文信息，使其在提示词中更易读
        base_willingness = decision_context.get('base_willingness', 0.5)
        user_score = user_impression.get('score', 0.5)
        group_activity = decision_context.get('group_activity', 0.5)
        fatigue_level = decision_context.get('fatigue_level', 0.0)
        interaction_mode = decision_context.get('interaction_mode', 'normal')
        impression_summary = user_impression.get('summary', '无印象信息')
        
        memories_str = "\n".join([f"- {mem.get('content', '')}" for mem in relevant_memories[:3]]) if relevant_memories else "无相关记忆。"
        history_str = "\n".join([f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in conversation_history[-3:]]) if conversation_history else "无最近对话。"

        # 构建提示
        no_reply_marker = getattr(self.config, 'air_reading_no_reply_marker', "[DO_NOT_REPLY]")
        prompt = f"""你是一个拟人化的聊天助手，需要判断是否应该回复以下消息。你的任务是“读空气”，即根据上下文判断当前聊天氛围是否适合回复。

【当前消息】
用户ID: {user_id}
消息内容: {message_content}

【上下文信息】
- 基础回复意愿分数: {base_willingness:.2f} (0-1之间，越高越想回复)
- 用户好感度: {user_score:.2f} (0-1之间，越高表示关系越好)
- 群组活跃度: {group_activity:.2f} (0-1之间，越高表示群越活跃)
- 疲劳度: {fatigue_level:.2f} (0-1之间，越高表示越疲劳，越不想说话)
- 当前交互模式: {interaction_mode} (normal: 普通, focus: 专注, observation: 观察)

【用户印象摘要】
{impression_summary}

【相关记忆】
{memories_str}

【最近对话】
{history_str}

【判断与回复指令】
请根据以上所有信息，判断是否应该回复这条消息。

**如果你认为不应该回复**，请只回复以下标记，不要添加任何其他文字或解释：
{no_reply_marker}

**如果你认为应该回复**，请直接给出你自然、友好的回复内容。

**判断和回复时请综合考虑以下因素：**
1.  **相关性**：消息是否直接与你相关，或是在与你对话？
2.  **必要性**：这条消息是否需要一个回应？
3.  **氛围**：当前的聊天氛围是开放、轻松的，还是严肃、私密的？你的加入是否合适？
4.  **打扰**：你的回复是否会打断别人的重要对话或破坏当前氛围？
5.  **内容**：消息是否有实质性的内容值得回应？（例如，简单的“哈哈哈”或表情包可能不需要回应）

请开始你的判断和回复："""
        
        logger.debug(f"ResponseEngine: 读空气提示词构建完成。长度: {len(prompt)}")
        return prompt


    async def _call_llm_for_air_reading(self, prompt: str, event: Any) -> str:
        """
        调用LLM进行“读空气”决策。
        
        Args:
            prompt: 发送给LLM的提示词。
            
        Returns:
            LLM的原始回复文本。如果调用失败，返回空字符串。
        """
        try:
            provider = self.context.get_using_provider()
            if not provider:
                logger.error("ResponseEngine: 未找到可用的 LLM 提供商，无法进行读空气决策。")
                return ""
            
            logger.debug("ResponseEngine: 正在调用LLM提供商进行读空气...")
            # 使用 AstrBot 的 LLM 调用接口
            # 注意：这里不传入历史对话记录，因为读空气是一个独立的判断过程
            base_sys_prompt = "你是一个极其擅长'读空气'的聊天助手。你的核心任务是判断在特定聊天场景下，回复是否恰当。你需要理解社交暗示、聊天氛围和人际关系，从而做出最合适的决定：回复或保持沉默。"
            persona = await self._resolve_persona_text(event)
            sys_prompt = self._compose_system_prompt_with_persona(base_sys_prompt, persona)
            llm_response = await provider.text_chat(
                prompt=prompt,
                contexts=[], # 读空气是独立判断，不依赖历史对话
                image_urls=[],
                system_prompt=sys_prompt
            )
            
            if llm_response and llm_response.completion_text:
                logger.info(f"ResponseEngine: LLM读空气调用成功。回复: {llm_response.completion_text.strip()}")
                return llm_response.completion_text
            else:
                logger.warning("ResponseEngine: LLM读空气调用成功，但返回内容为空。")
                return ""
            
        except Exception as e:
            logger.error(f"ResponseEngine: LLM 读空气调用过程中发生异常: {e}", exc_info=True)
            return ""  # 出错时默认不回复，保证系统稳定性
    
    async def _generate_normal_response(self, event: Any, chat_context: Dict[str, Any]) -> str:
        """
        在决定需要回复后，调用LLM生成正常的回复内容。
        
        Args:
            event: AstrBot的消息事件对象。
            chat_context: 完整的聊天上下文。
            
        Returns:
            LLM生成的回复内容。如果生成失败，返回空字符串。
        """
        logger.debug("ResponseEngine: 构建正常回复提示词。")
        # 构建传统回复提示
        response_prompt = await self._build_response_prompt(event, chat_context)
        
        try:
            provider = self.context.get_using_provider()
            if not provider:
                logger.error("ResponseEngine: 未找到可用的 LLM 提供商，无法生成回复。")
                return "抱歉，我现在有点问题，稍后再回复你。"
            
            logger.debug("ResponseEngine: 正在调用LLM提供商生成回复...")
            # 在生成正常回复时，可以考虑传入最近的对话历史作为上下文
            # 但要注意，AstrBot的conversation_manager已经处理了对话历史，这里可能不需要重复传入
            # 为了简单和避免上下文过长，这里也选择不传入，让LLM基于当前prompt独立生成
            base_sys_prompt = "你是一个拟人化的聊天助手。你的回复风格应该自然、友好、富有同理心，并且完全符合当前的聊天语境。请避免过于机械或官方的语气。"
            persona = await self._resolve_persona_text(event)
            sys_prompt = self._compose_system_prompt_with_persona(base_sys_prompt, persona)
            llm_response = await provider.text_chat(
                prompt=response_prompt,
                contexts=[],
                image_urls=[],
                system_prompt=sys_prompt
            )
            
            if llm_response and llm_response.completion_text:
                logger.info(f"ResponseEngine: LLM回复生成成功。内容: {llm_response.completion_text.strip()}")
                return llm_response.completion_text
            else:
                logger.warning("ResponseEngine: LLM回复生成成功，但返回内容为空。")
                return ""
            
        except Exception as e:
            logger.error(f"ResponseEngine: 生成正常回复时发生异常: {e}", exc_info=True)
            return "抱歉，我刚才走神了，能再说一遍吗？" # 友好的错误提示
    
    async def _build_response_prompt(self, event: Any, chat_context: Dict[str, Any]) -> str:
        """
        为生成正常回复构建发送给LLM的提示词。
        
        Args:
            event: AstrBot的消息事件对象。
            chat_context: 完整的聊天上下文。
            
        Returns:
            构建好的提示词字符串。
        """
        user_id = event.get_sender_id()
        message_content = event.message_str
        
        # 获取上下文信息，并提供默认值
        user_impression = chat_context.get("user_impression", {})
        conversation_history = chat_context.get("conversation_history", [])
        relevant_memories = chat_context.get("relevant_memories", [])
        
        impression_summary = user_impression.get('summary', '无印象信息')
        memories_str = "\n".join([f"- {mem.get('content', '')}" for mem in relevant_memories[:2]]) if relevant_memories else "无相关记忆。"
        history_str = "\n".join([f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in conversation_history[-2:]]) if conversation_history else "无最近对话。"

        prompt = f"""请根据以下上下文，对收到的消息生成一个自然、友好的回复。

【收到的消息】
用户ID: {user_id}
消息内容: {message_content}

【辅助上下文信息】
- 用户印象摘要: {impression_summary}
- 相关记忆: {memories_str}
- 最近对话片段: {history_str}

【回复要求】
1.  **自然拟人**：你的回复应该像一个真实的人，而不是机器人。可以使用口语化的表达。
2.  **语境贴合**：回复内容需要与当前聊天的主题和氛围保持一致。
3.  **简洁明了**：避免过长或过于复杂的句子，直接回应消息的核心内容。
4.  **富有个性**：如果用户印象中有相关信息（例如，用户喜欢幽默），可以适当融入你的回复风格中。

请开始你的回复："""
        
        logger.debug(f"ResponseEngine: 正常回复提示词构建完成。长度: {len(prompt)}")
        return prompt
