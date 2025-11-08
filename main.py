from typing import Dict, List, Optional, Any, AsyncGenerator
import sys
import os
import time
from pathlib import Path

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger

# 添加src目录到Python路径 - 使用更安全的方式
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 导入自定义模块
from active_chat_manager import ActiveChatManager
from group_list_manager import GroupListManager
from impression_manager import ImpressionManager
from memory_integration import MemoryIntegration
from interaction_manager import InteractionManager
from response_engine import ResponseEngine
from willingness_calculator import WillingnessCalculator
from focus_chat_manager import FocusChatManager
from fatigue_system import FatigueSystem
from context_analyzer import ContextAnalyzer
from state_manager import StateManager

@register("astrbot_plugin_group_chat", "qa296", "一个先进的群聊交互插件，采用AI算法实现智能回复决策，能像真人一样主动参与对话，实现拟人化的主动交互体验。", "1.0.3", "https://github.com/qa296/astrbot_plugin_group_chat")
class GroupChatPlugin(Star):
    _instance = None
    def __init__(self, context: Context, config: Any):
        super().__init__(context)
        self.config = config
        # 记录实例用于静态包装器访问
        GroupChatPlugin._instance = self
        
        # 初始化状态管理器（符合文档要求的持久化存储）
        self.state_manager = StateManager(context, config)
        
        # 初始化组件
        self.group_list_manager = GroupListManager(config)
        self.impression_manager = ImpressionManager(context, config)
        self.memory_integration = MemoryIntegration(context, config)
        self.interaction_manager = InteractionManager(context, config, self.state_manager)
        self.response_engine = ResponseEngine(context, config)
        self.willingness_calculator = WillingnessCalculator(context, config, self.impression_manager, self.state_manager)
        self.focus_chat_manager = FocusChatManager(context, config, self.state_manager)
        self.fatigue_system = FatigueSystem(config, self.state_manager)
        self.context_analyzer = ContextAnalyzer(context, config, self.state_manager, self.impression_manager, self.memory_integration)
        
        # 初始化主动聊天管理器（注入依赖）
        self.active_chat_manager = ActiveChatManager(
            context,
            self.state_manager,
            response_engine=self.response_engine,
            context_analyzer=self.context_analyzer,
            willingness_calculator=self.willingness_calculator,
            plugin_config=self.config
        )
        
        logger.info("群聊插件初始化完成")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """AstrBot 初始化完成后启动主动聊天管理器。"""
        self.active_chat_manager.start_all_flows()
    
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群聊消息的主入口"""
        group_id = event.get_group_id()
        
        # 1. 群组权限检查
        if not self.group_list_manager.check_group_permission(group_id):
            return
        
        # 记录会话标识并确保该群心跳存在
        self.state_manager.set_group_umo(group_id, event.unified_msg_origin)
        self.active_chat_manager.ensure_flow(group_id)
        # 将消息传递给 ActiveChatManager 以进行频率分析
        if group_id in self.active_chat_manager.group_flows:
            self.active_chat_manager.group_flows[group_id].on_message(event)
        
        # 2. 处理消息
        async for result in self._process_group_message(event):
            yield result
    
    async def _process_group_message(self, event: AstrMessageEvent):
        """处理群聊消息的核心逻辑"""
        group_id = event.get_group_id()
        user_id = event.get_sender_id()
        
        # 获取聊天上下文
        chat_context = await self.context_analyzer.analyze_chat_context(event)
        
        # 判断交互模式
        interaction_mode = self.interaction_manager.determine_interaction_mode(chat_context)
        
        # 观察模式不回复
        if interaction_mode == "observation":
            return
        
        # 计算回复意愿
        willingness_result = await self.willingness_calculator.calculate_response_willingness(event, chat_context)
        
        # 如果不需要 LLM 决策且意愿不足，直接跳过
        if not willingness_result.get("requires_llm_decision") and not willingness_result.get("should_respond"):
            return
        
        # 检查连续回复限制
        max_consecutive = getattr(self.config, 'max_consecutive_responses', 3)
        consecutive_count = self.state_manager.get_consecutive_responses().get(group_id, 0)
        if consecutive_count >= max_consecutive:
            return
        
        # 生成回复（包含读空气功能）
        response_result = await self.response_engine.generate_response(event, chat_context, willingness_result)
        
        # 根据结果决定是否回复
        if response_result.get("should_reply"):
            response_content = response_result.get("content")
            if response_content:
                yield event.plain_result(response_content)

                # 更新连续回复计数
                self.state_manager.increment_consecutive_response(group_id)

                # 心流算法：回复成功后更新状态
                await self.willingness_calculator.on_bot_reply_update(event, len(response_content))

                # 记录决策信息（用于调试）
                decision_method = response_result.get("decision_method")
                willingness_score = response_result.get("willingness_score")
                logger.debug(f"群组 {group_id} 回复 - 方法: {decision_method}, 意愿分: {willingness_score:.2f}")
        else:
            # 记录跳过回复的原因
            decision_method = response_result.get("decision_method")
            skip_reason = response_result.get("skip_reason", "意愿不足")
            willingness_score = response_result.get("willingness_score")
            logger.debug(f"群组 {group_id} 跳过回复 - 方法: {decision_method}, 原因: {skip_reason}, 意愿分: {willingness_score:.2f}")
        
        # 更新交互状态
        await self.interaction_manager.update_interaction_state(event, chat_context, response_result)

    # 读空气功能：处理LLM回复，进行文本过滤
    from astrbot.api.provider import LLMResponse
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """处理大模型回复，进行文本过滤"""
        try:
            if resp.role != "assistant":
                return
            # 这里可以添加文本过滤逻辑，目前保持简单
            # resp.completion_text = self._filter_text(resp.completion_text)
        except Exception as e:
            logger.error(f"处理LLM回复时发生错误: {e}")

    # 读空气功能：在消息发送前检查是否包含<NO_RESPONSE>标记
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """在消息发送前处理读空气功能"""
        try:
            result = event.get_result()
            if result is None or not result.chain:
                return

            # 检查是否为LLM结果且包含不回复标记
            if result.is_llm_result():
                # 获取消息文本内容
                message_text = ""
                for comp in result.chain:
                    if hasattr(comp, 'text'):
                        message_text += comp.text

                # 兼容多种不回复标记（包括配置项与历史写法）
                cfg_marker = getattr(self.config, 'air_reading_no_reply_marker', None)
                markers = [m for m in [cfg_marker, "<NO_RESPONSE>", "[DO_NOT_REPLY]"] if m]

                # 如果包含任一不回复标记，清空事件结果以阻止消息发送
                if any(m in message_text for m in markers):
                    logger.debug("检测到读空气不回复标记，阻止消息发送")
                    event.clear_result()
                    logger.debug("已清空事件结果，消息发送被阻止")

        except Exception as e:
            logger.error(f"处理消息发送前事件时发生错误: {e}")
    
    @filter.command("群聊主动状态")
    async def gcstatus(self, event: AstrMessageEvent):
        """显示当前群的主动对话状态"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("请在群聊中使用此命令。")
            return

        # 确保会话映射与心跳存在
        self.state_manager.set_group_umo(group_id, event.unified_msg_origin)
        self.active_chat_manager.ensure_flow(group_id)

        stats = self.active_chat_manager.get_stats(group_id)

        has_flow = "✅" if stats.get("has_flow") else "❌"
        has_umo = "✅" if stats.get("has_umo") else "❌"
        focus = float(stats.get("focus", 0.0) or 0.0)
        at_boost = float(stats.get("at_boost", 0.0) or 0.0)
        effective = float(stats.get("effective", 0.0) or 0.0)
        mlm = int(stats.get("messages_last_minute", 0) or 0)
        cd = float(stats.get("cooldown_remaining", 0.0) or 0.0)
        last_ts = float(stats.get("last_trigger_ts", 0.0) or 0.0)
        elapsed = (time.time() - last_ts) if last_ts > 0 else 0.0

        # 配置值
        hb_thr = float(getattr(self.config, "heartbeat_threshold", 0.55) or 0.55)
        wil_thr = float(getattr(self.config, "willingness_threshold", 0.5) or 0.5)
        obs_thr = float(getattr(self.config, "observation_mode_threshold", 0.2) or 0.2)
        min_interest = float(getattr(self.config, "min_interest_score", 0.6) or 0.6)
        at_boost_cfg = float(getattr(self.config, "at_boost_value", 0.5) or 0.5)

        # 计算意愿分与群活跃度
        chat_context = await self.context_analyzer.analyze_chat_context(event)
        will_res = await self.willingness_calculator.calculate_response_willingness(event, chat_context)
        willingness_score = float(will_res.get("willingness_score", 0.0) or 0.0)
        decision_ctx = will_res.get("decision_context", {}) or {}
        group_activity = float(decision_ctx.get("group_activity", 0.0) or 0.0)

        # 评估专注兴趣度
        try:
            interest = float(await self.focus_chat_manager.evaluate_focus_interest(event, chat_context))
        except Exception:
            interest = 0.0

        # 从 flow 读取心跳/冷却常量（回退到默认）
        flow = self.active_chat_manager.group_flows.get(group_id)
        hb_int = getattr(flow, "HEARTBEAT_INTERVAL", 15) if flow else 15
        cd_total = getattr(flow, "COOLDOWN_SECONDS", 120) if flow else 120

        msg = (
            "主动对话状态\n"
            f"心跳: {has_flow}    UMO: {has_umo}\n"
            f"最近1分钟消息: {mlm}\n"
            f"焦点: {focus:.2f}\n"
            f"@增强(当前/设定): {at_boost:.2f} / {at_boost_cfg:.2f}\n"
            f"心跳: ({effective:.2f}/{hb_thr:.2f})\n"
            f"意愿: ({willingness_score:.2f}/{wil_thr:.2f})\n"
            f"观察: ({group_activity:.2f}/{obs_thr:.2f})\n"
            f"专注: ({interest:.2f}/{min_interest:.2f})\n"
            f"冷却剩余: {cd:.1f}s  上次触发: {elapsed:.1f}s\n"
            f"心跳/冷却: {hb_int}s / {cd_total}s"
        )
        yield event.plain_result(msg)

    async def terminate(self):
        """插件终止时的清理工作"""
        logger.info("群聊插件正在终止...")
        # 停止主动聊天管理器
        self.active_chat_manager.stop_all_flows()
        # 使用状态管理器清理所有持久化状态
        self.state_manager.clear_all_state()
        logger.info("群聊插件已终止")
