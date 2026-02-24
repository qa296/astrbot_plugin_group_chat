"""
AstrBot 群聊插件 v2.0
基于心流理论的群聊主动对话插件

核心特性：
- 四态心流状态机（观察/沉浸/活跃/疲劳）
- 三路学习机制（策略检索/离线蒸馏/在线学习）
- 能量系统动态调节
- 读空气功能
- 自适应群规模
"""

import asyncio
from typing import Any

from core.decision_engine import DecisionEngine
from core.energy_system import EnergySystem
from core.state_machine import FlowState, FlowStateMachine
from execution.feedback_collector import FeedbackCollector
from execution.response_generator import ResponseGenerator
from execution.timing_controller import TimingController
from learning.offline_distiller import OfflineDistiller
from learning.online_learner import OnlineLearner
from learning.strategy_store import StrategyStore
from perception.activity_meter import ActivityMeter
from perception.context_analyzer import ContextAnalyzer
from perception.topic_tracker import TopicTracker
from storage.persistence import PersistenceManager

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register


@register(
    "astrbot_plugin_group_chat",
    "qa296",
    "一个先进的群聊交互插件，采用AI算法实现智能回复决策，能像真人一样主动参与对话，实现拟人化的主动交互体验。",
    "2.0.0",
    "https://github.com/qa296/astrbot_plugin_group_chat",
)
class GroupChatPlugin(Star):
    """群聊插件主类"""

    def __init__(self, context: Context, config: Any):
        super().__init__(context)
        self.config = config

        logger.info("群聊插件 v2.0 初始化中...")

        # 1. 持久化管理器
        self.persistence = PersistenceManager("astrbot_plugin_group_chat")

        # 2. 存储层
        self.strategy_store = StrategyStore(self.persistence, config)

        # 3. 学习层
        self.online_learner = OnlineLearner(config, self.persistence)
        self.offline_distiller = OfflineDistiller(context, config, self.persistence)

        # 4. 核心层
        self.state_machine = FlowStateMachine(config, self.persistence)
        self.energy_system = EnergySystem(config, self.persistence)
        self.decision_engine = DecisionEngine(
            config, self.strategy_store, self.online_learner, self.offline_distiller
        )

        # 5. 感知层
        self.context_analyzer = ContextAnalyzer(context, config, self.persistence)
        self.activity_meter = ActivityMeter(config, self.persistence)
        self.topic_tracker = TopicTracker(config, self.persistence)

        # 6. 执行层
        self.response_generator = ResponseGenerator(context, config)
        self.timing_controller = TimingController(config)
        self.feedback_collector = FeedbackCollector(config, self.persistence)

        # 后台任务
        self._heartbeat_task = None
        self._distillation_task = None
        self._save_task = None
        self._running = False

        # 统计信息
        self._stats = {
            "messages_processed": 0,
            "replies_sent": 0,
            "proactive_messages": 0,
            "feedbacks_collected": 0,
        }

        logger.info("群聊插件初始化完成")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """AstrBot 初始化完成后启动后台任务"""
        self._running = True

        # 启动心跳循环
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # 启动离线蒸馏循环
        learning_config = self.config.get("learning", {})
        if learning_config.get("enable_offline_distillation", True):
            self._distillation_task = asyncio.create_task(self._distillation_loop())

        # 启动定期保存任务
        self._save_task = asyncio.create_task(self._periodic_save())

        logger.info("群聊插件后台任务已启动")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群聊消息的主入口"""
        group_id = event.get_group_id()
        user_id = event.get_sender_id()

        # 1. 权限检查
        if not self._check_group_permission(group_id):
            return

        self._stats["messages_processed"] += 1

        # 2. 记录 UMO
        self.persistence.set_group_umo(group_id, event.unified_msg_origin)

        # 3. 记录活跃度
        self.activity_meter.record_message(group_id, user_id)

        # 4. 感知层：分析上下文
        context = await self.context_analyzer.analyze(event)

        # 补充活跃度指标
        activity_metrics = self.activity_meter.measure(group_id)
        context.group_activity = activity_metrics.overall_activity

        # 5. 话题追踪
        topic_result = self.topic_tracker.track(group_id, event.message_str, user_id)
        context.topic_coherence = topic_result.topic_coherence

        # 6. 状态机：处理消息并更新状态
        self.state_machine.on_message(event, context)

        # 7. 获取群组状态和能量
        state = self.state_machine.get_state(group_id)
        energy = self.energy_system.get_energy(group_id)

        # 8. 时间恢复能量
        self.energy_system.time_based_recovery(
            group_id, state.flow_state.value, context.group_activity
        )

        # 9. 被@时恢复能量
        if context.is_at_bot:
            self.energy_system.recover_on_at(group_id)

        # 10. 决策引擎：决定是否回复
        decision = await self.decision_engine.decide(state, context, energy)

        # 11. 时机控制
        timing = self.timing_controller.should_reply_now(
            group_id, state.flow_state.value, energy, context.group_activity
        )

        # 12. 执行层：生成并发送回复
        if decision.should_act and timing.should_reply_now:
            # 添加延迟
            if decision.delay_seconds > 0:
                await asyncio.sleep(decision.delay_seconds)

            # 生成回复
            response = await self.response_generator.generate(event, context, decision)

            if response:
                yield event.plain_result(response)

                # 更新统计
                self._stats["replies_sent"] += 1

                # 更新状态机
                self.state_machine.on_reply_sent(group_id, len(response))

                # 消耗能量
                self.energy_system.consume(group_id, len(response), state.reply_streak)

                # 记录时机
                self.timing_controller.record_reply(group_id)

                # 添加到对话历史
                self.context_analyzer.add_bot_reply_to_history(group_id, response)

                # 注册反馈收集
                self.feedback_collector.register_pending_feedback(
                    group_id, event, response
                )

                # 记录决策元数据
                decision.metadata["state_vec"] = self.decision_engine._encode_state(
                    state, context
                )

                logger.debug(
                    f"群组 {group_id} 回复成功: "
                    f"state={state.flow_state.value}, energy={energy:.2f}, "
                    f"decision={decision.action.value}"
                )

        # 13. 检查是否有反馈
        feedback = self.feedback_collector.check_message_for_feedback(
            group_id, user_id, event.message_str, context.is_at_bot
        )

        if feedback:
            self._handle_feedback(group_id, feedback, decision)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM 响应钩子：处理读空气标记"""
        try:
            if resp.role != "assistant":
                return

            # 检查读空气标记
            air_config = self.config.get("air_reading", {})
            if air_config.get("enabled", True):
                marker = air_config.get("no_reply_marker", "<NO_RESPONSE>")
                if marker in (resp.completion_text or ""):
                    logger.debug("检测到读空气标记，阻止消息发送")
                    event.clear_result()

        except Exception as e:
            logger.error(f"处理 LLM 响应钩子时发生错误: {e}")

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """消息发送后钩子"""
        pass

    @filter.command("心流状态")
    async def flow_status(self, event: AstrMessageEvent):
        """显示当前群的心流状态"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("请在群聊中使用此命令。")
            return

        # 获取状态
        state = self.state_machine.get_state(group_id)
        energy = self.energy_system.get_energy(group_id)
        timing_stats = self.timing_controller.get_stats(group_id)
        strategy_stats = self.strategy_store.get_stats(group_id)
        decision_stats = self.decision_engine.get_stats()
        feedback_stats = self.feedback_collector.get_stats()

        # 活跃度
        activity_metrics = self.activity_meter.measure(group_id)

        # 构建消息
        msg = (
            f"📊 心流状态面板\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💠 心流状态: {state.flow_state.value}\n"
            f"⚡ 能量值: {energy:.2f}\n"
            f"🔄 连续回复: {state.reply_streak}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📈 群活跃度: {activity_metrics.overall_activity:.2f}\n"
            f"💬 每分钟消息: {activity_metrics.messages_per_minute:.1f}\n"
            f"👥 活跃用户: {activity_metrics.active_users_count}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏱️ 冷却剩余: {timing_stats['cooldown_remaining']:.1f}s\n"
            f"📚 策略命中率: {strategy_stats['hit_rate']:.1%}\n"
            f"🎯 决策统计: {decision_stats['total_decisions']}\n"
            f"📨 反馈收集: {feedback_stats['total_feedbacks']}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📤 发送消息: {self._stats['replies_sent']}\n"
            f"📨 处理消息: {self._stats['messages_processed']}"
        )

        yield event.plain_result(msg)

    @filter.command("心流调试")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def flow_debug(
        self, event: AstrMessageEvent, action: str = "", value: str = ""
    ):
        """调试命令（仅管理员）"""
        group_id = event.get_group_id()
        if not group_id:
            yield event.plain_result("请在群聊中使用此命令。")
            return

        if not action:
            yield event.plain_result(
                "用法: /心流调试 <action> [value]\n"
                "actions: state, energy, cooldown, reset"
            )
            return

        action = action.lower()

        if action == "state":
            # 强制设置状态
            if value in ["observer", "flow", "active", "fatigue"]:
                self.state_machine.force_state(group_id, FlowState(value))
                yield event.plain_result(f"状态已设置为: {value}")
            else:
                yield event.plain_result(
                    "无效状态，可选: observer, flow, active, fatigue"
                )

        elif action == "energy":
            # 设置能量
            try:
                energy_value = float(value)
                self.energy_system.set_energy(group_id, energy_value)
                yield event.plain_result(f"能量已设置为: {energy_value:.2f}")
            except ValueError:
                yield event.plain_result("请提供有效的数值")

        elif action == "cooldown":
            # 重置冷却
            self.timing_controller.reset_cooldown(group_id)
            yield event.plain_result("冷却已重置")

        elif action == "reset":
            # 完全重置
            self.state_machine.force_state(group_id, FlowState.OBSERVER)
            self.energy_system.reset_energy(group_id)
            self.timing_controller.reset_cooldown(group_id)
            yield event.plain_result("群组状态已完全重置")

        else:
            yield event.plain_result(f"未知操作: {action}")

    def _check_group_permission(self, group_id: str) -> bool:
        """检查群组权限"""
        list_mode = self.config.get("list_mode", "blacklist")
        groups = self.config.get("groups", [])

        if list_mode == "blacklist":
            return group_id not in groups
        else:
            return group_id in groups

    def _handle_feedback(self, group_id: str, feedback, decision):
        """处理反馈"""
        self._stats["feedbacks_collected"] += 1

        # 更新能量
        if feedback.reward > 0:
            self.energy_system.recover_on_positive_feedback(group_id)
        else:
            self.energy_system.penalty_on_negative_feedback(group_id)

        # 更新状态机
        self.state_machine.on_user_feedback(group_id, feedback.reward > 0)

        # 记录决策结果
        self.decision_engine.record_outcome(
            decision, feedback.reward, feedback.feedback_type.value
        )

        logger.debug(
            f"群组 {group_id} 收到反馈: type={feedback.feedback_type.value}, "
            f"reward={feedback.reward:.2f}"
        )

    async def _heartbeat_loop(self):
        """心跳循环：主动消息触发"""
        timing_config = self.config.get("timing", {})
        interval = timing_config.get("heartbeat_interval_seconds", 15.0)

        while self._running:
            try:
                await asyncio.sleep(interval)

                # 遍历所有活跃群组
                for group_id in self.persistence.get_active_groups():
                    try:
                        # 检查状态转换
                        transition = self.state_machine.on_timeout(group_id)

                        if transition.should_trigger:
                            await self._trigger_proactive_message(group_id)

                    except Exception as e:
                        logger.error(f"心跳处理群组 {group_id} 时出错: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳循环异常: {e}")

        logger.info("心跳循环已停止")

    async def _trigger_proactive_message(self, group_id: str):
        """触发主动消息"""
        # 获取 UMO
        umo = self.persistence.get_group_umo(group_id)
        if not umo:
            return

        # 获取状态
        state = self.state_machine.get_state(group_id)
        energy = self.energy_system.get_energy(group_id)

        # 分析上下文
        context = await self.context_analyzer.analyze_proactive(group_id)

        # 活跃度
        activity_metrics = self.activity_meter.measure(group_id)
        context.group_activity = activity_metrics.overall_activity

        # 决策
        decision = await self.decision_engine.decide(state, context, energy)

        if not decision.should_act:
            return

        # 时机检查
        timing = self.timing_controller.should_reply_now(
            group_id, state.flow_state.value, energy, context.group_activity
        )

        if not timing.should_reply_now:
            return

        # 生成主动消息
        response = await self.response_generator.generate_proactive(
            group_id, context, decision
        )

        if not response:
            return

        try:
            # 发送消息
            chain = MessageChain().message(response)
            await self.context.send_message(umo, chain)

            # 更新统计和状态
            self._stats["proactive_messages"] += 1
            self.state_machine.on_reply_sent(group_id, len(response))
            self.energy_system.consume(group_id, len(response))
            self.timing_controller.record_reply(group_id)
            self.context_analyzer.add_bot_reply_to_history(group_id, response)

            logger.info(f"群组 {group_id} 主动消息发送成功")

        except Exception as e:
            logger.error(f"发送主动消息失败: {e}")

    async def _distillation_loop(self):
        """离线蒸馏循环"""
        learning_config = self.config.get("learning", {})
        interval_hours = learning_config.get("distillation_interval_hours", 24)

        while self._running:
            try:
                await asyncio.sleep(interval_hours * 3600)

                logger.info("开始执行离线蒸馏...")
                rules_count = await self.offline_distiller.distill()
                logger.info(f"离线蒸馏完成，生成 {rules_count} 条规则")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"离线蒸馏异常: {e}")

        logger.info("离线蒸馏循环已停止")

    async def _periodic_save(self):
        """定期保存"""
        while self._running:
            try:
                await asyncio.sleep(300)  # 每5分钟保存一次
                self.persistence.save_dirty()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期保存异常: {e}")

        logger.info("定期保存任务已停止")

    async def terminate(self):
        """插件终止"""
        logger.info("群聊插件正在终止...")

        self._running = False

        # 取消后台任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._distillation_task:
            self._distillation_task.cancel()
        if self._save_task:
            self._save_task.cancel()

        # 保存所有数据
        self.persistence.save_all()

        logger.info(
            f"群聊插件已终止。"
            f"统计: 处理消息 {self._stats['messages_processed']} 条, "
            f"发送回复 {self._stats['replies_sent']} 条, "
            f"主动消息 {self._stats['proactive_messages']} 条"
        )
