"""
心流状态机

基于心流理论的四态状态机：观察态(OBSERVER) -> 沉浸态(FLOW) -> 活跃态(ACTIVE) -> 疲劳态(FATIGUE)

状态转换规则：
- OBSERVER -> FLOW: 群活跃度超过阈值
- FLOW -> ACTIVE: 与机器人相关性超过阈值
- ACTIVE -> FATIGUE: 连续回复次数超过限制
- FATIGUE -> OBSERVER: 恢复时间结束
- 任意状态 -> OBSERVER: 长时间无活动
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from astrbot.api import logger


class FlowState(Enum):
    """心流状态枚举"""

    OBSERVER = "observer"  # 观察态：静默观察，仅@触发
    FLOW = "flow"  # 沉浸态：适度参与，跟随话题
    ACTIVE = "active"  # 活跃态：高意愿回复，短冷却
    FATIGUE = "fatigue"  # 疲劳态：强制休息，恢复能量


@dataclass
class GroupState:
    """群组状态"""

    group_id: str
    flow_state: FlowState = FlowState.OBSERVER
    energy: float = 0.8
    last_reply_ts: float = 0.0
    reply_streak: int = 0
    last_message_ts: float = 0.0
    fatigue_start_ts: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "flow_state": self.flow_state.value,
            "energy": self.energy,
            "last_reply_ts": self.last_reply_ts,
            "reply_streak": self.reply_streak,
            "last_message_ts": self.last_message_ts,
            "fatigue_start_ts": self.fatigue_start_ts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroupState":
        return cls(
            group_id=data["group_id"],
            flow_state=FlowState(data["flow_state"]),
            energy=data.get("energy", 0.8),
            last_reply_ts=data.get("last_reply_ts", 0.0),
            reply_streak=data.get("reply_streak", 0),
            last_message_ts=data.get("last_message_ts", 0.0),
            fatigue_start_ts=data.get("fatigue_start_ts", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StateTransition:
    """状态转换结果"""

    from_state: FlowState
    to_state: FlowState
    should_trigger: bool = False
    trigger_reason: str = ""
    should_reply: bool = False
    reply_probability: float = 0.0

    def is_transition(self) -> bool:
        return self.from_state != self.to_state


class FlowStateMachine:
    """
    心流状态机

    管理每个群组的心流状态，根据上下文信息进行状态转换
    """

    def __init__(self, config, persistence):
        self.config = config
        self.persistence = persistence

        # 内存缓存群组状态
        self._states: dict[str, GroupState] = {}

        # 加载持久化状态
        self._load_states()

        # 配置参数
        sm_config = config.get("state_machine", {})
        self.observer_to_flow_threshold = sm_config.get(
            "observer_to_flow_threshold", 0.4
        )
        self.flow_to_active_threshold = sm_config.get("flow_to_active_threshold", 0.7)
        self.max_reply_streak = sm_config.get("max_reply_streak", 5)
        self.fatigue_recovery_minutes = sm_config.get("fatigue_recovery_minutes", 5)

        # 时机配置
        timing_config = config.get("timing", {})
        self.base_cooldown = timing_config.get("base_cooldown_seconds", 45.0)

        logger.info(f"心流状态机初已加载 {len(self._states)} 个群组状态")

    def _load_states(self):
        """从持久化加载状态"""
        for group_id in self.persistence.get_all_group_ids():
            state_data = self.persistence.get_group_state(group_id)
            if state_data:
                self._states[group_id] = GroupState.from_dict(
                    {
                        "group_id": state_data.group_id,
                        "flow_state": state_data.flow_state,
                        "energy": state_data.energy,
                        "last_reply_ts": state_data.last_reply_ts,
                        "reply_streak": state_data.reply_streak,
                        "last_message_ts": state_data.last_message_ts,
                        "fatigue_start_ts": state_data.fatigue_start_ts,
                        "metadata": state_data.metadata
                        if hasattr(state_data, "metadata")
                        else {},
                    }
                )

    def get_state(self, group_id: str) -> GroupState:
        """获取群组状态，不存在则创建"""
        if group_id not in self._states:
            self._states[group_id] = GroupState(group_id=group_id)
        return self._states[group_id]

    def on_message(self, event, context) -> StateTransition:
        """
        处理消息事件

        Args:
            event: 消息事件
            context: 分析后的上下文（AnalyzedContext）

        Returns:
            StateTransition: 状态转换结果
        """
        group_id = event.get_group_id()
        current_time = time.time()

        state = self.get_state(group_id)
        from_state = state.flow_state

        # 更新最后消息时间
        state.last_message_ts = current_time

        # 获取上下文信息
        group_activity = getattr(context, "group_activity", 0.0)
        relevance_to_bot = getattr(context, "relevance_to_bot", 0.0)
        is_at_bot = relevance_to_bot >= 0.9

        # 计算状态转换
        transition = self._calculate_transition(
            state, group_activity, relevance_to_bot, is_at_bot, current_time
        )

        # 应用转换
        if transition.is_transition():
            self._apply_transition(state, transition, current_time)
            logger.debug(
                f"群组 {group_id} 状态转换: {from_state.value} -> {transition.to_state.value}, "
                f"原因: {transition.trigger_reason}"
            )

        return transition

    def on_timeout(self, group_id: str) -> StateTransition:
        """
        处理超时事件（心跳触发）

        Returns:
            StateTransition: 状态转换结果，包含是否应触发主动消息
        """
        current_time = time.time()
        state = self.get_state(group_id)
        from_state = state.flow_state

        # 检查疲劳恢复
        if state.flow_state == FlowState.FATIGUE:
            recovery_time = self.fatigue_recovery_minutes * 60
            if current_time - state.fatigue_start_ts >= recovery_time:
                transition = StateTransition(
                    from_state=from_state,
                    to_state=FlowState.OBSERVER,
                    should_trigger=False,
                    trigger_reason="疲劳恢复完成",
                )
                self._apply_transition(state, transition, current_time)
                return transition
            else:
                return StateTransition(
                    from_state=from_state,
                    to_state=from_state,
                    should_trigger=False,
                    trigger_reason="疲劳恢复中",
                )

        # 检查冷却时间
        time_since_last_reply = current_time - state.last_reply_ts
        if time_since_last_reply < self.base_cooldown:
            return StateTransition(
                from_state=from_state,
                to_state=from_state,
                should_trigger=False,
                trigger_reason="冷却中",
            )

        # 根据当前状态决定是否触发主动消息
        should_trigger = False
        trigger_reason = ""

        if state.flow_state == FlowState.ACTIVE:
            # 活跃态：高概率触发
            if state.energy >= 0.5:
                should_trigger = True
                trigger_reason = "活跃态-能量充足"
        elif state.flow_state == FlowState.FLOW:
            # 沉浸态：中等概率触发
            if state.energy >= 0.6:
                should_trigger = True
                trigger_reason = "沉浸态-适度参与"
        elif state.flow_state == FlowState.OBSERVER:
            # 观察态：低概率触发，需要较高能量
            if state.energy >= 0.8:
                should_trigger = True
                trigger_reason = "观察态-尝试激活"

        return StateTransition(
            from_state=from_state,
            to_state=state.flow_state,
            should_trigger=should_trigger,
            trigger_reason=trigger_reason,
        )

    def _calculate_transition(
        self,
        state: GroupState,
        group_activity: float,
        relevance_to_bot: float,
        is_at_bot: bool,
        current_time: float,
    ) -> StateTransition:
        """计算状态转换"""
        from_state = state.flow_state

        # 疲劳态特殊处理：只能恢复到观察态
        if from_state == FlowState.FATIGUE:
            recovery_time = self.fatigue_recovery_minutes * 60
            if current_time - state.fatigue_start_ts >= recovery_time:
                return StateTransition(
                    from_state=from_state,
                    to_state=FlowState.OBSERVER,
                    trigger_reason="疲劳恢复完成",
                )
            return StateTransition(
                from_state=from_state,
                to_state=from_state,
                should_reply=False,
                reply_probability=0.0,
            )

        # 被@直接进入活跃态
        if is_at_bot:
            return StateTransition(
                from_state=from_state,
                to_state=FlowState.ACTIVE,
                should_reply=True,
                reply_probability=1.0,
                trigger_reason="被@直接激活",
            )

        # 根据活跃度和相关性计算转换
        if from_state == FlowState.OBSERVER:
            if group_activity >= self.observer_to_flow_threshold:
                return StateTransition(
                    from_state=from_state,
                    to_state=FlowState.FLOW,
                    trigger_reason=f"群活跃度达标({group_activity:.2f})",
                )
            # 观察态低回复概率
            return StateTransition(
                from_state=from_state,
                to_state=from_state,
                should_reply=False,
                reply_probability=0.1 * relevance_to_bot,
            )

        elif from_state == FlowState.FLOW:
            if relevance_to_bot >= self.flow_to_active_threshold:
                return StateTransition(
                    from_state=from_state,
                    to_state=FlowState.ACTIVE,
                    should_reply=True,
                    reply_probability=0.8,
                    trigger_reason=f"相关性达标({relevance_to_bot:.2f})",
                )
            elif group_activity < self.observer_to_flow_threshold * 0.5:
                # 活跃度下降，退回观察态
                return StateTransition(
                    from_state=from_state,
                    to_state=FlowState.OBSERVER,
                    trigger_reason="群活跃度下降",
                )
            # 沉浸态中等回复概率
            return StateTransition(
                from_state=from_state,
                to_state=from_state,
                should_reply=relevance_to_bot > 0.3,
                reply_probability=0.3 + 0.3 * relevance_to_bot,
            )

        elif from_state == FlowState.ACTIVE:
            # 检查是否需要进入疲劳态
            if state.reply_streak >= self.max_reply_streak:
                return StateTransition(
                    from_state=from_state,
                    to_state=FlowState.FATIGUE,
                    trigger_reason=f"连续回复次数达到上限({state.reply_streak})",
                )
            # 活跃态高回复概率
            return StateTransition(
                from_state=from_state,
                to_state=from_state,
                should_reply=True,
                reply_probability=0.7 + 0.2 * relevance_to_bot,
            )

        # 默认不转换
        return StateTransition(
            from_state=from_state, to_state=from_state, should_reply=False
        )

    def _apply_transition(
        self, state: GroupState, transition: StateTransition, current_time: float
    ):
        """应用状态转换"""
        state.flow_state = transition.to_state

        if transition.to_state == FlowState.FATIGUE:
            state.fatigue_start_ts = current_time
            state.reply_streak = 0

        elif transition.to_state == FlowState.OBSERVER:
            state.reply_streak = 0

        # 持久化状态
        self._persist_state(state)

    def on_reply_sent(self, group_id: str, reply_length: int):
        """
        回复发送后的状态更新

        Args:
            group_id: 群组ID
            reply_length: 回复长度
        """
        state = self.get_state(group_id)
        current_time = time.time()

        state.last_reply_ts = current_time
        state.reply_streak += 1

        # 能量消耗
        energy_config = self.config.get("energy_system", {})
        cost_base = energy_config.get("energy_cost_base", 0.1)
        cost_per_char = energy_config.get("energy_cost_per_char", 0.0005)

        energy_cost = cost_base + reply_length * cost_per_char
        state.energy = max(0.1, state.energy - energy_cost)

        # 持久化
        self._persist_state(state)

        logger.debug(
            f"群组 {group_id} 回复后状态更新: "
            f"streak={state.reply_streak}, energy={state.energy:.2f}"
        )

    def on_user_feedback(self, group_id: str, is_positive: bool):
        """
        用户反馈后的状态更新

        Args:
            group_id: 群组ID
            is_positive: 是否为正面反馈
        """
        state = self.get_state(group_id)

        if is_positive:
            # 正面反馈：恢复能量，减少连续计数
            state.energy = min(1.0, state.energy + 0.1)
            state.reply_streak = max(0, state.reply_streak - 1)
        else:
            # 负面反馈：降低能量
            state.energy = max(0.1, state.energy - 0.15)

        self._persist_state(state)

    def on_energy_recover(self, group_id: str, amount: float):
        """
        能量恢复

        Args:
            group_id: 群组ID
            amount: 恢复量
        """
        state = self.get_state(group_id)
        state.energy = min(1.0, state.energy + amount)
        self._persist_state(state)

    def _persist_state(self, state: GroupState):
        """持久化状态"""
        from storage.persistence import GroupStateData

        state_data = GroupStateData(
            group_id=state.group_id,
            flow_state=state.flow_state.value,
            energy=state.energy,
            last_reply_ts=state.last_reply_ts,
            reply_streak=state.reply_streak,
            last_message_ts=state.last_message_ts,
            fatigue_start_ts=state.fatigue_start_ts,
        )
        self.persistence.update_group_state(state_data)

    def get_all_active_groups(self) -> dict[str, GroupState]:
        """获取所有活跃群组状态"""
        return {
            gid: state
            for gid, state in self._states.items()
            if state.flow_state != FlowState.OBSERVER or state.energy > 0.5
        }

    def force_state(self, group_id: str, new_state: FlowState):
        """强制设置状态（用于调试/管理）"""
        state = self.get_state(group_id)
        state.flow_state = new_state
        self._persist_state(state)
        logger.info(f"群组 {group_id} 状态已强制设置为 {new_state.value}")
