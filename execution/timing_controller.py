"""
时机控制器

控制回复时机：
- 冷却时间计算
- 回复延迟
- 时间节奏
- 动态调整
"""

import random
import time
from dataclasses import dataclass

from astrbot.api import logger


@dataclass
class TimingDecision:
    """时机决策"""

    should_reply_now: bool
    delay_seconds: float
    cooldown_remaining: float
    reason: str


class TimingController:
    """
    时机控制器

    控制回复的时机
    """

    def __init__(self, config):
        self.config = config

        # 配置
        timing_config = config.get("timing", {})
        self.base_cooldown = timing_config.get("base_cooldown_seconds", 45.0)
        self.min_delay = timing_config.get("min_reply_delay", 1.0)
        self.max_delay = timing_config.get("max_reply_delay", 10.0)
        self.heartbeat_interval = timing_config.get("heartbeat_interval_seconds", 15.0)

        # 状态
        self._last_reply_times: dict[str, float] = {}
        self._cooldown_adjustments: dict[str, float] = {}



    def should_reply_now(
        self, group_id: str, flow_state: str, energy: float, group_activity: float
    ) -> TimingDecision:
        """
        判断是否应该立即回复

        Args:
            group_id: 群组ID
            flow_state: 心流状态
            energy: 能量值
            group_activity: 群活跃度

        Returns:
            TimingDecision: 时机决策
        """
        current_time = time.time()

        # 计算冷却时间
        cooldown = self._calc_cooldown(group_id, flow_state, group_activity)

        # 检查冷却
        last_reply = self._last_reply_times.get(group_id, 0)
        time_since_last = current_time - last_reply
        cooldown_remaining = max(0, cooldown - time_since_last)

        if cooldown_remaining > 0:
            return TimingDecision(
                should_reply_now=False,
                delay_seconds=cooldown_remaining,
                cooldown_remaining=cooldown_remaining,
                reason=f"冷却中，剩余 {cooldown_remaining:.1f} 秒",
            )

        # 计算延迟
        delay = self._calc_delay(flow_state, energy, group_activity)

        return TimingDecision(
            should_reply_now=True,
            delay_seconds=delay,
            cooldown_remaining=0,
            reason="可以回复",
        )

    def get_delay(self, flow_state: str, energy: float, relevance: float) -> float:
        """
        获取回复延迟

        Args:
            flow_state: 心流状态
            energy: 能量值
            relevance: 相关性

        Returns:
            延迟秒数
        """
        return self._calc_delay(flow_state, energy, relevance)

    def calc_cooldown(
        self, group_id: str, flow_state: str, group_activity: float
    ) -> float:
        """
        计算冷却时间

        Args:
            group_id: 群组ID
            flow_state: 心流状态
            group_activity: 群活跃度

        Returns:
            冷却时间（秒）
        """
        return self._calc_cooldown(group_id, flow_state, group_activity)

    def _calc_cooldown(
        self, group_id: str, flow_state: str, group_activity: float
    ) -> float:
        """计算冷却时间"""
        base = self.base_cooldown

        # 根据心流状态调整
        if flow_state == "active":
            base *= 0.6  # 活跃态缩短冷却
        elif flow_state == "flow":
            base *= 0.8
        elif flow_state == "fatigue":
            base *= 2.0  # 疲劳态延长冷却

        # 根据群活跃度调整
        # 活跃群缩短冷却
        activity_factor = 1.0 - 0.3 * group_activity
        base *= activity_factor

        # 应用额外调整
        adjustment = self._cooldown_adjustments.get(group_id, 0)

        return max(10.0, base + adjustment)

    def _calc_delay(self, flow_state: str, energy: float, factor: float) -> float:
        """计算回复延迟"""
        # 基础延迟
        base_delay = self.min_delay

        # 根据状态调整
        if flow_state == "active":
            # 活跃态快速回复
            base_delay = self.min_delay
        elif flow_state == "flow":
            # 沉浸态适度延迟
            base_delay = (self.min_delay + self.max_delay) / 3
        elif flow_state == "observer":
            # 观察态较长延迟
            base_delay = (self.min_delay + self.max_delay) / 2

        # 根据能量调整（能量低时延迟更长）
        energy_factor = 1.0 + (1.0 - energy) * 0.5
        base_delay *= energy_factor

        # 根据因子调整（相关性或活跃度低时延迟更长）
        factor_delay = (1.0 - factor) * (self.max_delay - self.min_delay) * 0.5
        base_delay += factor_delay

        # 添加随机抖动
        jitter = random.uniform(-0.5, 1.0)
        base_delay += jitter

        return max(self.min_delay, min(self.max_delay, base_delay))

    def record_reply(self, group_id: str):
        """记录回复时间"""
        self._last_reply_times[group_id] = time.time()

    def adjust_cooldown(self, group_id: str, adjustment: float):
        """
        调整冷却时间

        Args:
            group_id: 群组ID
            adjustment: 调整值（正数延长，负数缩短）
        """
        current = self._cooldown_adjustments.get(group_id, 0)
        self._cooldown_adjustments[group_id] = current + adjustment

    def get_cooldown_remaining(self, group_id: str) -> float:
        """获取剩余冷却时间"""
        last_reply = self._last_reply_times.get(group_id, 0)
        cooldown = self._calc_cooldown(group_id, "flow", 0.5)

        elapsed = time.time() - last_reply
        return max(0, cooldown - elapsed)

    def is_in_cooldown(self, group_id: str) -> bool:
        """是否在冷却中"""
        return self.get_cooldown_remaining(group_id) > 0

    def reset_cooldown(self, group_id: str):
        """重置冷却"""
        if group_id in self._last_reply_times:
            del self._last_reply_times[group_id]
        if group_id in self._cooldown_adjustments:
            del self._cooldown_adjustments[group_id]

    def force_cooldown(self, group_id: str, seconds: float):
        """强制设置冷却"""
        self._last_reply_times[group_id] = time.time() - self.base_cooldown + seconds

    def get_stats(self, group_id: str) -> dict:
        """获取统计信息"""
        last_reply = self._last_reply_times.get(group_id, 0)
        cooldown_adj = self._cooldown_adjustments.get(group_id, 0)

        return {
            "last_reply_ts": last_reply,
            "seconds_since_last_reply": time.time() - last_reply,
            "cooldown_adjustment": cooldown_adj,
            "cooldown_remaining": self.get_cooldown_remaining(group_id),
        }
