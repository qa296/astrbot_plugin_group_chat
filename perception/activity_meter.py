"""
活跃度计量器

计量群组的活跃度，包括：
- 消息频率
- 用户多样性
- 互动强度
- 时间分布
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

from astrbot.api import logger


@dataclass
class ActivityMetrics:
    """活跃度指标"""

    messages_per_minute: float = 0.0
    active_users_count: int = 0
    user_diversity: float = 0.0
    interaction_intensity: float = 0.0
    peak_activity: float = 0.0
    activity_trend: float = 0.0  # 正值表示上升，负值表示下降
    overall_activity: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ActivityMeter:
    """
    活跃度计量器

    计量群组的活跃度
    """

    def __init__(self, config, persistence=None):
        self.config = config
        self.persistence = persistence

        # 消息时间戳缓存 {group_id: [timestamp1, timestamp2, ...]}
        self._message_times: dict[str, list[float]] = defaultdict(list)

        # 用户消息计数 {group_id: {user_id: count}}
        self._user_messages: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # 历史活跃度 {group_id: [(timestamp, activity), ...]}
        self._activity_history: dict[str, list[tuple]] = defaultdict(list)

        # 清理间隔
        self._cleanup_interval = 3600  # 1小时
        self._last_cleanup = time.time()



    def record_message(self, group_id: str, user_id: str):
        """记录消息"""
        current_time = time.time()

        # 记录消息时间
        self._message_times[group_id].append(current_time)

        # 记录用户消息
        self._user_messages[group_id][user_id] += 1

        # 清理旧数据
        self._cleanup_if_needed(current_time)

    def measure(self, group_id: str) -> ActivityMetrics:
        """
        计量活跃度

        Args:
            group_id: 群组ID

        Returns:
            ActivityMetrics: 活跃度指标
        """
        current_time = time.time()

        metrics = ActivityMetrics()

        # 1. 每分钟消息数
        metrics.messages_per_minute = self._calc_messages_per_minute(
            group_id, current_time
        )

        # 2. 活跃用户数
        metrics.active_users_count = self._calc_active_users(group_id, current_time)

        # 3. 用户多样性（基于基尼系数）
        metrics.user_diversity = self._calc_user_diversity(group_id)

        # 4. 互动强度
        metrics.interaction_intensity = self._calc_interaction_intensity(
            group_id, current_time
        )

        # 5. 峰值活跃度
        metrics.peak_activity = self._calc_peak_activity(group_id, current_time)

        # 6. 活跃度趋势
        metrics.activity_trend = self._calc_activity_trend(group_id, current_time)

        # 7. 综合活跃度
        metrics.overall_activity = self._calc_overall_activity(metrics)

        # 记录历史
        self._activity_history[group_id].append(
            (current_time, metrics.overall_activity)
        )

        # 只保留最近100条
        if len(self._activity_history[group_id]) > 100:
            self._activity_history[group_id] = self._activity_history[group_id][-100:]

        return metrics

    def get_messages_per_minute(self, group_id: str) -> float:
        """获取每分钟消息数"""
        return self._calc_messages_per_minute(group_id, time.time())

    def get_user_diversity(self, group_id: str) -> float:
        """获取用户多样性"""
        return self._calc_user_diversity(group_id)

    def get_activity_trend(self, group_id: str) -> float:
        """获取活跃度趋势"""
        return self._calc_activity_trend(group_id, time.time())

    def _calc_messages_per_minute(self, group_id: str, current_time: float) -> float:
        """计算每分钟消息数"""
        times = self._message_times.get(group_id, [])

        # 最近5分钟内的消息
        recent_times = [t for t in times if current_time - t < 300]

        if not recent_times:
            return 0.0

        return len(recent_times) / 5.0

    def _calc_active_users(self, group_id: str, current_time: float) -> int:
        """计算活跃用户数（最近5分钟内发言的用户）"""
        # 简化实现：使用用户消息计数
        user_counts = self._user_messages.get(group_id, {})

        # 假设有消息的用户都是活跃的
        # 实际应该根据时间戳筛选
        return len(user_counts)

    def _calc_user_diversity(self, group_id: str) -> float:
        """
        计算用户多样性

        使用归一化熵来衡量
        值越高表示消息分布越均匀
        """
        user_counts = self._user_messages.get(group_id, {})

        if not user_counts:
            return 0.0

        total = sum(user_counts.values())
        if total == 0:
            return 0.0

        import math

        # 计算熵
        entropy = 0.0
        for count in user_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # 归一化：最大熵是 log2(n)
        n = len(user_counts)
        max_entropy = math.log2(n) if n > 1 else 1

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calc_interaction_intensity(self, group_id: str, current_time: float) -> float:
        """
        计算互动强度

        基于@、回复等互动行为
        简化实现：基于消息密度和用户多样性
        """
        mpm = self._calc_messages_per_minute(group_id, current_time)
        diversity = self._calc_user_diversity(group_id)

        # 高消息密度 + 高多样性 = 高互动强度
        intensity = mpm * (0.5 + 0.5 * diversity) / 5.0

        return min(1.0, intensity)

    def _calc_peak_activity(self, group_id: str, current_time: float) -> float:
        """计算峰值活跃度"""
        times = self._message_times.get(group_id, [])

        if not times:
            return 0.0

        # 计算滑动窗口内的最大消息数
        window = 60  # 1分钟窗口
        max_count = 0

        for i, t in enumerate(times):
            if current_time - t > 600:  # 只看最近10分钟
                continue
            count = sum(1 for t2 in times[i:] if t2 - t < window)
            max_count = max(max_count, count)

        return min(1.0, max_count / 10.0)

    def _calc_activity_trend(self, group_id: str, current_time: float) -> float:
        """
        计算活跃度趋势

        比较最近2分钟和之前2分钟的消息数
        返回 -1 到 1 之间的值
        """
        times = self._message_times.get(group_id, [])

        recent = [t for t in times if current_time - t < 120]
        previous = [t for t in times if 120 <= current_time - t < 240]

        recent_count = len(recent)
        previous_count = len(previous)

        if previous_count == 0:
            return 1.0 if recent_count > 0 else 0.0

        change = (recent_count - previous_count) / max(previous_count, 1)
        return max(-1.0, min(1.0, change))

    def _calc_overall_activity(self, metrics: ActivityMetrics) -> float:
        """计算综合活跃度"""
        weights = {
            "messages_per_minute": 0.3,
            "user_diversity": 0.2,
            "interaction_intensity": 0.2,
            "peak_activity": 0.15,
            "activity_trend": 0.15,
        }

        score = (
            weights["messages_per_minute"] * min(1.0, metrics.messages_per_minute / 5.0)
            + weights["user_diversity"] * metrics.user_diversity
            + weights["interaction_intensity"] * metrics.interaction_intensity
            + weights["peak_activity"] * metrics.peak_activity
            + weights["activity_trend"] * (0.5 + 0.5 * metrics.activity_trend)
        )

        return min(1.0, max(0.0, score))

    def _cleanup_if_needed(self, current_time: float):
        """清理过期数据"""
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time

        # 清理消息时间戳（保留最近1小时）
        cutoff = current_time - 3600
        for group_id in list(self._message_times.keys()):
            self._message_times[group_id] = [
                t for t in self._message_times[group_id] if t > cutoff
            ]

        # 清理用户消息计数（每小时重置一次）
        # 可选：保留长期统计

    def get_stats(self, group_id: str) -> dict:
        """获取统计信息"""
        metrics = self.measure(group_id)
        return {
            "messages_per_minute": metrics.messages_per_minute,
            "active_users_count": metrics.active_users_count,
            "user_diversity": metrics.user_diversity,
            "interaction_intensity": metrics.interaction_intensity,
            "peak_activity": metrics.peak_activity,
            "activity_trend": metrics.activity_trend,
            "overall_activity": metrics.overall_activity,
        }

    def reset(self, group_id: str):
        """重置群组数据"""
        if group_id in self._message_times:
            del self._message_times[group_id]
        if group_id in self._user_messages:
            del self._user_messages[group_id]
        if group_id in self._activity_history:
            del self._activity_history[group_id]
