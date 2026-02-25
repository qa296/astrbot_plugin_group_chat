"""
反馈收集器

收集用户对机器人回复的反馈：
- 用户回复检测
- 点赞/表情检测
- 忽略检测
- 负面反馈检测
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from astrbot.api import logger


class FeedbackType(Enum):
    """反馈类型"""

    REPLY = "reply"  # 用户回复了
    AT_FOLLOW = "at_follow"  # 用户@跟进
    LIKE = "like"  # 用户点赞/表情回应
    IGNORE = "ignore"  # 用户忽略
    NEGATIVE = "negative"  # 负面反馈
    NEUTRAL = "neutral"  # 中性


@dataclass
class FeedbackEvent:
    """反馈事件"""

    group_id: str
    user_id: str
    bot_message_id: str
    bot_message_content: str
    feedback_type: FeedbackType
    reward: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class PendingFeedback:
    """待收集的反馈"""

    group_id: str
    bot_message_id: str
    bot_message_content: str
    bot_message_ts: float
    last_user_id: str
    expires_at: float


class FeedbackCollector:
    """
    反馈收集器

    收集用户对机器人回复的反馈
    """

    def __init__(self, config, persistence=None):
        self.config = config
        self.persistence = persistence

        # 奖励配置
        reward_config = config.get("reward", {})
        self.reward_user_reply = reward_config.get("user_reply", 0.5)
        self.reward_user_at_follow = reward_config.get("user_at_follow", 1.0)
        self.reward_user_like = reward_config.get("user_like", 0.3)
        self.penalty_ignore = reward_config.get("ignore_penalty", -0.2)
        self.penalty_negative = reward_config.get("negative_penalty", -1.0)
        self.feedback_window = reward_config.get(
            "feedback_detection_window_seconds", 60
        )

        # 待收集的反馈 {group_id: PendingFeedback}
        self._pending: dict[str, PendingFeedback] = {}

        # 已收集的反馈历史
        self._history: list[FeedbackEvent] = []

        # 负面关键词
        self._negative_keywords = {
            "闭嘴",
            "滚",
            "烦",
            "讨厌",
            "闭嘴",
            "别说了",
            "吵",
            "垃圾",
            "废物",
            "傻",
            "笨",
            "恶心",
            "能不能别",
            "不要再",
            "能不能不说话",
        }

        # 点赞表情关键词（某些平台）
        self._like_keywords = {
            "👍",
            "❤️",
            "💕",
            "😂",
            "🤣",
            "👏",
            "666",
            "厉害",
            "棒",
            "赞",
            "牛逼",
        }



    def register_pending_feedback(self, group_id: str, event, bot_message: str):
        """
        注册待收集反馈

        Args:
            group_id: 群组ID
            event: 消息事件
            bot_message: 机器人回复内容
        """
        pending = PendingFeedback(
            group_id=group_id,
            bot_message_id=str(int(time.time() * 1000)),
            bot_message_content=bot_message,
            bot_message_ts=time.time(),
            last_user_id=event.get_sender_id(),
            expires_at=time.time() + self.feedback_window,
        )

        self._pending[group_id] = pending

    def check_message_for_feedback(
        self, group_id: str, user_id: str, message: str, is_at_bot: bool
    ) -> FeedbackEvent | None:
        """
        检查消息是否为反馈

        Args:
            group_id: 群组ID
            user_id: 用户ID
            message: 消息内容
            is_at_bot: 是否@机器人

        Returns:
            FeedbackEvent 或 None
        """
        pending = self._pending.get(group_id)

        if not pending:
            return None

        # 忽略机器人自己的消息
        if user_id == pending.last_user_id:
            return None

        current_time = time.time()

        # 检查是否过期
        if current_time > pending.expires_at:
            # 过期视为忽略
            feedback = FeedbackEvent(
                group_id=group_id,
                user_id=user_id,
                bot_message_id=pending.bot_message_id,
                bot_message_content=pending.bot_message_content,
                feedback_type=FeedbackType.IGNORE,
                reward=self.penalty_ignore,
            )
            del self._pending[group_id]
            return feedback

        # 检测反馈类型
        feedback_type = self._detect_feedback_type(
            message, is_at_bot, pending.bot_message_content
        )

        if feedback_type == FeedbackType.NEUTRAL:
            # 尚未有明确反馈
            return None

        # 计算奖励
        reward = self._calc_reward(feedback_type)

        # 创建反馈事件
        feedback = FeedbackEvent(
            group_id=group_id,
            user_id=user_id,
            bot_message_id=pending.bot_message_id,
            bot_message_content=pending.bot_message_content,
            feedback_type=feedback_type,
            reward=reward,
        )

        # 记录历史
        self._history.append(feedback)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        # 移除待收集
        del self._pending[group_id]

        return feedback

    def _detect_feedback_type(
        self, message: str, is_at_bot: bool, bot_message: str
    ) -> FeedbackType:
        """检测反馈类型"""
        message_lower = message.lower()

        # 检测负面反馈
        for kw in self._negative_keywords:
            if kw in message_lower:
                return FeedbackType.NEGATIVE

        # 检测点赞
        for kw in self._like_keywords:
            if kw in message_lower:
                return FeedbackType.LIKE

        # 检测@跟进
        if is_at_bot:
            return FeedbackType.AT_FOLLOW

        # 检测普通回复
        # 简单判断：消息长度 > 3 且包含回复意图
        if len(message) > 3:
            # 检查是否是对机器人消息的回复
            if self._is_reply_to_bot(message, bot_message):
                return FeedbackType.REPLY

        return FeedbackType.NEUTRAL

    def _is_reply_to_bot(self, message: str, bot_message: str) -> bool:
        """判断是否是对机器人消息的回复"""
        # 简单实现：检查是否包含回复特征
        reply_indicators = [
            "呢",
            "啊",
            "呀",
            "吧",
            "嘛",
            "好的",
            "嗯",
            "哦",
            "是",
            "对",
            "不是",
            "不对",
            "没有",
        ]

        for indicator in reply_indicators:
            if indicator in message:
                return True

        return False

    def _calc_reward(self, feedback_type: FeedbackType) -> float:
        """计算奖励值"""
        if feedback_type == FeedbackType.REPLY:
            return self.reward_user_reply
        elif feedback_type == FeedbackType.AT_FOLLOW:
            return self.reward_user_at_follow
        elif feedback_type == FeedbackType.LIKE:
            return self.reward_user_like
        elif feedback_type == FeedbackType.IGNORE:
            return self.penalty_ignore
        elif feedback_type == FeedbackType.NEGATIVE:
            return self.penalty_negative
        else:
            return 0.0

    async def collect_later(self, group_id: str, context, strategy_store=None):
        """
        延迟收集反馈（用于主动消息后）

        Args:
            group_id: 群组ID
            context: 上下文（用于获取最近消息）
            strategy_store: 策略库（用于更新成功率）
        """
        await asyncio.sleep(self.feedback_window)

        pending = self._pending.get(group_id)
        if not pending:
            return

        # 检查是否收到了用户消息
        # 这里需要从上下文获取最近消息
        # 简化实现：超时视为忽略

        feedback = FeedbackEvent(
            group_id=group_id,
            user_id="system",
            bot_message_id=pending.bot_message_id,
            bot_message_content=pending.bot_message_content,
            feedback_type=FeedbackType.IGNORE,
            reward=self.penalty_ignore,
        )

        self._history.append(feedback)
        del self._pending[group_id]

        logger.debug(f"群组 {group_id} 反馈收集超时，视为忽略")

    def get_pending_count(self) -> int:
        """获取待收集数量"""
        return len(self._pending)

    def get_history(self, limit: int = 100) -> list[FeedbackEvent]:
        """获取历史反馈"""
        return self._history[-limit:]

    def get_stats(self) -> dict:
        """获取统计信息"""
        if not self._history:
            return {
                "total_feedbacks": 0,
                "avg_reward": 0.0,
                "by_type": {},
            }

        by_type = {}
        for feedback in self._history:
            ft = feedback.feedback_type.value
            if ft not in by_type:
                by_type[ft] = {"count": 0, "total_reward": 0.0}
            by_type[ft]["count"] += 1
            by_type[ft]["total_reward"] += feedback.reward

        return {
            "total_feedbacks": len(self._history),
            "avg_reward": sum(f.reward for f in self._history) / len(self._history),
            "by_type": by_type,
        }

    def get_recent_rewards(self, limit: int = 20) -> list[float]:
        """获取最近的奖励值"""
        return [f.reward for f in self._history[-limit:]]

    def clear_expired(self):
        """清除过期的待收集反馈"""
        current_time = time.time()
        expired = []

        for group_id, pending in self._pending.items():
            if current_time > pending.expires_at:
                expired.append(group_id)

        for group_id in expired:
            del self._pending[group_id]

        return len(expired)

    def add_negative_keyword(self, keyword: str):
        """添加负面关键词"""
        self._negative_keywords.add(keyword)

    def add_like_keyword(self, keyword: str):
        """添加点赞关键词"""
        self._like_keywords.add(keyword)
