"""
话题追踪器

追踪群聊中的话题：
- 当前话题识别
- 话题连贯性评估
- 话题转换检测
- 话题关键词提取
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from astrbot.api import logger

try:
    import jieba
    import jieba.analyse

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


@dataclass
class Topic:
    """话题"""

    id: str
    keywords: list[str]
    start_time: float
    last_update: float
    message_count: int = 0
    participants: set[str] = field(default_factory=set)
    coherence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "keywords": self.keywords,
            "start_time": self.start_time,
            "last_update": self.last_update,
            "message_count": self.message_count,
            "participants": list(self.participants),
            "coherence": self.coherence,
        }


@dataclass
class TopicTrackerResult:
    """话题追踪结果"""

    current_topic: Topic | None
    topic_coherence: float
    topic_shift_detected: bool
    shift_type: str = ""  # new, drift, continuation
    related_topics: list[Topic] = field(default_factory=list)


class TopicTracker:
    """
    话题追踪器

    追踪群聊中的话题变化
    """

    def __init__(self, config, persistence=None):
        self.config = config
        self.persistence = persistence

        # 当前话题 {group_id: Topic}
        self._current_topics: dict[str, Topic] = {}

        # 话题历史 {group_id: [Topic, ...]}
        self._topic_history: dict[str, list[Topic]] = defaultdict(list)

        # 消息关键词缓存 {group_id: [(keywords, timestamp), ...]}
        self._message_keywords: dict[str, list[tuple[set[str], float]]] = defaultdict(
            list
        )

        # 话题ID计数器
        self._topic_counter = 0

        # 停用词
        self.stop_words = {
            "的",
            "了",
            "在",
            "是",
            "和",
            "与",
            "或",
            "这",
            "那",
            "我",
            "你",
            "他",
            "她",
            "它",
            "们",
            "有",
            "没",
            "不",
            "就",
            "也",
            "都",
            "会",
            "说",
            "要",
            "去",
            "来",
            "能",
            "什么",
            "怎么",
            "为什么",
            "哪",
            "谁",
            "多少",
            "几",
            "一个",
            "这个",
            "那个",
            "就是",
            "不是",
            "没有",
        }

        # 话题转换阈值
        self.shift_threshold = 0.3  # 关键词重叠低于此值认为话题转换
        self.topic_timeout = 600  # 话题超时时间（秒）

        logger.info("话题追踪器初始化完成")

    def track(self, group_id: str, message: str, user_id: str) -> TopicTrackerResult:
        """
        追踪话题

        Args:
            group_id: 群组ID
            message: 消息内容
            user_id: 用户ID

        Returns:
            TopicTrackerResult: 追踪结果
        """
        current_time = time.time()

        # 提取当前消息关键词
        current_keywords = self._extract_keywords(message)

        # 记录消息关键词
        self._message_keywords[group_id].append((current_keywords, current_time))

        # 只保留最近的
        if len(self._message_keywords[group_id]) > 50:
            self._message_keywords[group_id] = self._message_keywords[group_id][-50:]

        # 获取当前话题
        current_topic = self._current_topics.get(group_id)

        # 检查话题超时
        if current_topic:
            if current_time - current_topic.last_update > self.topic_timeout:
                # 话题超时，保存历史
                self._topic_history[group_id].append(current_topic)
                current_topic = None
                del self._current_topics[group_id]

        # 检测话题转换
        topic_shift_detected = False
        shift_type = "continuation"

        if current_topic:
            # 计算与当前话题的重叠度
            overlap = self._calc_keyword_overlap(
                current_keywords, set(current_topic.keywords)
            )

            if overlap < self.shift_threshold:
                topic_shift_detected = True

                # 判断转换类型
                if len(current_keywords) == 0:
                    shift_type = "drift"  # 话题漂移（无法识别）
                else:
                    shift_type = "new"  # 新话题

                # 保存旧话题
                self._topic_history[group_id].append(current_topic)

                # 创建新话题
                current_topic = self._create_topic(
                    current_keywords, user_id, current_time
                )
                self._current_topics[group_id] = current_topic
            else:
                # 更新当前话题
                current_topic.last_update = current_time
                current_topic.message_count += 1
                current_topic.participants.add(user_id)

                # 合并关键词
                all_keywords = set(current_topic.keywords) | current_keywords
                current_topic.keywords = self._get_top_keywords(all_keywords, 10)

                # 更新连贯性
                current_topic.coherence = self._calc_coherence(group_id, current_time)
        else:
            # 创建新话题
            current_topic = self._create_topic(current_keywords, user_id, current_time)
            self._current_topics[group_id] = current_topic
            topic_shift_detected = True
            shift_type = "new"

        # 计算话题连贯性
        topic_coherence = current_topic.coherence if current_topic else 0.0

        # 获取相关话题
        related_topics = self._get_related_topics(group_id, current_keywords)

        return TopicTrackerResult(
            current_topic=current_topic,
            topic_coherence=topic_coherence,
            topic_shift_detected=topic_shift_detected,
            shift_type=shift_type,
            related_topics=related_topics,
        )

    def get_current_topic(self, group_id: str) -> Topic | None:
        """获取当前话题"""
        return self._current_topics.get(group_id)

    def calc_coherence(self, group_id: str) -> float:
        """计算话题连贯性"""
        return self._calc_coherence(group_id, time.time())

    def detect_shift(self, group_id: str, message: str) -> bool:
        """检测话题转换"""
        current_keywords = self._extract_keywords(message)
        current_topic = self._current_topics.get(group_id)

        if not current_topic:
            return True

        overlap = self._calc_keyword_overlap(
            current_keywords, set(current_topic.keywords)
        )

        return overlap < self.shift_threshold

    def _extract_keywords(self, text: str) -> set[str]:
        """提取关键词"""
        if not text:
            return set()

        if HAS_JIEBA:
            keywords = jieba.analyse.extract_tags(text, topK=10)
            return {kw for kw in keywords if kw not in self.stop_words and len(kw) > 1}
        else:
            # 简单分词
            words = re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z]+", text.lower())
            return {w for w in words if w not in self.stop_words and len(w) > 1}

    def _calc_keyword_overlap(self, keywords1: set[str], keywords2: set[str]) -> float:
        """计算关键词重叠度"""
        if not keywords1 or not keywords2:
            return 0.0

        intersection = keywords1 & keywords2
        union = keywords1 | keywords2

        return len(intersection) / len(union) if union else 0.0

    def _create_topic(
        self, keywords: set[str], user_id: str, timestamp: float
    ) -> Topic:
        """创建新话题"""
        self._topic_counter += 1

        return Topic(
            id=f"topic_{self._topic_counter}",
            keywords=list(keywords)[:10],
            start_time=timestamp,
            last_update=timestamp,
            message_count=1,
            participants={user_id},
            coherence=0.5,
        )

    def _get_top_keywords(self, keywords: set[str], top_n: int) -> list[str]:
        """获取前N个关键词"""
        # 简化实现：直接返回前N个
        # 实际应该根据词频排序
        return list(keywords)[:top_n]

    def _calc_coherence(self, group_id: str, current_time: float) -> float:
        """计算话题连贯性"""
        keywords_history = self._message_keywords.get(group_id, [])

        if len(keywords_history) < 2:
            return 0.5

        # 只看最近的
        recent = [(kw, ts) for kw, ts in keywords_history if current_time - ts < 300]

        if len(recent) < 2:
            return 0.5

        # 计算相邻消息的关键词重叠度
        overlaps = []
        for i in range(len(recent) - 1):
            kw1, _ = recent[i]
            kw2, _ = recent[i + 1]
            overlap = self._calc_keyword_overlap(kw1, kw2)
            overlaps.append(overlap)

        if not overlaps:
            return 0.5

        return sum(overlaps) / len(overlaps)

    def _get_related_topics(self, group_id: str, keywords: set[str]) -> list[Topic]:
        """获取相关话题"""
        history = self._topic_history.get(group_id, [])

        if not history:
            return []

        # 计算与历史话题的相关性
        scored = []
        for topic in history[-10:]:
            overlap = self._calc_keyword_overlap(keywords, set(topic.keywords))
            if overlap > 0.2:
                scored.append((topic, overlap))

        # 按相关性排序
        scored.sort(key=lambda x: x[1], reverse=True)

        return [t for t, _ in scored[:3]]

    def get_topic_history(self, group_id: str, limit: int = 10) -> list[dict]:
        """获取话题历史"""
        history = self._topic_history.get(group_id, [])
        return [t.to_dict() for t in history[-limit:]]

    def get_stats(self, group_id: str) -> dict:
        """获取统计信息"""
        current = self._current_topics.get(group_id)
        history = self._topic_history.get(group_id, [])

        return {
            "current_topic": current.to_dict() if current else None,
            "history_count": len(history),
            "total_topics": len(history) + (1 if current else 0),
        }

    def reset(self, group_id: str):
        """重置群组话题"""
        if group_id in self._current_topics:
            del self._current_topics[group_id]
        if group_id in self._topic_history:
            del self._topic_history[group_id]
        if group_id in self._message_keywords:
            del self._message_keywords[group_id]
