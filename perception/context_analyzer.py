"""
上下文分析器

分析消息上下文，提取关键特征用于决策：
- 群活跃度
- 话题连贯性
- 与机器人相关性
- 情感倾向
- 对话历史
"""

import re
import time
from dataclasses import dataclass, field

from astrbot.api import logger

try:
    import jieba
    import jieba.analyse

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    logger.info("jieba 未安装，使用内置简单分词")


@dataclass
class AnalyzedContext:
    """分析后的上下文"""

    group_id: str
    user_id: str
    message_content: str

    # 时间维度
    message_count_1m: int = 0  # 1分钟内消息数
    message_count_5m: int = 0  # 5分钟内消息数
    message_count_30m: int = 0  # 30分钟内消息数

    # 用户维度
    active_users: list[str] = field(default_factory=list)
    active_user_count: int = 0

    # 话题维度
    topic_coherence: float = 0.0  # 话题连贯性 0-1
    current_topic_keywords: list[str] = field(default_factory=list)
    topic_shift_detected: bool = False

    # 相关性维度
    relevance_to_bot: float = 0.0  # 与机器人相关性 0-1
    is_direct_question: bool = False
    is_at_bot: bool = False

    # 情感维度
    sentiment: float = 0.0  # 情感倾向 -1~1
    emotion_type: str = "neutral"  # 情感类型

    # 对话历史
    conversation_history: list[dict] = field(default_factory=list)
    last_bot_reply_ts: float = 0.0
    time_since_last_bot_reply: float = 0.0

    # 综合评估
    group_activity: float = 0.5  # 综合群活跃度 0-1
    interaction_quality: float = 0.5  # 交互质量 0-1

    # 元数据
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "user_id": self.user_id,
            "message_content": self.message_content,
            "message_count_1m": self.message_count_1m,
            "message_count_5m": self.message_count_5m,
            "message_count_30m": self.message_count_30m,
            "active_user_count": self.active_user_count,
            "topic_coherence": self.topic_coherence,
            "relevance_to_bot": self.relevance_to_bot,
            "is_at_bot": self.is_at_bot,
            "sentiment": self.sentiment,
            "group_activity": self.group_activity,
            "timestamp": self.timestamp,
        }


class ContextAnalyzer:
    """
    上下文分析器

    分析消息上下文，提取关键特征
    """

    def __init__(self, context, config, persistence=None):
        self.context = context
        self.config = config
        self.persistence = persistence

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
        }

        # 机器人关键词（从人格系统获取）
        self._bot_keywords: list[str] = []
        self._keywords_updated: float = 0

        # 缓存
        self._history_cache: dict[str, list[dict]] = {}

        logger.info("上下文分析器初始化完成")

    async def analyze(self, event) -> AnalyzedContext:
        """
        分析消息上下文

        Args:
            event: 消息事件

        Returns:
            AnalyzedContext: 分析结果
        """
        group_id = event.get_group_id()
        user_id = event.get_sender_id()
        message_str = event.message_str
        current_time = time.time()

        # 更新对话历史
        self._update_history(group_id, user_id, message_str, "user", current_time)

        # 获取历史记录
        history = self._get_history(group_id)

        # 创建分析结果
        result = AnalyzedContext(
            group_id=group_id,
            user_id=user_id,
            message_content=message_str,
            timestamp=current_time,
        )

        # 1. 时间维度分析
        result.message_count_1m = self._count_messages(history, current_time, 60)
        result.message_count_5m = self._count_messages(history, current_time, 300)
        result.message_count_30m = self._count_messages(history, current_time, 1800)

        # 2. 用户维度分析
        result.active_users = self._get_active_users(history, current_time, 300)
        result.active_user_count = len(result.active_users)

        # 3. 话题连贯性分析
        result.topic_coherence = self._calc_topic_coherence(history, current_time)
        result.current_topic_keywords = self._extract_keywords(history[-10:])
        result.topic_shift_detected = self._detect_topic_shift(history, message_str)

        # 4. 相关性分析
        result.relevance_to_bot = self._calc_relevance(event)
        result.is_at_bot = result.relevance_to_bot >= 0.9
        result.is_direct_question = self._is_direct_question(message_str)

        # 5. 情感分析
        result.sentiment, result.emotion_type = self._analyze_sentiment(message_str)

        # 6. 对话历史
        result.conversation_history = history[-20:]
        result.last_bot_reply_ts = self._get_last_bot_reply_ts(history)
        result.time_since_last_bot_reply = current_time - result.last_bot_reply_ts

        # 7. 综合评估
        result.group_activity = self._calc_group_activity(result)
        result.interaction_quality = self._calc_interaction_quality(result)

        return result

    async def analyze_proactive(self, group_id: str) -> AnalyzedContext:
        """
        分析主动消息场景

        Args:
            group_id: 群组ID

        Returns:
            AnalyzedContext: 分析结果
        """
        current_time = time.time()
        history = self._get_history(group_id)

        result = AnalyzedContext(
            group_id=group_id,
            user_id="system",
            message_content="",
            timestamp=current_time,
        )

        # 时间维度
        result.message_count_1m = self._count_messages(history, current_time, 60)
        result.message_count_5m = self._count_messages(history, current_time, 300)
        result.message_count_30m = self._count_messages(history, current_time, 1800)

        # 用户维度
        result.active_users = self._get_active_users(history, current_time, 300)
        result.active_user_count = len(result.active_users)

        # 话题维度
        result.topic_coherence = self._calc_topic_coherence(history, current_time)
        result.current_topic_keywords = self._extract_keywords(history[-10:])

        # 对话历史
        result.conversation_history = history[-20:]
        result.last_bot_reply_ts = self._get_last_bot_reply_ts(history)
        result.time_since_last_bot_reply = current_time - result.last_bot_reply_ts

        # 综合评估
        result.group_activity = self._calc_group_activity(result)
        result.interaction_quality = self._calc_interaction_quality(result)

        return result

    def _update_history(
        self, group_id: str, user_id: str, content: str, role: str, timestamp: float
    ):
        """更新对话历史"""
        if group_id not in self._history_cache:
            self._history_cache[group_id] = []

        self._history_cache[group_id].append(
            {
                "user_id": user_id,
                "content": content,
                "role": role,
                "timestamp": timestamp,
            }
        )

        # 只保留最近200条
        if len(self._history_cache[group_id]) > 200:
            self._history_cache[group_id] = self._history_cache[group_id][-200:]

        # 同步到持久化
        if self.persistence:
            from storage.persistence import ConversationRecord

            self.persistence.add_conversation_record(
                group_id,
                ConversationRecord(
                    user_id=user_id,
                    content=content,
                    role=role,
                    timestamp=timestamp,
                ),
            )

    def _get_history(self, group_id: str) -> list[dict]:
        """获取对话历史"""
        if group_id in self._history_cache:
            return self._history_cache[group_id]

        # 从持久化加载
        if self.persistence:
            records = self.persistence.get_conversation_history(group_id, 100)
            history = [
                {
                    "user_id": r.user_id,
                    "content": r.content,
                    "role": r.role,
                    "timestamp": r.timestamp,
                }
                for r in records
            ]
            self._history_cache[group_id] = history
            return history

        return []

    def _count_messages(
        self, history: list[dict], current_time: float, window: int
    ) -> int:
        """计算时间窗口内的消息数"""
        return sum(1 for m in history if current_time - m.get("timestamp", 0) < window)

    def _get_active_users(
        self, history: list[dict], current_time: float, window: int
    ) -> list[str]:
        """获取活跃用户"""
        users = set()
        for m in history:
            if current_time - m.get("timestamp", 0) < window:
                users.add(m.get("user_id", ""))
        return list(users)

    def _calc_topic_coherence(self, history: list[dict], current_time: float) -> float:
        """计算话题连贯性"""
        recent = [
            m["content"]
            for m in history[-8:]
            if current_time - m.get("timestamp", 0) < 600
        ]
        if len(recent) < 2:
            return 0.0

        # 提取关键词集合
        keywords_list = []
        for content in recent:
            keywords = set(self._tokenize(content))
            keywords = keywords - self.stop_words
            if keywords:
                keywords_list.append(keywords)

        if len(keywords_list) < 2:
            return 0.0

        # 计算相邻消息的关键词重叠度
        overlaps = []
        for i in range(len(keywords_list) - 1):
            if keywords_list[i] and keywords_list[i + 1]:
                overlap = len(keywords_list[i] & keywords_list[i + 1])
                union = len(keywords_list[i] | keywords_list[i + 1])
                if union > 0:
                    overlaps.append(overlap / union)

        if not overlaps:
            return 0.0

        return min(1.0, sum(overlaps) / len(overlaps))

    def _extract_keywords(self, messages: list[dict]) -> list[str]:
        """提取关键词"""
        if not messages:
            return []

        text = " ".join(m.get("content", "") for m in messages)

        if HAS_JIEBA:
            keywords = jieba.analyse.extract_tags(text, topK=10)
            return keywords
        else:
            # 简单分词
            words = self._tokenize(text)
            # 词频统计
            from collections import Counter

            word_freq = Counter(
                w for w in words if w not in self.stop_words and len(w) > 1
            )
            return [w for w, _ in word_freq.most_common(10)]

    def _detect_topic_shift(self, history: list[dict], new_message: str) -> bool:
        """检测话题转换"""
        if len(history) < 3:
            return False

        recent_keywords = set(self._extract_keywords(history[-3:]))
        new_keywords = set(self._tokenize(new_message)) - self.stop_words

        if not recent_keywords or not new_keywords:
            return False

        overlap = len(recent_keywords & new_keywords)
        if overlap < len(new_keywords) * 0.2:
            return True

        return False

    def _calc_relevance(self, event) -> float:
        """计算与机器人的相关性"""
        message_str = event.message_str.lower()

        # 检查是否被@
        if hasattr(event, "is_at_or_wake_command") and event.is_at_or_wake_command:
            return 1.0

        # 检查消息链中的@组件
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            for comp in event.message_obj.message:
                if hasattr(comp, "type") and comp.type == "at":
                    if hasattr(comp, "qq"):
                        try:
                            if str(comp.qq) == str(event.get_self_id()):
                                return 0.95
                        except Exception:
                            pass
                    return 0.85  # 有@但不明确是@机器人

        # 获取机器人关键词
        bot_keywords = self._get_bot_keywords()

        # 检查关键词
        for kw in bot_keywords:
            if kw.lower() in message_str:
                return 0.7

        # 检查是否是直接提问
        if self._is_direct_question(event.message_str):
            return 0.4

        return 0.0

    def _get_bot_keywords(self) -> list[str]:
        """获取机器人相关关键词"""
        current_time = time.time()

        # 每5分钟更新一次
        if current_time - self._keywords_updated > 300:
            self._keywords_updated = current_time
            self._bot_keywords = self._fetch_bot_keywords()

        return self._bot_keywords

    def _fetch_bot_keywords(self) -> list[str]:
        """从人格系统获取关键词"""
        keywords = []

        try:
            if hasattr(self.context, "provider_manager"):
                pm = self.context.provider_manager

                # 默认人格名称
                default_persona = getattr(pm, "selected_default_persona", {})
                if default_persona and "name" in default_persona:
                    keywords.append(default_persona["name"])

                # 所有人格名称
                personas = getattr(pm, "personas", {})
                if isinstance(personas, dict):
                    keywords.extend(personas.keys())
                elif isinstance(personas, list):
                    for p in personas:
                        name = (
                            p.get("name")
                            if isinstance(p, dict)
                            else getattr(p, "name", None)
                        )
                        if name:
                            keywords.append(name)
        except Exception as e:
            logger.debug(f"获取人格关键词失败: {e}")

        # 默认关键词
        if not keywords:
            keywords = ["机器人", "bot", "助手"]

        return list(set(keywords))

    def _is_direct_question(self, text: str) -> bool:
        """判断是否是直接提问"""
        question_patterns = [
            r"吗[？?]",
            r"呢[？?]",
            r"怎么.*[？?]",
            r"什么.*[？?]",
            r"为什么.*[？?]",
            r"谁.*[？?]",
            r"哪.*[？?]",
            r"多少.*[？?]",
            r"是不是.*[？?]",
            r"能不能.*[？?]",
            r"可以.*[？?]",
            r"有没有.*[？?]",
        ]

        for pattern in question_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _analyze_sentiment(self, text: str) -> tuple:
        """分析情感"""
        # 简单的情感词典
        positive_words = {
            "好",
            "棒",
            "厉害",
            "赞",
            "喜欢",
            "爱",
            "开心",
            "高兴",
            "谢谢",
            "感谢",
            "牛逼",
            "牛逼",
            "太棒了",
            "哈哈",
            "嘻嘻",
            "可爱",
            "有趣",
            "不错",
            "可以",
            "行",
            "好的",
            "好的呢",
        }

        negative_words = {
            "烦",
            "讨厌",
            "恶心",
            "滚",
            "傻",
            "笨",
            "垃圾",
            "废物",
            "没用",
            "无聊",
            "差",
            "烂",
            "什么破",
            "什么鬼",
            "骂",
            "不爽",
            "生气",
            "愤怒",
            "无语",
            "郁闷",
        }

        words = set(self._tokenize(text))

        positive_count = len(words & positive_words)
        negative_count = len(words & negative_words)

        if positive_count > negative_count:
            return (0.5, "positive")
        elif negative_count > positive_count:
            return (-0.5, "negative")
        else:
            return (0.0, "neutral")

    def _get_last_bot_reply_ts(self, history: list[dict]) -> float:
        """获取最后一条机器人回复时间"""
        for m in reversed(history):
            if m.get("role") == "assistant":
                return m.get("timestamp", 0)
        return 0.0

    def _calc_group_activity(self, result: AnalyzedContext) -> float:
        """计算综合群活跃度"""
        # 时间窗口权重
        score = (
            result.message_count_1m * 0.4
            + result.message_count_5m * 0.3 / 5
            + result.message_count_30m * 0.2 / 30
            + result.active_user_count * 0.1
        )

        # 归一化到 0-1
        return min(1.0, score / 10)

    def _calc_interaction_quality(self, result: AnalyzedContext) -> float:
        """计算交互质量"""
        quality = 0.5

        # 话题连贯性加分
        quality += result.topic_coherence * 0.2

        # 相关性加分
        quality += result.relevance_to_bot * 0.2

        # 正面情感加分
        if result.sentiment > 0:
            quality += 0.1

        return min(1.0, quality)

    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        if HAS_JIEBA:
            return list(jieba.cut(text))
        else:
            # 简单正则分词
            return re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z]+|\d+", text.lower())

    def add_bot_reply_to_history(self, group_id: str, content: str):
        """添加机器人回复到历史"""
        self._update_history(group_id, "bot", content, "assistant", time.time())

    def clear_history(self, group_id: str):
        """清除对话历史"""
        if group_id in self._history_cache:
            del self._history_cache[group_id]
