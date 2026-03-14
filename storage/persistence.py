"""
持久化管理器

负责插件数据的持久化存储，数据存储在 data/plugin_data/astrbot_plugin_group_chat/ 目录下
遵循 AstrBot 插件开发规范：持久化数据存储于 data/plugin_data 目录下
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
    HAS_ASTRBOT_PATH = True
except ImportError:
    HAS_ASTRBOT_PATH = False


@dataclass
class GroupStateData:
    """群组状态数据"""

    group_id: str
    flow_state: str = "observer"
    energy: float = 0.8
    last_reply_ts: float = 0.0
    reply_streak: int = 0
    last_message_ts: float = 0.0
    fatigue_start_ts: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class StrategyEntryData:
    """策略条目数据"""

    id: str
    context_hash: str
    flow_state: str
    action_type: str
    action_params: dict[str, Any]
    success_count: int = 0
    total_count: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count


@dataclass
class LearningData:
    """学习数据"""

    q_table: dict[str, dict[str, float]] = field(default_factory=dict)
    episode_count: int = 0
    total_reward: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class ConversationRecord:
    """对话记录"""

    user_id: str
    content: str
    role: str
    timestamp: float
    message_id: str = ""


@dataclass
class GroupMessage:
    """群聊消息（用于离线蒸馏）"""

    id: str
    seq: int  # 消息序号
    user_id: str
    user_name: str
    content: str
    timestamp: float
    processed: bool = False  # 是否已被蒸馏处理


@dataclass
class GroupHistory:
    """群聊历史"""

    messages: list[dict] = field(default_factory=list)
    last_distill_time: float = 0.0
    total_messages: int = 0
    last_seq: int = 0


@dataclass
class GlobalVocabulary:
    """全局词表"""

    vocabulary: list[str] = field(default_factory=list)
    idf_values: dict[str, float] = field(default_factory=dict)
    doc_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class SimilarityRule:
    """相似度规则"""

    id: str
    original_text: str  # 原始文本（用于生成向量）
    threshold: float  # 相似度阈值
    source_group: str  # 来源群
    source_seq: int  # 来源消息序号
    created_at: float = field(default_factory=time.time)
    use_count: int = 0
    success_count: int = 0


@dataclass
class RegexRule:
    """正则规则"""

    id: str
    pattern: str  # 正则表达式
    trigger_count: int  # 触发需要的匹配次数
    current_count: int = 0  # 当前计数
    created_at: float = field(default_factory=time.time)
    use_count: int = 0
    success_count: int = 0


class PersistenceManager:
    """
    持久化管理器

    数据存储在 data/plugins/astrbot_plugin_group_chat/ 目录下
    使用内存缓存 + 异步持久化策略
    """

    def __init__(self, plugin_name: str = "astrbot_plugin_group_chat"):
        self.plugin_name = plugin_name

        if HAS_ASTRBOT_PATH:
            self.data_dir = Path(get_astrbot_data_path()) / "plugin_data" / plugin_name
        else:
            self.data_dir = Path("data") / "plugin_data" / plugin_name

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._cache: dict[str, Any] = {}
        self._dirty: set = set()
        self._lock = threading.RLock()

        # 群组状态缓存
        self._group_states: dict[str, GroupStateData] = {}

        # 策略记忆缓存
        self._strategies: dict[str, StrategyEntryData] = {}

        # 学习数据缓存
        self._learning_data: dict[str, LearningData] = {}

        # 对话历史缓存（每个群最近100条）
        self._conversation_history: dict[str, list[ConversationRecord]] = {}

        # 群组UMO映射
        self._group_umo: dict[str, str] = {}

        # 群聊历史（用于离线蒸馏）
        self._group_history: dict[str, GroupHistory] = {}

        # 全局词表
        self._global_vocabulary: GlobalVocabulary | None = None

        # 相似度规则
        self._similarity_rules: dict[str, SimilarityRule] = {}

        # 正则规则
        self._regex_rules: dict[str, RegexRule] = {}

        # 加载持久化数据
        self._load_all()

    def _get_file_path(self, key: str) -> Path:
        """获取数据文件路径"""
        return self.data_dir / f"{key}.json"

    def _load_all(self):
        """加载所有持久化数据"""
        self._load_group_states()
        self._load_strategies()
        self._load_learning_data()
        self._load_group_umo()
        self._load_conversation_history()
        self._load_group_history()
        self._load_global_vocabulary()
        self._load_similarity_rules()
        self._load_regex_rules()

    def _load_json_file(self, filename: str) -> dict:
        """加载JSON文件"""
        file_path = self._get_file_path(filename)
        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_json_file(self, filename: str, data: Any):
        """保存JSON文件"""
        file_path = self._get_file_path(filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ==================== 群组状态 ====================

    def _load_group_states(self):
        """加载群组状态"""
        data = self._load_json_file("group_states")
        for group_id, state_dict in data.items():
            self._group_states[group_id] = GroupStateData(**state_dict)

    def _save_group_states(self):
        """保存群组状态"""
        data = {gid: asdict(state) for gid, state in self._group_states.items()}
        self._save_json_file("group_states", data)

    def get_group_state(self, group_id: str) -> GroupStateData:
        """获取群组状态，不存在则创建"""
        with self._lock:
            if group_id not in self._group_states:
                self._group_states[group_id] = GroupStateData(group_id=group_id)
                self._dirty.add("group_states")
            return self._group_states[group_id]

    def update_group_state(self, state: GroupStateData):
        """更新群组状态"""
        with self._lock:
            state.updated_at = time.time()
            self._group_states[state.group_id] = state
            self._dirty.add("group_states")

    def get_all_group_ids(self) -> list[str]:
        """获取所有群组ID"""
        return list(self._group_states.keys())

    # ==================== 策略记忆 ====================

    def _load_strategies(self):
        """加载策略记忆"""
        data = self._load_json_file("strategies")
        for strategy_id, strategy_dict in data.items():
            self._strategies[strategy_id] = StrategyEntryData(**strategy_dict)

    def _save_strategies(self):
        """保存策略记忆"""
        data = {sid: asdict(strategy) for sid, strategy in self._strategies.items()}
        self._save_json_file("strategies", data)

    def get_strategy(self, strategy_id: str) -> StrategyEntryData | None:
        """获取策略"""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> list[StrategyEntryData]:
        """获取所有策略"""
        return list(self._strategies.values())

    def add_strategy(self, strategy: StrategyEntryData):
        """添加策略"""
        with self._lock:
            self._strategies[strategy.id] = strategy
            self._dirty.add("strategies")

    def update_strategy(self, strategy: StrategyEntryData):
        """更新策略"""
        with self._lock:
            self._strategies[strategy.id] = strategy
            self._dirty.add("strategies")

    def get_strategies_by_state(self, flow_state: str) -> list[StrategyEntryData]:
        """按心流状态获取策略"""
        return [s for s in self._strategies.values() if s.flow_state == flow_state]

    # ==================== 学习数据 ====================

    def _load_learning_data(self):
        """加载学习数据"""
        data = self._load_json_file("learning_data")
        for key, learning_dict in data.items():
            self._learning_data[key] = LearningData(**learning_dict)

    def _save_learning_data(self):
        """保存学习数据"""
        data = {key: asdict(ld) for key, ld in self._learning_data.items()}
        self._save_json_file("learning_data", data)

    def get_learning_data(self, key: str = "default") -> LearningData:
        """获取学习数据"""
        if key not in self._learning_data:
            self._learning_data[key] = LearningData()
        return self._learning_data[key]

    def update_learning_data(self, learning_data: LearningData, key: str = "default"):
        """更新学习数据"""
        with self._lock:
            learning_data.last_updated = time.time()
            self._learning_data[key] = learning_data
            self._dirty.add("learning_data")

    # ==================== 群组UMO映射 ====================

    def _load_group_umo(self):
        """加载群组UMO映射"""
        self._group_umo = self._load_json_file("group_umo")

    def _save_group_umo(self):
        """保存群组UMO映射"""
        self._save_json_file("group_umo", self._group_umo)

    def set_group_umo(self, group_id: str, umo: str):
        """设置群组UMO"""
        with self._lock:
            self._group_umo[group_id] = umo
            self._dirty.add("group_umo")

    def get_group_umo(self, group_id: str) -> str | None:
        """获取群组UMO"""
        return self._group_umo.get(group_id)

    def get_active_groups(self) -> list[str]:
        """获取所有活跃群组（有UMO记录的）"""
        return list(self._group_umo.keys())

    # ==================== 对话历史 ====================

    def _load_conversation_history(self):
        """加载对话历史"""
        data = self._load_json_file("conversation_history")
        for group_id, records in data.items():
            self._conversation_history[group_id] = [
                ConversationRecord(**r) for r in records
            ]

    def _save_conversation_history(self):
        """保存对话历史"""
        data = {}
        for group_id, records in self._conversation_history.items():
            data[group_id] = [asdict(r) for r in records]
        self._save_json_file("conversation_history", data)

    def add_conversation_record(self, group_id: str, record: ConversationRecord):
        """添加对话记录"""
        with self._lock:
            if group_id not in self._conversation_history:
                self._conversation_history[group_id] = []

            self._conversation_history[group_id].append(record)

            # 只保留最近100条
            if len(self._conversation_history[group_id]) > 100:
                self._conversation_history[group_id] = self._conversation_history[
                    group_id
                ][-100:]

            self._dirty.add("conversation_history")

    def get_conversation_history(
        self, group_id: str, limit: int = 20
    ) -> list[ConversationRecord]:
        """获取对话历史"""
        if group_id not in self._conversation_history:
            return []
        return self._conversation_history[group_id][-limit:]

    def get_last_bot_reply(self, group_id: str) -> ConversationRecord | None:
        """获取最后一条机器人回复"""
        if group_id not in self._conversation_history:
            return None
        for record in reversed(self._conversation_history[group_id]):
            if record.role == "assistant":
                return record
        return None

    # ==================== 群聊历史（离线蒸馏用） ====================

    def _load_group_history(self):
        """加载群聊历史"""
        data = self._load_json_file("group_history")
        for group_id, history_dict in data.items():
            self._group_history[group_id] = GroupHistory(
                messages=history_dict.get("messages", []),
                last_distill_time=history_dict.get("last_distill_time", 0.0),
                total_messages=history_dict.get("total_messages", 0),
                last_seq=history_dict.get("last_seq", 0),
            )

    def _save_group_history(self):
        """保存群聊历史"""
        data = {}
        for group_id, history in self._group_history.items():
            data[group_id] = {
                "messages": history.messages,
                "last_distill_time": history.last_distill_time,
                "total_messages": history.total_messages,
                "last_seq": history.last_seq,
            }
        self._save_json_file("group_history", data)

    def add_group_message(
        self, group_id: str, user_id: str, user_name: str, content: str
    ) -> int:
        """添加群聊消息，返回消息序号"""
        with self._lock:
            if group_id not in self._group_history:
                self._group_history[group_id] = GroupHistory()

            history = self._group_history[group_id]
            history.last_seq += 1
            history.total_messages += 1

            msg = {
                "id": f"msg_{history.last_seq}",
                "seq": history.last_seq,
                "user_id": user_id,
                "user_name": user_name,
                "content": content,
                "timestamp": time.time(),
                "processed": False,
            }
            history.messages.append(msg)

            # 只保留最近5000条
            if len(history.messages) > 5000:
                history.messages = history.messages[-5000:]

            self._dirty.add("group_history")
            return history.last_seq

    def get_unprocessed_messages(
        self, group_id: str, limit: int = 1000
    ) -> list[dict]:
        """获取未处理的消息"""
        if group_id not in self._group_history:
            return []

        history = self._group_history[group_id]
        unprocessed = [m for m in history.messages if not m.get("processed", False)]
        return unprocessed[:limit]

    def mark_messages_processed(self, group_id: str, seq_list: list[int]):
        """标记消息为已处理"""
        with self._lock:
            if group_id not in self._group_history:
                return

            history = self._group_history[group_id]
            seq_set = set(seq_list)
            for msg in history.messages:
                if msg.get("seq") in seq_set:
                    msg["processed"] = True

            history.last_distill_time = time.time()
            self._dirty.add("group_history")

    def get_messages_by_seq_range(
        self, group_id: str, start_seq: int, end_seq: int
    ) -> list[dict]:
        """根据序号范围获取消息"""
        if group_id not in self._group_history:
            return []

        history = self._group_history[group_id]
        return [
            m
            for m in history.messages
            if start_seq <= m.get("seq", 0) <= end_seq
        ]

    def get_messages_before_seq(
        self, group_id: str, seq: int, count: int = 10
    ) -> list[dict]:
        """获取指定序号之前的消息"""
        if group_id not in self._group_history:
            return []

        history = self._group_history[group_id]
        before = [m for m in history.messages if m.get("seq", 0) < seq]
        return before[-count:] if len(before) > count else before

    # ==================== 全局词表 ====================

    def _load_global_vocabulary(self):
        """加载全局词表"""
        data = self._load_json_file("global_vocabulary")
        if data:
            self._global_vocabulary = GlobalVocabulary(
                vocabulary=data.get("vocabulary", []),
                idf_values=data.get("idf_values", {}),
                doc_count=data.get("doc_count", 0),
                last_updated=data.get("last_updated", time.time()),
            )
        else:
            self._global_vocabulary = GlobalVocabulary()

    def _save_global_vocabulary(self):
        """保存全局词表"""
        if self._global_vocabulary:
            data = {
                "vocabulary": self._global_vocabulary.vocabulary,
                "idf_values": self._global_vocabulary.idf_values,
                "doc_count": self._global_vocabulary.doc_count,
                "last_updated": self._global_vocabulary.last_updated,
            }
            self._save_json_file("global_vocabulary", data)

    def get_global_vocabulary(self) -> GlobalVocabulary:
        """获取全局词表"""
        if self._global_vocabulary is None:
            self._global_vocabulary = GlobalVocabulary()
        return self._global_vocabulary

    def update_global_vocabulary(
        self, vocabulary: list[str], idf_values: dict[str, float], doc_count: int
    ):
        """更新全局词表"""
        with self._lock:
            if self._global_vocabulary is None:
                self._global_vocabulary = GlobalVocabulary()
            self._global_vocabulary.vocabulary = vocabulary
            self._global_vocabulary.idf_values = idf_values
            self._global_vocabulary.doc_count = doc_count
            self._global_vocabulary.last_updated = time.time()
            self._dirty.add("global_vocabulary")

    # ==================== 相似度规则 ====================

    def _load_similarity_rules(self):
        """加载相似度规则"""
        data = self._load_json_file("similarity_rules")
        for rule_id, rule_dict in data.items():
            self._similarity_rules[rule_id] = SimilarityRule(
                id=rule_id,
                original_text=rule_dict.get("original_text", ""),
                threshold=rule_dict.get("threshold", 0.6),
                source_group=rule_dict.get("source_group", ""),
                source_seq=rule_dict.get("source_seq", 0),
                created_at=rule_dict.get("created_at", time.time()),
                use_count=rule_dict.get("use_count", 0),
                success_count=rule_dict.get("success_count", 0),
            )

    def _save_similarity_rules(self):
        """保存相似度规则"""
        data = {}
        for rule_id, rule in self._similarity_rules.items():
            data[rule_id] = {
                "original_text": rule.original_text,
                "threshold": rule.threshold,
                "source_group": rule.source_group,
                "source_seq": rule.source_seq,
                "created_at": rule.created_at,
                "use_count": rule.use_count,
                "success_count": rule.success_count,
            }
        self._save_json_file("similarity_rules", data)

    def get_all_similarity_rules(self) -> list[SimilarityRule]:
        """获取所有相似度规则"""
        return list(self._similarity_rules.values())

    def add_similarity_rule(self, rule: SimilarityRule):
        """添加相似度规则"""
        with self._lock:
            self._similarity_rules[rule.id] = rule
            self._dirty.add("similarity_rules")

    def update_similarity_rule_usage(self, rule_id: str, success: bool):
        """更新相似度规则使用情况"""
        with self._lock:
            if rule_id in self._similarity_rules:
                rule = self._similarity_rules[rule_id]
                rule.use_count += 1
                if success:
                    rule.success_count += 1
                self._dirty.add("similarity_rules")

    def clear_similarity_rules(self):
        """清空相似度规则"""
        with self._lock:
            self._similarity_rules.clear()
            self._dirty.add("similarity_rules")

    # ==================== 正则规则 ====================

    def _load_regex_rules(self):
        """加载正则规则"""
        data = self._load_json_file("regex_rules")
        for rule_id, rule_dict in data.items():
            self._regex_rules[rule_id] = RegexRule(
                id=rule_id,
                pattern=rule_dict.get("pattern", ""),
                trigger_count=rule_dict.get("trigger_count", 1),
                current_count=rule_dict.get("current_count", 0),
                created_at=rule_dict.get("created_at", time.time()),
                use_count=rule_dict.get("use_count", 0),
                success_count=rule_dict.get("success_count", 0),
            )

    def _save_regex_rules(self):
        """保存正则规则"""
        data = {}
        for rule_id, rule in self._regex_rules.items():
            data[rule_id] = {
                "pattern": rule.pattern,
                "trigger_count": rule.trigger_count,
                "current_count": rule.current_count,
                "created_at": rule.created_at,
                "use_count": rule.use_count,
                "success_count": rule.success_count,
            }
        self._save_json_file("regex_rules", data)

    def get_all_regex_rules(self) -> list[RegexRule]:
        """获取所有正则规则"""
        return list(self._regex_rules.values())

    def add_regex_rule(self, rule: RegexRule):
        """添加正则规则"""
        with self._lock:
            self._regex_rules[rule.id] = rule
            self._dirty.add("regex_rules")

    def update_regex_rule_count(self, rule_id: str, increment: int = 1) -> bool:
        """
        更新正则规则计数

        Returns:
            是否达到触发条件
        """
        with self._lock:
            if rule_id not in self._regex_rules:
                return False

            rule = self._regex_rules[rule_id]
            rule.current_count += increment
            rule.use_count += 1
            self._dirty.add("regex_rules")

            if rule.current_count >= rule.trigger_count:
                rule.current_count = 0
                return True
            return False

    def clear_regex_rules(self):
        """清空正则规则"""
        with self._lock:
            self._regex_rules.clear()
            self._dirty.add("regex_rules")

    # ==================== 通用方法 ====================

    def get(self, key: str, default: Any = None) -> Any:
        """获取通用数据"""
        if key in self._cache:
            return self._cache[key]

        data = self._load_json_file(key)
        self._cache[key] = data if data else default
        return self._cache[key]

    def set(self, key: str, value: Any):
        """设置通用数据"""
        with self._lock:
            self._cache[key] = value
            self._dirty.add(key)

    def save_dirty(self):
        """保存所有脏数据"""
        with self._lock:
            dirty_keys = list(self._dirty)
            self._dirty.clear()

        for key in dirty_keys:
            if key == "group_states":
                self._save_group_states()
            elif key == "strategies":
                self._save_strategies()
            elif key == "learning_data":
                self._save_learning_data()
            elif key == "group_umo":
                self._save_group_umo()
            elif key == "conversation_history":
                self._save_conversation_history()
            elif key == "group_history":
                self._save_group_history()
            elif key == "global_vocabulary":
                self._save_global_vocabulary()
            elif key == "similarity_rules":
                self._save_similarity_rules()
            elif key == "regex_rules":
                self._save_regex_rules()
            elif key in self._cache:
                self._save_json_file(key, self._cache[key])

    def save_all(self):
        """保存所有数据"""
        self._save_group_states()
        self._save_strategies()
        self._save_learning_data()
        self._save_group_umo()
        self._save_conversation_history()
        self._save_group_history()
        self._save_global_vocabulary()
        self._save_similarity_rules()
        self._save_regex_rules()

        for key in self._cache:
            self._save_json_file(key, self._cache[key])

        self._dirty.clear()

    def clear_group_data(self, group_id: str):
        """清除指定群组的所有数据"""
        with self._lock:
            if group_id in self._group_states:
                del self._group_states[group_id]
            if group_id in self._group_umo:
                del self._group_umo[group_id]
            if group_id in self._conversation_history:
                del self._conversation_history[group_id]

            self._dirty.update(["group_states", "group_umo", "conversation_history"])

    def clear_all_state(self):
        """清除所有状态数据"""
        with self._lock:
            self._group_states.clear()
            self._strategies.clear()
            self._learning_data.clear()
            self._group_umo.clear()
            self._conversation_history.clear()
            self._group_history.clear()
            self._global_vocabulary = GlobalVocabulary()
            self._similarity_rules.clear()
            self._regex_rules.clear()
            self._cache.clear()
            self._dirty.clear()

    def get_distillation_stats(self) -> dict:
        """获取蒸馏相关统计"""
        return {
            "total_groups": len(self._group_history),
            "total_similarity_rules": len(self._similarity_rules),
            "total_regex_rules": len(self._regex_rules),
            "vocabulary_size": len(self._global_vocabulary.vocabulary) if self._global_vocabulary else 0,
            "doc_count": self._global_vocabulary.doc_count if self._global_vocabulary else 0,
        }
