"""
持久化管理器

负责插件数据的持久化存储，数据存储在 data/plugins/astrbot_plugin_group_chat/ 目录下
遵循 AstrBot 插件开发规范：持久化数据存储于 data 目录下，而非插件自身目录
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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


class PersistenceManager:
    """
    持久化管理器

    数据存储在 data/plugins/astrbot_plugin_group_chat/ 目录下
    使用内存缓存 + 异步持久化策略
    """

    def __init__(self, plugin_name: str = "astrbot_plugin_group_chat"):
        self.plugin_name = plugin_name
        self.data_dir = Path("data") / "plugins" / plugin_name
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
            elif key in self._cache:
                self._save_json_file(key, self._cache[key])

    def save_all(self):
        """保存所有数据"""
        self._save_group_states()
        self._save_strategies()
        self._save_learning_data()
        self._save_group_umo()
        self._save_conversation_history()

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
            self._cache.clear()
            self._dirty.clear()
