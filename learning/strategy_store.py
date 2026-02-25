"""
策略库存储

存储和管理策略记忆：
- 策略存储与检索
- 相似度计算
- 成功率统计
- 策略老化
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from astrbot.api import logger

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class StrategyEntry:
    """策略条目"""

    id: str
    context_hash: str
    flow_state: str
    action_type: str  # reply, wait, initiate, observe
    action_params: dict[str, Any] = field(default_factory=dict)
    context_features: dict[str, float] = field(default_factory=dict)
    success_count: int = 0
    total_count: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_success: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyEntry":
        return cls(**data)


class StrategyStore:
    """
    策略库存储

    存储和管理策略记忆
    """

    def __init__(self, persistence, config=None):
        self.persistence = persistence
        self.config = config or {}

        # 策略缓存
        self._strategies: dict[str, StrategyEntry] = {}

        # 上下文向量索引（用于相似度检索）
        self._context_index: dict[
            str, list[str]
        ] = {}  # {flow_state: [strategy_id, ...]}

        # 配置
        learning_config = self.config.get("learning", {})
        self.max_strategies = learning_config.get("strategy_memory_size", 1000)
        self.min_samples = 3  # 最小采样次数才纳入检索
        self.aging_factor = 0.99  # 老化因子
        self.similarity_threshold = 0.7  # 相似度阈值

        # 加载持久化数据
        self._load_strategies()

        logger.info(f"策略库已加载 {len(self._strategies)} 条策略")

    def _load_strategies(self):
        """加载策略"""
        strategies = self.persistence.get_all_strategies()
        for strategy in strategies:
            entry = StrategyEntry(
                id=strategy.id,
                context_hash=strategy.context_hash,
                flow_state=strategy.flow_state,
                action_type=strategy.action_type,
                action_params=strategy.action_params,
                success_count=strategy.success_count,
                total_count=strategy.total_count,
                last_used=strategy.last_used,
                created_at=strategy.created_at,
            )
            self._strategies[entry.id] = entry

            # 更新索引
            if entry.flow_state not in self._context_index:
                self._context_index[entry.flow_state] = []
            self._context_index[entry.flow_state].append(entry.id)

    def store(self, strategy: StrategyEntry) -> str:
        """存储策略"""
        # 检查是否已存在
        existing = self._find_similar_strategy(strategy)

        if existing:
            # 更新已有策略
            existing.total_count += 1
            existing.last_used = time.time()
            return existing.id

        # 创建新策略
        if len(self._strategies) >= self.max_strategies:
            self._evict_old_strategies()

        self._strategies[strategy.id] = strategy

        # 更新索引
        if strategy.flow_state not in self._context_index:
            self._context_index[strategy.flow_state] = []
        self._context_index[strategy.flow_state].append(strategy.id)

        # 持久化
        self.persistence.add_strategy(self._to_storage_entry(strategy))

        return strategy.id

    def retrieve(
        self, flow_state: str, context_features: dict[str, float], top_k: int = 5
    ) -> list[StrategyEntry]:
        """
        检索相似策略

        Args:
            flow_state: 心流状态
            context_features: 上下文特征
            top_k: 返回数量

        Returns:
            相似策略列表
        """
        # 获取同状态的策略ID
        strategy_ids = self._context_index.get(flow_state, [])

        if not strategy_ids:
            return []

        # 计算相似度
        scored = []
        for sid in strategy_ids:
            strategy = self._strategies.get(sid)
            if not strategy:
                continue

            if strategy.total_count < self.min_samples:
                continue

            similarity = self._calc_similarity(
                context_features, strategy.context_features
            )

            if similarity >= self.similarity_threshold:
                # 考虑成功率和时效性
                score = (
                    similarity * strategy.success_rate * self._get_freshness(strategy)
                )
                scored.append((strategy, score))

        # 排序并返回
        scored.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in scored[:top_k]]

    def get_strategy(self, strategy_id: str) -> StrategyEntry | None:
        """获取策略"""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> list[StrategyEntry]:
        """获取所有策略"""
        return list(self._strategies.values())

    def update_strategy(self, strategy: StrategyEntry):
        """更新策略"""
        self._strategies[strategy.id] = strategy
        self.persistence.update_strategy(self._to_storage_entry(strategy))

    def update_success_rate(self, strategy_id: str, success: bool):
        """更新成功率"""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return

        strategy.total_count += 1
        if success:
            strategy.success_count += 1
            strategy.last_success = time.time()

        strategy.last_used = time.time()

        self.persistence.update_strategy(self._to_storage_entry(strategy))

    def get_stats(self, group_id: str = None) -> dict[str, Any]:
        """获取统计信息"""
        if not self._strategies:
            return {
                "total_strategies": 0,
                "avg_success_rate": 0.0,
                "hit_rate": 0.0,
            }

        total = len(self._strategies)
        success_rates = [s.success_rate for s in self._strategies.values()]

        # 计算命中率（成功次数 / 总使用次数）
        total_uses = sum(s.total_count for s in self._strategies.values())
        total_successes = sum(s.success_count for s in self._strategies.values())
        hit_rate = total_successes / total_uses if total_uses > 0 else 0.0

        return {
            "total_strategies": total,
            "avg_success_rate": sum(success_rates) / total if total > 0 else 0.0,
            "hit_rate": hit_rate,
            "by_state": self._get_stats_by_state(),
        }

    def _get_stats_by_state(self) -> dict[str, dict]:
        """按状态统计"""
        stats = {}
        for state in self._context_index:
            strategies = [
                self._strategies[sid]
                for sid in self._context_index[state]
                if sid in self._strategies
            ]
            if strategies:
                stats[state] = {
                    "count": len(strategies),
                    "avg_success_rate": sum(s.success_rate for s in strategies)
                    / len(strategies),
                }
        return stats

    def _find_similar_strategy(self, strategy: StrategyEntry) -> StrategyEntry | None:
        """查找相似策略"""
        for existing in self._strategies.values():
            if existing.flow_state != strategy.flow_state:
                continue
            if existing.action_type != strategy.action_type:
                continue

            similarity = self._calc_similarity(
                strategy.context_features, existing.context_features
            )

            if similarity >= 0.95:  # 几乎相同
                return existing

        return None

    def _calc_similarity(
        self, features1: dict[str, float], features2: dict[str, float]
    ) -> float:
        """计算特征相似度（余弦相似度）"""
        if not features1 or not features2:
            return 0.0

        # 获取共同的键
        common_keys = set(features1.keys()) & set(features2.keys())

        if not common_keys:
            return 0.0

        if HAS_NUMPY:
            vec1 = np.array([features1[k] for k in common_keys])
            vec2 = np.array([features2[k] for k in common_keys])

            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot / (norm1 * norm2))
        else:
            # 纯Python实现
            dot = sum(features1[k] * features2[k] for k in common_keys)
            norm1 = math.sqrt(sum(v**2 for v in features1.values()))
            norm2 = math.sqrt(sum(v**2 for v in features2.values()))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot / (norm1 * norm2)

    def _get_freshness(self, strategy: StrategyEntry) -> float:
        """获取策略新鲜度（时间衰减）"""
        age_hours = (time.time() - strategy.last_used) / 3600
        return self.aging_factor**age_hours

    def _evict_old_strategies(self):
        """淘汰旧策略"""
        # 按成功率和使用时间排序
        strategies = list(self._strategies.values())
        strategies.sort(key=lambda s: s.success_rate * self._get_freshness(s))

        # 淘汰后10%
        evict_count = len(strategies) // 10

        for strategy in strategies[:evict_count]:
            del self._strategies[strategy.id]

            # 更新索引
            if strategy.flow_state in self._context_index:
                if strategy.id in self._context_index[strategy.flow_state]:
                    self._context_index[strategy.flow_state].remove(strategy.id)

    def _to_storage_entry(self, strategy: StrategyEntry):
        """转换为存储格式"""
        from storage.persistence import StrategyEntryData

        return StrategyEntryData(
            id=strategy.id,
            context_hash=strategy.context_hash,
            flow_state=strategy.flow_state,
            action_type=strategy.action_type,
            action_params=strategy.action_params,
            success_count=strategy.success_count,
            total_count=strategy.total_count,
            last_used=strategy.last_used,
            created_at=strategy.created_at,
        )

    def clear(self):
        """清空策略库"""
        self._strategies.clear()
        self._context_index.clear()

    def generate_context_hash(self, context_features: dict[str, float]) -> str:
        """生成上下文哈希"""
        # 离散化连续值
        discretized = {k: round(v, 1) for k, v in context_features.items()}
        content = json.dumps(discretized, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
