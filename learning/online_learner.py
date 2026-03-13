"""
在线强化学习

基于用户反馈实时优化策略：
- Q-learning 实现
- 状态编码
- 动作选择
- 奖励计算
"""

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class LearningConfig:
    """学习配置"""

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.2
    min_exploration_rate: float = 0.01
    exploration_decay: float = 0.995


class OnlineLearner:
    """
    在线强化学习

    使用 Q-learning 实时优化回复策略
    """

    def __init__(self, config, persistence=None):
        self.config = config
        self.persistence = persistence

        # 学习配置
        learning_config = config.get("learning", {})
        self.cfg = LearningConfig(
            learning_rate=learning_config.get("learning_rate", 0.1),
            discount_factor=learning_config.get("discount_factor", 0.95),
            exploration_rate=0.2,
            min_exploration_rate=0.01,
            exploration_decay=0.995,
        )

        # Q 表 {state: {action: q_value}}
        self._q_table: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # 状态-动作历史（用于学习）
        self._episode_history: list[tuple[str, str, float, str]] = []

        # 动作空间
        self.actions = ["reply", "wait", "initiate", "observe"]

        # 统计信息
        self._stats = {
            "total_updates": 0,
            "total_rewards": 0.0,
            "exploration_count": 0,
            "exploitation_count": 0,
        }

        # 加载持久化数据
        self._load_q_table()



    def _load_q_table(self):
        """加载 Q 表"""
        if self.persistence:
            learning_data = self.persistence.get_learning_data("q_table")
            if learning_data and learning_data.q_table:
                for state, actions in learning_data.q_table.items():
                    self._q_table[state] = defaultdict(float, actions)

    def _save_q_table(self):
        """保存 Q 表"""
        if self.persistence:
            from ..storage.persistence import LearningData

            q_table_dict = {
                state: dict(actions) for state, actions in self._q_table.items()
            }

            learning_data = LearningData(
                q_table=q_table_dict,
                episode_count=len(self._episode_history),
                total_reward=self._stats["total_rewards"],
            )

            self.persistence.update_learning_data(learning_data, "q_table")

    def get_best_action(self, state: str) -> tuple[str | None, float]:
        """
        获取最佳动作

        Args:
            state: 状态编码

        Returns:
            (action, q_value) 最佳动作及其 Q 值
        """
        if not self._q_table[state]:
            # 新状态，返回默认动作
            return None, 0.0

        # ε-greedy 策略
        if random.random() < self.cfg.exploration_rate:
            self._stats["exploration_count"] += 1
            action = random.choice(self.actions)
            return action, self._q_table[state][action]

        self._stats["exploitation_count"] += 1

        # 选择最大 Q 值的动作
        best_action = max(
            self._q_table[state].items(), key=lambda x: x[1], default=(None, 0.0)
        )

        return best_action

    def update(self, state: str, action: str, reward: float, next_state: str = None): # pyright: ignore[reportArgumentType]
        """
        更新 Q 值

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
        """
        self._stats["total_updates"] += 1
        self._stats["total_rewards"] += reward

        # Q-learning 更新
        current_q = self._q_table[state][action]

        if next_state and self._q_table[next_state]:
            max_next_q = max(self._q_table[next_state].values())
        else:
            max_next_q = 0.0

        # TD 更新
        new_q = current_q + self.cfg.learning_rate * (
            reward + self.cfg.discount_factor * max_next_q - current_q
        )

        self._q_table[state][action] = new_q

        # 记录历史
        self._episode_history.append((state, action, reward, next_state or ""))

        # 只保留最近1000条
        if len(self._episode_history) > 1000:
            self._episode_history = self._episode_history[-1000:]

        # 衰减探索率
        self._decay_exploration()

        # 定期保存
        if self._stats["total_updates"] % 100 == 0:
            self._save_q_table()

    def batch_update(self, episodes: list[tuple[str, str, float, str]]):
        """
        批量更新

        Args:
            episodes: [(state, action, reward, next_state), ...]
        """
        for state, action, reward, next_state in episodes:
            self.update(state, action, reward, next_state)

    def _decay_exploration(self):
        """衰减探索率"""
        self.cfg.exploration_rate = max(
            self.cfg.min_exploration_rate,
            self.cfg.exploration_rate * self.cfg.exploration_decay,
        )

    def encode_state(
        self,
        flow_state: str,
        group_activity: float,
        topic_coherence: float,
        relevance_to_bot: float,
        energy: float,
        reply_streak: int,
    ) -> str:
        """
        编码状态

        Args:
            flow_state: 心流状态
            group_activity: 群活跃度
            topic_coherence: 话题连贯性
            relevance_to_bot: 相关性
            energy: 能量值
            reply_streak: 连续回复次数

        Returns:
            状态编码字符串
        """
        # 离散化连续值
        activity_bucket = int(group_activity * 10)
        coherence_bucket = int(topic_coherence * 10)
        relevance_bucket = int(relevance_to_bot * 10)
        energy_bucket = int(energy * 10)
        streak_bucket = min(5, reply_streak)

        return f"{flow_state}_{activity_bucket}_{coherence_bucket}_{relevance_bucket}_{energy_bucket}_{streak_bucket}"

    def get_q_value(self, state: str, action: str) -> float:
        """获取 Q 值"""
        return self._q_table[state][action]

    def get_state_values(self, state: str) -> dict[str, float]:
        """获取状态的所有动作 Q 值"""
        return dict(self._q_table[state])

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = self._stats["exploration_count"] + self._stats["exploitation_count"]

        return {
            **self._stats,
            "exploration_rate": self.cfg.exploration_rate,
            "states_learned": len(self._q_table),
            "exploration_ratio": (
                self._stats["exploration_count"] / total if total > 0 else 0
            ),
            "avg_reward": (
                self._stats["total_rewards"] / self._stats["total_updates"]
                if self._stats["total_updates"] > 0
                else 0
            ),
        }

    def get_q_table_summary(self) -> dict:
        """获取 Q 表摘要"""
        summary = {}

        for state, actions in self._q_table.items():
            if actions:
                best_action = max(actions.items(), key=lambda x: x[1])
                summary[state] = {
                    "best_action": best_action[0],
                    "best_q": best_action[1],
                    "action_count": len(actions),
                }

        return summary

    def reset(self):
        """重置学习器"""
        self._q_table.clear()
        self._episode_history.clear()
        self._stats = {
            "total_updates": 0,
            "total_rewards": 0.0,
            "exploration_count": 0,
            "exploitation_count": 0,
        }
        self.cfg.exploration_rate = 0.2

        if self.persistence:
            self.persistence.update_learning_data(
                type(
                    "LearningData",
                    (),
                    {"q_table": {}, "episode_count": 0, "total_reward": 0.0},
                )(),
                "q_table",
            )

    def force_exploration(self, rate: float):
        """强制设置探索率"""
        self.cfg.exploration_rate = min(1.0, max(0.0, rate))

    def import_q_table(self, q_table: dict[str, dict[str, float]]):
        """导入 Q 表"""
        for state, actions in q_table.items():
            self._q_table[state] = defaultdict(float, actions)

    def export_q_table(self) -> dict[str, dict[str, float]]:
        """导出 Q 表"""
        return {state: dict(actions) for state, actions in self._q_table.items()}
