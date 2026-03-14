"""
决策引擎

整合三路学习机制，做出回复决策：
1. 策略检索：从历史策略库中检索相似场景的成功策略
2. 离线蒸馏：使用预蒸馏的策略规则
3. 在线学习：基于强化学习的实时策略优化

决策流程：
1. 检索策略库中相似上下文的策略
2. 如果有高置信度策略，直接使用
3. 否则，结合在线学习模型和启发式规则决策
4. 记录决策结果，等待反馈用于学习
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from astrbot.api import logger


class ActionType(Enum):
    """动作类型"""

    REPLY = "reply"  # 回复
    WAIT = "wait"  # 等待
    INITIATE = "initiate"  # 主动发起话题
    OBSERVE = "observe"  # 继续观察


class DecisionSource(Enum):
    """决策来源"""

    STRATEGY = "strategy"  # 策略库检索
    DISTILLATION = "distillation"  # 离线蒸馏
    ONLINE_LEARNING = "online"  # 在线学习
    HEURISTIC = "heuristic"  # 启发式规则
    FORCED = "forced"  # 强制决策（如被@）


@dataclass
class Decision:
    """决策结果"""

    action: ActionType
    confidence: float
    source: DecisionSource
    should_act: bool = False
    reply_probability: float = 0.0
    delay_seconds: float = 0.0
    context_hash: str = ""
    strategy_id: str | None = None
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "source": self.source.value,
            "should_act": self.should_act,
            "reply_probability": self.reply_probability,
            "delay_seconds": self.delay_seconds,
            "context_hash": self.context_hash,
            "strategy_id": self.strategy_id,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


@dataclass
class DecisionOutcome:
    """决策结果反馈"""

    decision_id: str
    decision: Decision
    reward: float
    feedback_type: str
    timestamp: float = field(default_factory=time.time)


class DecisionEngine:
    """
    决策引擎

    整合三路学习机制，做出回复决策
    """

    def __init__(
        self, config, strategy_store, online_learner=None, offline_distiller=None,
        rule_matcher=None
    ):
        self.config = config
        self.strategy_store = strategy_store
        self.online_learner = online_learner
        self.offline_distiller = offline_distiller
        self.rule_matcher = rule_matcher

        # 配置参数
        learning_config = config.get("learning", {})
        self.enable_online_learning = learning_config.get(
            "enable_online_learning", True
        )

        offline_distillation_config = config.get("offline_distillation", {})
        self.enable_offline_distillation = offline_distillation_config.get(
            "enabled", True
        )

        # 性能模式
        self.performance_mode = config.get("performance_mode", "balanced")

        # 决策历史（用于反馈学习）
        self._pending_outcomes: dict[str, DecisionOutcome] = {}
        self._decision_history: list[DecisionOutcome] = []

        # 统计信息
        self._stats = {
            "total_decisions": 0,
            "strategy_hits": 0,
            "online_decisions": 0,
            "heuristic_decisions": 0,
            "forced_decisions": 0,
            "rule_matches": 0,
        }

    def set_rule_matcher(self, rule_matcher):
        """设置规则匹配器"""
        self.rule_matcher = rule_matcher

    async def decide(self, state, context, energy: float = 0.8) -> Decision:
        """
        做出决策

        Args:
            state: 群组状态（GroupState）
            context: 分析后的上下文（AnalyzedContext）
            energy: 当前能量值

        Returns:
            Decision: 决策结果
        """
        self._stats["total_decisions"] += 1

        # 计算上下文哈希
        context_hash = self._compute_context_hash(state, context)

        # 强制决策：被@或高相关性
        if context.relevance_to_bot >= 0.9:
            self._stats["forced_decisions"] += 1
            return Decision(
                action=ActionType.REPLY,
                confidence=1.0,
                source=DecisionSource.FORCED,
                should_act=True,
                reply_probability=1.0,
                delay_seconds=0.5,
                context_hash=context_hash,
                reasoning="被@或高相关性触发强制回复",
            )

        # 能量检查
        if energy < 0.2:
            return Decision(
                action=ActionType.OBSERVE,
                confidence=0.8,
                source=DecisionSource.HEURISTIC,
                should_act=False,
                reply_probability=0.0,
                context_hash=context_hash,
                reasoning=f"能量过低({energy:.2f})，进入观察",
            )

        # 1. 策略库检索
        strategy_decision = await self._retrieve_strategy(state, context, context_hash)
        if strategy_decision and strategy_decision.confidence >= 0.7:
            self._stats["strategy_hits"] += 1
            strategy_decision.context_hash = context_hash
            return strategy_decision

        # 2. 离线蒸馏策略
        if self.enable_offline_distillation and self.offline_distiller:
            distilled_decision = await self._distilled_decision(
                state, context, context_hash
            )
            if distilled_decision and distilled_decision.confidence >= 0.6:
                distilled_decision.context_hash = context_hash
                return distilled_decision

        # 3. 在线学习决策
        if self.enable_online_learning and self.online_learner:
            online_decision = await self._online_decision(state, context, context_hash)
            if online_decision:
                self._stats["online_decisions"] += 1
                online_decision.context_hash = context_hash
                return online_decision

        # 4. 启发式规则决策
        self._stats["heuristic_decisions"] += 1
        heuristic_decision = self._heuristic_decision(
            state, context, energy, context_hash
        )

        return heuristic_decision

    def _compute_context_hash(self, state, context) -> str:
        """计算上下文哈希"""
        context_dict = {
            "flow_state": state.flow_state.value
            if hasattr(state.flow_state, "value")
            else str(state.flow_state),
            "group_activity": round(getattr(context, "group_activity", 0.5), 2),
            "topic_coherence": round(getattr(context, "topic_coherence", 0.5), 2),
            "relevance_to_bot": round(getattr(context, "relevance_to_bot", 0.0), 2),
            "message_count_1m": getattr(context, "message_count_1m", 0) // 5,  # 分桶
        }
        content = json.dumps(context_dict, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def _retrieve_strategy(
        self, state, context, context_hash: str
    ) -> Decision | None:
        """从策略库检索"""
        try:
            strategies = self.strategy_store.get_all_strategies()
            if not strategies:
                return None

            flow_state_value = (
                state.flow_state.value
                if hasattr(state.flow_state, "value")
                else str(state.flow_state)
            )

            # 筛选同一心流状态的策略
            matching_strategies = [
                s
                for s in strategies
                if s.flow_state == flow_state_value and s.total_count >= 3
            ]

            if not matching_strategies:
                return None

            # 按成功率排序
            matching_strategies.sort(key=lambda s: s.success_rate, reverse=True)

            best_strategy = matching_strategies[0]

            if best_strategy.success_rate < 0.5:
                return None

            action = ActionType(best_strategy.action_type)

            return Decision(
                action=action,
                confidence=min(0.9, best_strategy.success_rate),
                source=DecisionSource.STRATEGY,
                should_act=action in [ActionType.REPLY, ActionType.INITIATE],
                reply_probability=best_strategy.success_rate,
                strategy_id=best_strategy.id,
                reasoning=f"策略库匹配(id={best_strategy.id}, 成功率={best_strategy.success_rate:.2f})",
            )
        except Exception as e:
            logger.error(f"策略检索失败: {e}")
            return None

    async def _distilled_decision(
        self, state, context, context_hash: str
    ) -> Decision | None:
        """离线蒸馏决策"""
        if not self.offline_distiller:
            return None

        try:
            rules = self.offline_distiller.get_rules()
            if not rules:
                return None

            group_activity = getattr(context, "group_activity", 0.5)
            topic_coherence = getattr(context, "topic_coherence", 0.5)
            relevance = getattr(context, "relevance_to_bot", 0.0)
            flow_state = (
                state.flow_state.value
                if hasattr(state.flow_state, "value")
                else str(state.flow_state)
            )

            for rule in rules:
                pattern = rule.get("pattern", {})
                conditions = pattern.get("conditions", {})

                if (
                    conditions.get("flow_state")
                    and conditions["flow_state"] != flow_state
                ):
                    continue

                if (
                    conditions.get("min_activity")
                    and group_activity < conditions["min_activity"]
                ):
                    continue

                if (
                    conditions.get("min_coherence")
                    and topic_coherence < conditions["min_coherence"]
                ):
                    continue

                if (
                    conditions.get("min_relevance")
                    and relevance < conditions["min_relevance"]
                ):
                    continue

                # 匹配成功
                strategy = rule.get("strategy", {})
                action_type = strategy.get("action", "wait")
                confidence = rule.get("confidence", 0.6)

                return Decision(
                    action=ActionType(action_type),
                    confidence=confidence,
                    source=DecisionSource.DISTILLATION,
                    should_act=action_type in ["reply", "initiate"],
                    reply_probability=confidence,
                    reasoning=f"蒸馏规则匹配: {rule.get('description', '')}",
                )

            return None
        except Exception as e:
            logger.error(f"蒸馏决策失败: {e}")
            return None

    async def _online_decision(
        self, state, context, context_hash: str
    ) -> Decision | None:
        """在线学习决策"""
        if not self.online_learner:
            return None

        try:
            state_vec = self._encode_state(state, context)
            action, q_value = self.online_learner.get_best_action(state_vec)

            if action is None:
                return None

            confidence = min(0.9, max(0.3, (q_value + 1) / 2))  # 归一化Q值

            return Decision(
                action=ActionType(action),
                confidence=confidence,
                source=DecisionSource.ONLINE_LEARNING,
                should_act=action in ["reply", "initiate"],
                reply_probability=confidence,
                reasoning=f"在线学习决策(q={q_value:.2f})",
            )
        except Exception as e:
            logger.error(f"在线学习决策失败: {e}")
            return None

    def _heuristic_decision(
        self, state, context, energy: float, context_hash: str
    ) -> Decision:
        """启发式规则决策"""
        flow_state = (
            state.flow_state.value
            if hasattr(state.flow_state, "value")
            else str(state.flow_state)
        )
        relevance = getattr(context, "relevance_to_bot", 0.0)
        topic_coherence = getattr(context, "topic_coherence", 0.5)

        # 基础回复概率
        base_prob = 0.2

        # 根据心流状态调整
        if flow_state == "active":
            base_prob = 0.6
        elif flow_state == "flow":
            base_prob = 0.35
        elif flow_state == "observer":
            base_prob = 0.1
        elif flow_state == "fatigue":
            base_prob = 0.0

        # 根据相关性调整
        relevance_bonus = relevance * 0.3

        # 根据话题连贯性调整
        coherence_bonus = topic_coherence * 0.1

        # 根据能量调整
        energy_modifier = 0.5 + energy * 0.5

        # 最终概率
        final_prob = (base_prob + relevance_bonus + coherence_bonus) * energy_modifier
        final_prob = min(1.0, max(0.0, final_prob))

        # 决策
        should_act = final_prob >= 0.3
        action = ActionType.REPLY if should_act else ActionType.WAIT

        # 计算延迟
        if should_act:
            delay = 1.0 + (1.0 - final_prob) * 5.0  # 低概率时延迟更长
        else:
            delay = 0.0

        return Decision(
            action=action,
            confidence=final_prob,
            source=DecisionSource.HEURISTIC,
            should_act=should_act,
            reply_probability=final_prob,
            delay_seconds=delay,
            context_hash=context_hash,
            reasoning=f"启发式决策: flow={flow_state}, rel={relevance:.2f}, energy={energy:.2f}",
        )

    def _encode_state(self, state, context) -> str:
        """编码状态向量"""
        flow_state = (
            state.flow_state.value
            if hasattr(state.flow_state, "value")
            else str(state.flow_state)
        )

        # 离散化连续值
        activity_bucket = int(getattr(context, "group_activity", 0.5) * 10)
        coherence_bucket = int(getattr(context, "topic_coherence", 0.5) * 10)
        relevance_bucket = int(getattr(context, "relevance_to_bot", 0.0) * 10)
        energy_bucket = int(state.energy * 10)
        streak_bucket = min(5, state.reply_streak)

        return f"{flow_state}_{activity_bucket}_{coherence_bucket}_{relevance_bucket}_{energy_bucket}_{streak_bucket}"

    def record_outcome(
        self, decision: Decision, reward: float, feedback_type: str
    ) -> str:
        """
        记录决策结果

        Args:
            decision: 决策
            reward: 奖励值
            feedback_type: 反馈类型

        Returns:
            决策ID
        """
        decision_id = f"{decision.context_hash}_{int(time.time() * 1000)}"

        outcome = DecisionOutcome(
            decision_id=decision_id,
            decision=decision,
            reward=reward,
            feedback_type=feedback_type,
        )

        self._decision_history.append(outcome)

        # 只保留最近1000条
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-1000:]

        # 更新策略库
        if decision.strategy_id and decision.source == DecisionSource.STRATEGY:
            strategy = self.strategy_store.get_strategy(decision.strategy_id)
            if strategy:
                if reward > 0:
                    strategy.success_count += 1
                strategy.total_count += 1
                self.strategy_store.update_strategy(strategy)

        # 更新在线学习
        if self.enable_online_learning and self.online_learner:
            state_vec = decision.metadata.get("state_vec", "")
            if state_vec:
                self.online_learner.update(state_vec, decision.action.value, reward)

        logger.debug(
            f"决策结果记录: id={decision_id}, action={decision.action.value}, "
            f"reward={reward:.2f}, feedback={feedback_type}"
        )

        return decision_id

    def get_stats(self) -> dict[str, Any]:
        """获取决策统计"""
        total = self._stats["total_decisions"]
        if total == 0:
            return self._stats

        return {
            **self._stats,
            "strategy_hit_rate": self._stats["strategy_hits"] / total,
            "online_rate": self._stats["online_decisions"] / total,
            "heuristic_rate": self._stats["heuristic_decisions"] / total,
            "forced_rate": self._stats["forced_decisions"] / total,
        }

    def get_recent_decisions(self, limit: int = 20) -> list[dict]:
        """获取最近的决策记录"""
        return [
            {
                "decision_id": o.decision_id,
                "action": o.decision.action.value,
                "source": o.decision.source.value,
                "confidence": o.decision.confidence,
                "reward": o.reward,
                "feedback_type": o.feedback_type,
                "timestamp": o.timestamp,
            }
            for o in self._decision_history[-limit:]
        ]
