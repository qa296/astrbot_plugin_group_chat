"""
离线知识蒸馏

基于 LLM 自玩模拟生成策略规则：
1. 生成模拟群聊场景
2. 让 LLM 决定最佳回复策略
3. 蒸馏为结构化规则
4. 存储到规则库

参考 PRINCIPLES 论文的方法
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger


@dataclass
class DistilledRule:
    """蒸馏规则"""

    id: str
    pattern: dict[str, Any]
    strategy: dict[str, Any]
    confidence: float
    description: str
    created_at: float = field(default_factory=time.time)
    use_count: int = 0
    success_count: int = 0


class OfflineDistiller:
    """
    离线知识蒸馏

    通过 LLM 自玩模拟生成策略规则
    """

    def __init__(self, context, config, persistence=None):
        self.context = context
        self.config = config
        self.persistence = persistence

        # 蒸馏规则
        self._rules: dict[str, DistilledRule] = {}

        # 配置
        learning_config = config.get("learning", {})
        self.enabled = learning_config.get("enable_offline_distillation", True)
        self.max_rules = 100

        # 预定义的场景模板
        self._scenario_templates = self._create_scenario_templates()

        # 加载已蒸馏的规则
        self._load_rules()

        logger.info(f"离线蒸馏器初始化完成，已加载 {len(self._rules)} 条规则")

    def _create_scenario_templates(self) -> list[dict]:
        """创建场景模板"""
        return [
            {
                "name": "用户提问",
                "context": {
                    "relevance_to_bot": 0.8,
                    "is_direct_question": True,
                    "group_activity": 0.3,
                },
                "expected_action": "reply",
                "description": "用户直接向机器人提问",
            },
            {
                "name": "被@触发",
                "context": {
                    "relevance_to_bot": 1.0,
                    "is_at_bot": True,
                    "group_activity": 0.5,
                },
                "expected_action": "reply",
                "description": "用户@机器人",
            },
            {
                "name": "话题讨论中",
                "context": {
                    "topic_coherence": 0.7,
                    "group_activity": 0.6,
                    "relevance_to_bot": 0.3,
                },
                "expected_action": "wait",
                "description": "群内正在讨论话题，相关性低",
            },
            {
                "name": "沉默后活跃",
                "context": {
                    "time_since_last_bot_reply": 300,
                    "group_activity": 0.5,
                    "topic_coherence": 0.5,
                },
                "expected_action": "initiate",
                "description": "群沉寂数分钟后开始活跃",
            },
            {
                "name": "连续对话",
                "context": {
                    "reply_streak": 2,
                    "energy": 0.6,
                    "group_activity": 0.7,
                },
                "expected_action": "reply",
                "description": "正在进行连续对话",
            },
            {
                "name": "高活跃群",
                "context": {
                    "group_activity": 0.9,
                    "active_users": 8,
                    "relevance_to_bot": 0.2,
                },
                "expected_action": "observe",
                "description": "群非常活跃但与机器人无关",
            },
            {
                "name": "疲劳恢复",
                "context": {
                    "flow_state": "fatigue",
                    "energy": 0.3,
                    "recovery_time_elapsed": True,
                },
                "expected_action": "observe",
                "description": "从疲劳状态恢复",
            },
            {
                "name": "低活跃群",
                "context": {
                    "group_activity": 0.1,
                    "message_count_5m": 1,
                    "relevance_to_bot": 0.0,
                },
                "expected_action": "observe",
                "description": "群几乎无活动",
            },
        ]

    def _load_rules(self):
        """加载规则"""
        if self.persistence:
            rules_data = self.persistence.get("distilled_rules", {})
            for rule_id, rule_dict in rules_data.items():
                self._rules[rule_id] = DistilledRule(
                    id=rule_id,
                    pattern=rule_dict.get("pattern", {}),
                    strategy=rule_dict.get("strategy", {}),
                    confidence=rule_dict.get("confidence", 0.5),
                    description=rule_dict.get("description", ""),
                    created_at=rule_dict.get("created_at", time.time()),
                    use_count=rule_dict.get("use_count", 0),
                    success_count=rule_dict.get("success_count", 0),
                )

    def _save_rules(self):
        """保存规则"""
        if self.persistence:
            rules_data = {
                rule_id: {
                    "pattern": rule.pattern,
                    "strategy": rule.strategy,
                    "confidence": rule.confidence,
                    "description": rule.description,
                    "created_at": rule.created_at,
                    "use_count": rule.use_count,
                    "success_count": rule.success_count,
                }
                for rule_id, rule in self._rules.items()
            }
            self.persistence.set("distilled_rules", rules_data)
            self.persistence.save_dirty()

    async def distill(self) -> int:
        """
        执行蒸馏

        Returns:
            生成的规则数量
        """
        if not self.enabled:
            logger.info("离线蒸馏已禁用")
            return 0

        logger.info("开始离线蒸馏...")

        new_rules = 0

        # 1. 基于场景模板生成规则
        for template in self._scenario_templates:
            rule = self._create_rule_from_template(template)
            if rule:
                self._rules[rule.id] = rule
                new_rules += 1

        # 2. 尝试使用 LLM 生成更多规则
        llm_rules = await self._distill_with_llm()
        for rule in llm_rules:
            if rule.id not in self._rules:
                self._rules[rule.id] = rule
                new_rules += 1

        # 3. 限制规则数量
        if len(self._rules) > self.max_rules:
            self._prune_rules()

        # 保存
        self._save_rules()

        logger.info(
            f"离线蒸馏完成，生成 {new_rules} 条规则，总计 {len(self._rules)} 条"
        )

        return new_rules

    def _create_rule_from_template(self, template: dict) -> DistilledRule | None:
        """从模板创建规则"""
        rule_id = f"rule_template_{template['name']}"

        return DistilledRule(
            id=rule_id,
            pattern={
                "conditions": template["context"],
            },
            strategy={
                "action": template["expected_action"],
                "params": {},
            },
            confidence=0.8,
            description=template["description"],
        )

    async def _distill_with_llm(self) -> list[DistilledRule]:
        """使用 LLM 蒸馏规则"""
        rules = []

        try:
            provider = self.context.get_using_provider()
            if not provider:
                logger.warning("无可用 LLM 提供商，跳过 LLM 蒸馏")
                return rules

            # 构建 prompt
            prompt = self._build_distillation_prompt()

            # 调用 LLM
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                image_urls=[],
                system_prompt=self._get_system_prompt(),
            )

            if response and response.completion_text:
                # 解析 LLM 返回的规则
                rules = self._parse_llm_rules(response.completion_text)

        except Exception as e:
            logger.error(f"LLM 蒸馏失败: {e}")

        return rules

    def _build_distillation_prompt(self) -> str:
        """构建蒸馏 prompt"""
        return """请分析群聊场景，为机器人生成回复策略规则。

输出格式要求（JSON数组）：
[
  {
    "name": "规则名称",
    "conditions": {
      "group_activity_min": 0.0-1.0,
      "relevance_to_bot_min": 0.0-1.0,
      "topic_coherence_min": 0.0-1.0,
      "flow_state": "observer/flow/active/fatigue",
      "energy_min": 0.0-1.0
    },
    "action": "reply/wait/initiate/observe",
    "confidence": 0.0-1.0,
    "description": "规则描述"
  }
]

请生成5-10条规则，覆盖不同场景：
1. 高活跃+高相关性场景
2. 中等活跃+低相关性场景
3. 低活跃场景
4. 连续对话场景
5. 话题转换场景
6. 疲劳恢复场景
"""

    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个群聊策略专家，负责为聊天机器人生成回复决策规则。
规则应该考虑：群活跃度、话题相关性、机器人能量、心流状态等因素。
目标是让机器人的回复更自然、更人性化。"""

    def _parse_llm_rules(self, text: str) -> list[DistilledRule]:
        """解析 LLM 返回的规则"""
        rules = []

        try:
            # 尝试提取 JSON
            import re

            json_match = re.search(r"\[[\s\S]*\]", text)

            if json_match:
                rules_data = json.loads(json_match.group())

                for i, rule_dict in enumerate(rules_data):
                    rule_id = f"rule_llm_{int(time.time())}_{i}"

                    rule = DistilledRule(
                        id=rule_id,
                        pattern={"conditions": rule_dict.get("conditions", {})},
                        strategy={
                            "action": rule_dict.get("action", "wait"),
                            "params": {},
                        },
                        confidence=rule_dict.get("confidence", 0.6),
                        description=rule_dict.get(
                            "description", rule_dict.get("name", "")
                        ),
                    )

                    rules.append(rule)

        except Exception as e:
            logger.error(f"解析 LLM 规则失败: {e}")

        return rules

    def _prune_rules(self):
        """修剪规则"""
        # 按成功率和使用次数排序
        sorted_rules = sorted(
            self._rules.items(), key=lambda x: (x[1].success_count, x[1].use_count)
        )

        # 保留后 max_rules 条
        self._rules = dict(sorted_rules[-self.max_rules :])

    def get_rules(self) -> list[DistilledRule]:
        """获取所有规则"""
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> DistilledRule | None:
        """获取规则"""
        return self._rules.get(rule_id)

    def update_rule_usage(self, rule_id: str, success: bool):
        """更新规则使用情况"""
        rule = self._rules.get(rule_id)
        if rule:
            rule.use_count += 1
            if success:
                rule.success_count += 1
            self._save_rules()

    def get_stats(self) -> dict:
        """获取统计信息"""
        if not self._rules:
            return {
                "total_rules": 0,
                "avg_confidence": 0.0,
            }

        return {
            "total_rules": len(self._rules),
            "avg_confidence": sum(r.confidence for r in self._rules.values())
            / len(self._rules),
            "total_uses": sum(r.use_count for r in self._rules.values()),
            "total_successes": sum(r.success_count for r in self._rules.values()),
        }

    def clear_rules(self):
        """清空规则"""
        self._rules.clear()
        self._save_rules()
