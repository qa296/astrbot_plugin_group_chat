"""
离线知识蒸馏

基于真实群聊数据的离线蒸馏方案：
1. 收集群聊历史消息
2. LLM分析真实对话，标注插话时机
3. 生成相似度规则和正则规则
4. 更新全局词表

核心流程：
- 每天凌晨3点触发
- 获取未处理消息（最多1000条）
- LLM分析 → insert_after序号 + regex_rules
- 取序号前10条消息作为原始文本
- 更新词表、存储规则
"""

import json
import math
import re
import time
from collections import Counter
from typing import Any

from astrbot.api import logger

try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    logger.warning("jieba未安装，将使用简单分词")

from ..storage.persistence import (
    GlobalVocabulary,
    RegexRule,
    SimilarityRule,
)


class OfflineDistiller:
    """
    离线知识蒸馏 v2.0

    基于真实群聊数据，通过LLM分析生成插话规则
    """

    def __init__(self, context, config, persistence=None):
        self.context = context
        self.config = config
        self.persistence = persistence

        # 配置
        distill_config = config.get("offline_distillation", {})
        self.enabled = distill_config.get("enabled", True)
        self.max_messages = distill_config.get("max_messages", 1000)
        self.context_window = distill_config.get("similarity", {}).get(
            "context_window", 10
        )
        self.similarity_threshold = distill_config.get("similarity", {}).get(
            "threshold", 0.6
        )
        self.max_similarity_rules = distill_config.get("rules", {}).get(
            "max_similarity_rules", 500
        )
        self.max_regex_rules = distill_config.get("rules", {}).get(
            "max_regex_rules", 100
        )

        # 统计信息
        self._stats = {
            "total_distillations": 0,
            "total_messages_processed": 0,
            "total_rules_generated": 0,
            "last_distill_time": 0,
        }

        logger.info(
            f"离线蒸馏器初始化: enabled={self.enabled}, "
            f"max_messages={self.max_messages}, threshold={self.similarity_threshold}"
        )

    async def distill(self) -> dict[str, int]:
        """
        执行蒸馏

        Returns:
            统计信息字典
        """
        if not self.enabled:
            logger.info("离线蒸馏已禁用")
            return {"rules_generated": 0}

        if not self.persistence:
            logger.warning("无持久化存储，跳过蒸馏")
            return {"rules_generated": 0}

        logger.info("开始离线蒸馏...")
        start_time = time.time()
        self._stats["total_distillations"] += 1

        results = {
            "groups_processed": 0,
            "messages_processed": 0,
            "similarity_rules_generated": 0,
            "regex_rules_generated": 0,
        }

        # 获取所有活跃群组
        active_groups = self.persistence.get_active_groups()

        for group_id in active_groups:
            try:
                group_result = await self._distill_group(group_id)
                results["groups_processed"] += 1
                results["messages_processed"] += group_result.get("messages_processed", 0)
                results["similarity_rules_generated"] += group_result.get(
                    "similarity_rules", 0
                )
                results["regex_rules_generated"] += group_result.get("regex_rules", 0)
            except Exception as e:
                logger.error(f"群组 {group_id} 蒸馏失败: {e}")

        # 更新全局词表
        await self._update_global_vocabulary()

        # 清理超出限制的规则
        self._prune_rules()

        # 保存
        self.persistence.save_dirty()

        elapsed = time.time() - start_time
        self._stats["total_messages_processed"] += results["messages_processed"]
        self._stats["total_rules_generated"] += (
            results["similarity_rules_generated"] + results["regex_rules_generated"]
        )
        self._stats["last_distill_time"] = time.time()

        logger.info(
            f"离线蒸馏完成: {results['groups_processed']} 个群组, "
            f"{results['messages_processed']} 条消息, "
            f"{results['similarity_rules_generated']} 条相似度规则, "
            f"{results['regex_rules_generated']} 条正则规则, "
            f"耗时 {elapsed:.2f}s"
        )

        return results

    async def _distill_group(self, group_id: str) -> dict[str, int]:
        """蒸馏单个群组"""
        result = {
            "messages_processed": 0,
            "similarity_rules": 0,
            "regex_rules": 0,
        }

        # 获取未处理消息
        unprocessed = self.persistence.get_unprocessed_messages(group_id, self.max_messages)

        if not unprocessed:
            logger.debug(f"群组 {group_id} 无未处理消息")
            return result

        result["messages_processed"] = len(unprocessed)

        # 构建消息列表文本
        messages_text = self._format_messages_for_llm(unprocessed)

        # 调用LLM分析
        llm_result = await self._analyze_with_llm(messages_text)

        if not llm_result:
            logger.warning(f"群组 {group_id} LLM分析失败")
            return result

        # 处理相似度规则
        insert_after_list = llm_result.get("insert_after", [])
        for seq in insert_after_list:
            rule = self._create_similarity_rule(group_id, seq, unprocessed)
            if rule:
                self.persistence.add_similarity_rule(rule)
                result["similarity_rules"] += 1

        # 处理正则规则
        regex_rules = llm_result.get("regex_rules", [])
        for regex_data in regex_rules:
            rule = self._create_regex_rule(regex_data)
            if rule:
                self.persistence.add_regex_rule(rule)
                result["regex_rules"] += 1

        # 标记消息为已处理
        processed_seqs = [m.get("seq") for m in unprocessed]
        self.persistence.mark_messages_processed(group_id, processed_seqs)

        return result

    def _format_messages_for_llm(self, messages: list[dict]) -> str:
        """格式化消息列表供LLM分析"""
        lines = []
        for msg in messages:
            seq = msg.get("seq", 0)
            user_name = msg.get("user_name", "未知")
            content = msg.get("content", "")
            lines.append(f"[{seq}] {user_name}: {content}")
        return "\n".join(lines)

    async def _analyze_with_llm(self, messages_text: str) -> dict | None:
        """调用LLM分析消息"""
        try:
            provider = self.context.get_using_provider()
            if not provider:
                logger.warning("无可用LLM提供商")
                return None

            prompt = self._build_prompt(messages_text)
            system_prompt = self._get_system_prompt()

            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                image_urls=[],
                system_prompt=system_prompt,
            )

            if response and response.completion_text:
                return self._parse_llm_response(response.completion_text)

        except Exception as e:
            logger.error(f"LLM分析失败: {e}")

        return None

    def _build_prompt(self, messages_text: str) -> str:
        """构建LLM提示"""
        return f"""你是群聊对话分析专家。以下是群聊消息记录：

【消息列表】
{messages_text}

请分析：
1. 哪些位置适合机器人插话？输出消息序号
2. 有哪些对话模式可以用正则表达？

输出JSON格式：
{{
    "insert_after": [3, 5],
    "regex_rules": [
        {{"pattern": "有人.{0,5}[吗？?]", "trigger_count": 1}},
        {{"pattern": "(大家|有没有).{0,10}知道", "trigger_count": 1}}
    ]
}}

注意：
- 只输出真正适合插话的位置（用户提问、寻求帮助、话题中断等）
- 正则要能匹配相似的对话场景，不要太具体
- trigger_count表示需要匹配几次才触发（通常为1）
- 输出纯JSON，不要有其他内容"""

    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个群聊策略专家，负责分析群聊记录并识别适合机器人插话的时机。
目标：
1. 找出用户提问、寻求帮助、话题讨论停顿等适合插话的位置
2. 提取可复用的对话模式作为正则规则
原则：
- 不要过度插话，只在真正合适的位置
- 正则规则要具有一定的泛化能力"""

    def _parse_llm_response(self, text: str) -> dict | None:
        """解析LLM响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")

        return None

    def _create_similarity_rule(
        self, group_id: str, seq: int, messages: list[dict]
    ) -> SimilarityRule | None:
        """创建相似度规则"""
        # 获取seq之前的context_window条消息
        before_messages = [m for m in messages if m.get("seq", 0) < seq]
        context_messages = before_messages[-self.context_window :]

        if not context_messages:
            return None

        # 合并文本
        original_text = "/".join(
            m.get("content", "") for m in context_messages
        )

        if len(original_text) < 10:  # 太短的不处理
            return None

        rule_id = f"sim_{group_id}_{seq}_{int(time.time())}"

        return SimilarityRule(
            id=rule_id,
            original_text=original_text,
            threshold=self.similarity_threshold,
            source_group=group_id,
            source_seq=seq,
        )

    def _create_regex_rule(self, regex_data: dict) -> RegexRule | None:
        """创建正则规则"""
        pattern = regex_data.get("pattern", "")
        trigger_count = regex_data.get("trigger_count", 1)

        if not pattern:
            return None

        # 验证正则表达式是否有效
        try:
            re.compile(pattern)
        except re.error as e:
            logger.warning(f"无效的正则表达式: {pattern}, 错误: {e}")
            return None

        rule_id = f"regex_{int(time.time())}_{hash(pattern) % 10000:04d}"

        return RegexRule(
            id=rule_id,
            pattern=pattern,
            trigger_count=trigger_count,
        )

    async def _update_global_vocabulary(self):
        """更新全局词表"""
        # 获取所有相似度规则的原始文本
        rules = self.persistence.get_all_similarity_rules()
        if not rules:
            return

        # 收集所有文档
        documents = [rule.original_text for rule in rules]

        # 提取所有词
        all_words = []
        doc_word_sets = []

        for doc in documents:
            words = self._tokenize(doc)
            all_words.extend(words)
            doc_word_sets.append(set(words))

        # 计算词频
        word_freq = Counter(all_words)

        # 计算IDF值
        doc_count = len(documents)
        idf_values = {}

        for word, freq in word_freq.items():
            # 计算包含该词的文档数
            doc_freq = sum(1 for doc_set in doc_word_sets if word in doc_set)
            # IDF = log(总文档数 / 包含该词的文档数)
            if doc_freq > 0:
                idf_values[word] = math.log(doc_count / doc_freq)
            else:
                idf_values[word] = 0

        # 构建词表
        vocabulary = list(word_freq.keys())

        # 更新持久化
        self.persistence.update_global_vocabulary(vocabulary, idf_values, doc_count)

        logger.info(
            f"全局词表已更新: {len(vocabulary)} 个词, {doc_count} 个文档"
        )

    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        if HAS_JIEBA:
            return list(jieba.cut(text))
        else:
            # 简单正则分词
            return re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z]+|\d+", text.lower())

    def _prune_rules(self):
        """清理超出限制的规则"""
        # 清理相似度规则
        sim_rules = self.persistence.get_all_similarity_rules()
        if len(sim_rules) > self.max_similarity_rules:
            # 按成功率排序，保留最好的
            sorted_rules = sorted(
                sim_rules,
                key=lambda r: (
                    r.success_count / max(1, r.use_count),
                    r.use_count,
                ),
                reverse=True,
            )
            # 清空后重新添加
            self.persistence.clear_similarity_rules()
            for rule in sorted_rules[: self.max_similarity_rules]:
                self.persistence.add_similarity_rule(rule)
            logger.info(
                f"相似度规则已修剪: {len(sim_rules)} -> {self.max_similarity_rules}"
            )

        # 清理正则规则
        regex_rules = self.persistence.get_all_regex_rules()
        if len(regex_rules) > self.max_regex_rules:
            sorted_rules = sorted(
                regex_rules,
                key=lambda r: (
                    r.success_count / max(1, r.use_count),
                    r.use_count,
                ),
                reverse=True,
            )
            self.persistence.clear_regex_rules()
            for rule in sorted_rules[: self.max_regex_rules]:
                self.persistence.add_regex_rule(rule)
            logger.info(f"正则规则已修剪: {len(regex_rules)} -> {self.max_regex_rules}")

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "enabled": self.enabled,
            "max_messages": self.max_messages,
            "similarity_threshold": self.similarity_threshold,
        }

    def get_rules_summary(self) -> dict:
        """获取规则摘要"""
        if not self.persistence:
            return {}

        sim_rules = self.persistence.get_all_similarity_rules()
        regex_rules = self.persistence.get_all_regex_rules()
        vocab = self.persistence.get_global_vocabulary()

        return {
            "similarity_rules_count": len(sim_rules),
            "regex_rules_count": len(regex_rules),
            "vocabulary_size": len(vocab.vocabulary),
            "doc_count": vocab.doc_count,
        }
