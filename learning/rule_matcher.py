"""
规则匹配器

运行时匹配：
1. 相似度匹配：TF-IDF向量 + 余弦相似度
2. 正则匹配：即时消息模式匹配

核心流程：
1. 加载规则（启动时）
   - 加载全局词表
   - 加载相似度规则（原始文本）
   - 加载正则规则
   - 生成所有向量
2. 相似度匹配
   - 获取最近N条消息合并
   - 构建TF-IDF向量
   - 计算与规则向量的余弦相似度
3. 正则匹配
   - 当前消息匹配正则
   - 计数触发
"""

import math
import re
from collections import Counter
from typing import Any

from astrbot.api import logger

try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    logger.warning("jieba未安装，将使用简单分词")


class RuleMatcher:
    """
    规则匹配器

    支持相似度匹配和正则匹配双重触发
    """

    def __init__(self, persistence, config):
        self.persistence = persistence
        self.config = config

        # 配置
        distill_config = config.get("offline_distillation", {})
        sim_config = distill_config.get("similarity", {})
        self.context_window = sim_config.get("context_window", 10)
        self.default_threshold = sim_config.get("threshold", 0.6)

        # 缓存向量（运行时生成）
        self._vectors_cache: dict[str, list[float]] = {}
        self._vocabulary: list[str] = []
        self._idf_values: dict[str, float] = {}
        self._doc_count: int = 0

        # 统计信息
        self._stats = {
            "similarity_matches": 0,
            "regex_matches": 0,
            "total_checks": 0,
        }

        # 初始化加载
        self._load_vocabulary()

        logger.info(
            f"规则匹配器初始化: context_window={self.context_window}, "
            f"threshold={self.default_threshold}, vocab_size={len(self._vocabulary)}"
        )

    def _load_vocabulary(self):
        """加载全局词表"""
        vocab = self.persistence.get_global_vocabulary()
        self._vocabulary = vocab.vocabulary
        self._idf_values = vocab.idf_values
        self._doc_count = vocab.doc_count

    def build_vector(self, text: str) -> list[float]:
        """
        构建TF-IDF向量

        Args:
            text: 输入文本

        Returns:
            TF-IDF向量
        """
        # 分词
        words = self._tokenize(text)

        if not words or not self._vocabulary:
            return [0.0] * len(self._vocabulary)

        # 计算词频
        word_count = Counter(words)
        total = len(words)

        # 构建向量
        vector = []
        for word in self._vocabulary:
            tf = word_count.get(word, 0) / total if total > 0 else 0
            idf = self._idf_values.get(word, 0)
            vector.append(tf * idf)

        return vector

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """
        计算余弦相似度

        Args:
            v1: 向量1
            v2: 向量2

        Returns:
            相似度 (0-1)
        """
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        if HAS_JIEBA:
            return list(jieba.cut(text))
        else:
            # 简单正则分词
            return re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z]+|\d+", text.lower())

    def _get_rule_vector(self, rule_id: str, original_text: str) -> list[float]:
        """获取规则向量（带缓存）"""
        if rule_id in self._vectors_cache:
            return self._vectors_cache[rule_id]

        # 生成向量
        vector = self.build_vector(original_text)
        self._vectors_cache[rule_id] = vector
        return vector

    def match_similarity(
        self, current_context: str, threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """
        相似度匹配

        Args:
            current_context: 当前对话上下文文本
            threshold: 相似度阈值，None则使用默认值

        Returns:
            匹配结果列表
        """
        if threshold is None:
            threshold = self.default_threshold

        self._stats["total_checks"] += 1

        if not self._vocabulary:
            return []

        # 构建当前上下文向量
        current_vector = self.build_vector(current_context)

        # 检查是否为零向量
        if all(v == 0 for v in current_vector):
            return []

        # 获取所有相似度规则
        rules = self.persistence.get_all_similarity_rules()
        matches = []

        for rule in rules:
            # 获取规则向量
            rule_vector = self._get_rule_vector(rule.id, rule.original_text)

            # 计算相似度
            similarity = self.cosine_similarity(current_vector, rule_vector)

            if similarity >= threshold:
                matches.append(
                    {
                        "rule_id": rule.id,
                        "similarity": similarity,
                        "threshold": rule.threshold,
                        "source_group": rule.source_group,
                        "source_seq": rule.source_seq,
                    }
                )

        if matches:
            self._stats["similarity_matches"] += 1
            # 按相似度排序
            matches.sort(key=lambda x: x["similarity"], reverse=True)

        return matches

    def match_regex(self, current_message: str) -> list[dict[str, Any]]:
        """
        正则匹配

        Args:
            current_message: 当前消息内容

        Returns:
            匹配结果列表
        """
        self._stats["total_checks"] += 1

        rules = self.persistence.get_all_regex_rules()
        matches = []

        for rule in rules:
            try:
                if re.search(rule.pattern, current_message):
                    # 更新计数
                    triggered = self.persistence.update_regex_rule_count(rule.id, 1)

                    if triggered:
                        matches.append(
                            {
                                "rule_id": rule.id,
                                "pattern": rule.pattern,
                                "trigger_count": rule.trigger_count,
                            }
                        )
                        self._stats["regex_matches"] += 1

            except re.error as e:
                logger.warning(f"正则匹配失败: {rule.pattern}, 错误: {e}")
                continue

        return matches

    def check_match(
        self,
        current_context: str,
        current_message: str,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        综合匹配检查

        Args:
            current_context: 当前对话上下文
            current_message: 当前消息
            similarity_threshold: 相似度阈值

        Returns:
            匹配结果
        """
        result = {
            "matched": False,
            "similarity_matches": [],
            "regex_matches": [],
            "best_match": None,
        }

        # 相似度匹配
        sim_matches = self.match_similarity(current_context, similarity_threshold)
        result["similarity_matches"] = sim_matches

        # 正则匹配
        regex_matches = self.match_regex(current_message)
        result["regex_matches"] = regex_matches

        # 判断是否匹配
        if sim_matches or regex_matches:
            result["matched"] = True

            # 选择最佳匹配
            if sim_matches:
                result["best_match"] = {
                    "type": "similarity",
                    **sim_matches[0],
                }
            elif regex_matches:
                result["best_match"] = {
                    "type": "regex",
                    **regex_matches[0],
                }

        return result

    def refresh_vectors(self):
        """刷新向量缓存（词表更新后调用）"""
        self._load_vocabulary()
        self._vectors_cache.clear()
        logger.info("向量缓存已刷新")

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        total = self._stats["total_checks"]
        return {
            **self._stats,
            "similarity_match_rate": (
                self._stats["similarity_matches"] / total if total > 0 else 0
            ),
            "regex_match_rate": (
                self._stats["regex_matches"] / total if total > 0 else 0
            ),
            "vocabulary_size": len(self._vocabulary),
            "doc_count": self._doc_count,
            "cached_vectors": len(self._vectors_cache),
        }

    def update_match_result(self, rule_id: str, success: bool, rule_type: str):
        """
        更新匹配结果

        Args:
            rule_id: 规则ID
            success: 是否成功（用户正向反馈）
            rule_type: 规则类型 (similarity/regex)
        """
        if rule_type == "similarity":
            self.persistence.update_similarity_rule_usage(rule_id, success)
        elif rule_type == "regex":
            # 正则规则的计数已经在匹配时更新
            pass
