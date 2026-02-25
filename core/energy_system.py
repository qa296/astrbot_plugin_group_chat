"""
能量系统

管理机器人的"能量"概念，用于动态调节回复频率和意愿

能量机制：
- 回复消耗能量（基础消耗 + 长度消耗）
- 群活跃时缓慢恢复
- 被@时快速恢复
- 正面反馈恢复
- 能量低于阈值时降低回复意愿
"""

import time
from dataclasses import dataclass

from astrbot.api import logger


@dataclass
class EnergyConfig:
    """能量配置"""

    initial_energy: float = 0.8
    min_energy: float = 0.1
    max_energy: float = 1.0

    energy_cost_base: float = 0.1
    energy_cost_per_char: float = 0.0005
    energy_cost_streak_multiplier: float = 0.05

    energy_recovery_rate: float = 0.02
    energy_recovery_on_at: float = 0.3
    energy_recovery_on_positive_feedback: float = 0.1

    active_recovery_bonus: float = 0.01
    flow_recovery_bonus: float = 0.005
    observer_recovery_penalty: float = 0.5


class EnergySystem:
    """
    能量系统

    管理每个群组的能量状态，与心流状态机协同工作
    """

    def __init__(self, config, persistence):
        self.config = config
        self.persistence = persistence

        # 能量配置
        energy_config = config.get("energy_system", {})
        self.cfg = EnergyConfig(
            initial_energy=energy_config.get("initial_energy", 0.8),
            min_energy=0.1,
            max_energy=1.0,
            energy_cost_base=energy_config.get("energy_cost_base", 0.1),
            energy_cost_per_char=energy_config.get("energy_cost_per_char", 0.0005),
            energy_cost_streak_multiplier=0.05,
            energy_recovery_rate=energy_config.get("energy_recovery_rate", 0.02),
            energy_recovery_on_at=energy_config.get("energy_recovery_on_at", 0.3),
            energy_recovery_on_positive_feedback=0.1,
            active_recovery_bonus=0.01,
            flow_recovery_bonus=0.005,
            observer_recovery_penalty=0.5,
        )

        # 群组能量缓存
        self._energies: dict[str, float] = {}
        self._last_update: dict[str, float] = {}



    def get_energy(self, group_id: str) -> float:
        """获取群组当前能量"""
        if group_id not in self._energies:
            self._energies[group_id] = self.cfg.initial_energy
            self._last_update[group_id] = time.time()
        return self._energies[group_id]

    def consume(self, group_id: str, reply_length: int, streak: int = 0) -> float:
        """
        消耗能量

        Args:
            group_id: 群组ID
            reply_length: 回复长度
            streak: 连续回复次数

        Returns:
            消耗后的能量值
        """
        current = self.get_energy(group_id)

        # 基础消耗
        cost = self.cfg.energy_cost_base

        # 长度消耗
        cost += reply_length * self.cfg.energy_cost_per_char

        # 连续回复惩罚
        if streak > 0:
            cost += streak * self.cfg.energy_cost_streak_multiplier

        new_energy = max(self.cfg.min_energy, current - cost)
        self._energies[group_id] = new_energy
        self._last_update[group_id] = time.time()

        logger.debug(
            f"群组 {group_id} 能量消耗: {current:.2f} -> {new_energy:.2f} "
            f"(cost={cost:.3f}, length={reply_length}, streak={streak})"
        )

        return new_energy

    def recover(self, group_id: str, amount: float | None = None) -> float:
        """
        恢复能量

        Args:
            group_id: 群组ID
            amount: 恢复量，不指定则使用默认恢复率

        Returns:
            恢复后的能量值
        """
        current = self.get_energy(group_id)

        if amount is None:
            amount = self.cfg.energy_recovery_rate

        new_energy = min(self.cfg.max_energy, current + amount)
        self._energies[group_id] = new_energy

        return new_energy

    def recover_on_at(self, group_id: str) -> float:
        """
        被@时的能量恢复

        Args:
            group_id: 群组ID

        Returns:
            恢复后的能量值
        """
        return self.recover(group_id, self.cfg.energy_recovery_on_at)

    def recover_on_positive_feedback(self, group_id: str) -> float:
        """
        正面反馈时的能量恢复

        Args:
            group_id: 群组ID

        Returns:
            恢复后的能量值
        """
        return self.recover(group_id, self.cfg.energy_recovery_on_positive_feedback)

    def time_based_recovery(
        self, group_id: str, flow_state: str, group_activity: float = 0.5
    ) -> float:
        """
        基于时间的能量恢复

        Args:
            group_id: 群组ID
            flow_state: 心流状态
            group_activity: 群活跃度

        Returns:
            恢复后的能量值
        """
        current_time = time.time()
        last_update = self._last_update.get(group_id, current_time)
        elapsed_minutes = (current_time - last_update) / 60.0

        if elapsed_minutes < 0.1:
            return self.get_energy(group_id)

        # 基础恢复量
        base_recovery = self.cfg.energy_recovery_rate * elapsed_minutes

        # 根据心流状态调整
        if flow_state == "active":
            bonus = self.cfg.active_recovery_bonus * elapsed_minutes
            base_recovery += bonus
        elif flow_state == "flow":
            bonus = self.cfg.flow_recovery_bonus * elapsed_minutes
            base_recovery += bonus
        elif flow_state == "observer":
            base_recovery *= self.cfg.observer_recovery_penalty

        # 根据群活跃度加成
        activity_bonus = base_recovery * group_activity * 0.5
        total_recovery = base_recovery + activity_bonus

        new_energy = self.recover(group_id, total_recovery)
        self._last_update[group_id] = current_time

        return new_energy

    def penalty_on_negative_feedback(self, group_id: str) -> float:
        """
        负面反馈时的能量惩罚

        Args:
            group_id: 群组ID

        Returns:
            惩罚后的能量值
        """
        current = self.get_energy(group_id)
        penalty = 0.15
        new_energy = max(self.cfg.min_energy, current - penalty)
        self._energies[group_id] = new_energy

        logger.debug(
            f"群组 {group_id} 负面反馈能量惩罚: {current:.2f} -> {new_energy:.2f}"
        )

        return new_energy

    def can_reply(self, group_id: str, threshold: float = 0.3) -> bool:
        """
        检查是否有足够能量回复

        Args:
            group_id: 群组ID
            threshold: 能量阈值

        Returns:
            是否可以回复
        """
        return self.get_energy(group_id) >= threshold

    def get_reply_willingness_modifier(self, group_id: str) -> float:
        """
        获取回复意愿修正系数

        Args:
            group_id: 群组ID

        Returns:
            意愿修正系数 (0.5 - 1.5)
        """
        energy = self.get_energy(group_id)

        if energy >= 0.8:
            return 1.2
        elif energy >= 0.6:
            return 1.0
        elif energy >= 0.4:
            return 0.8
        elif energy >= 0.2:
            return 0.6
        else:
            return 0.5

    def set_energy(self, group_id: str, value: float):
        """直接设置能量值（用于调试/管理）"""
        self._energies[group_id] = max(
            self.cfg.min_energy, min(self.cfg.max_energy, value)
        )
        self._last_update[group_id] = time.time()

    def reset_energy(self, group_id: str):
        """重置能量到初始值"""
        self._energies[group_id] = self.cfg.initial_energy
        self._last_update[group_id] = time.time()

    def get_all_energies(self) -> dict[str, float]:
        """获取所有群组的能量状态"""
        return dict(self._energies)
