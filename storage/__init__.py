"""
存储模块 - 持久化管理

负责插件数据的持久化存储，数据存储在 data/plugins/astrbot_plugin_group_chat/ 目录下
"""

from .persistence import PersistenceManager

__all__ = ["PersistenceManager"]
