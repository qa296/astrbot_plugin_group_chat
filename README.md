# AstrBot 群聊插件 v2.0

基于心流理论的群聊主动对话插件，实现拟人化的智能交互体验。

## 核心特性

### 四态心流状态机

```
观察态 ──活跃度↑──→ 沉浸态 ──相关性↑──→ 活跃态
    ↑                  │                  │
    │                  │                  ↓
    └──────────────────┴──────────────── 疲劳态
```

| 状态 | 行为 | 触发条件 |
|------|------|---------|
| **观察态** | 静默观察，仅@触发 | 默认状态/活跃度低 |
| **沉浸态** | 适度参与，跟随话题 | 群活跃度 > 0.4 |
| **活跃态** | 高意愿回复，短冷却 | 相关性 > 0.7 或被@ |
| **疲劳态** | 强制休息，恢复能量 | 连续回复 > 5 次 |

### 三路学习机制

1. **策略检索**：从历史成功对话中检索相似场景的最优策略
2. **离线蒸馏**：LLM自玩模拟生成策略规则（每日执行）
3. **在线学习**：基于用户反馈的Q-learning实时优化

### 能量系统

- 回复消耗能量（基础 + 长度惩罚 + 连续惩罚）
- 活跃时缓慢恢复
- 被@时快速恢复 (+0.3)
- 正面反馈恢复 (+0.1)

### 读空气功能

LLM根据上下文判断是否应该回复，避免强行介入对话。

## 安装

```bash
# 1. 克隆到插件目录
cd AstrBot/data/plugins
git clone https://github.com/qa296/astrbot_plugin_group_chat.git

# 2. 安装依赖
pip install jieba

# 3. 重启 AstrBot 并在 WebUI 中启用插件
```

## 配置

在 WebUI 的插件配置页面进行设置：

### 基础配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `list_mode` | blacklist | 名单模式：blacklist/whitelist |
| `groups` | [] | 群组名单 |
| `performance_mode` | balanced | 性能模式：lightweight/balanced/quality |

### 心流状态机

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `observer_to_flow_threshold` | 0.4 | 观察态→沉浸态阈值 |
| `flow_to_active_threshold` | 0.7 | 沉浸态→活跃态阈值 |
| `max_reply_streak` | 5 | 最大连续回复数 |
| `fatigue_recovery_minutes` | 5 | 疲劳恢复时间(分钟) |

### 能量系统

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `initial_energy` | 0.8 | 初始能量 |
| `energy_cost_base` | 0.1 | 基础回复消耗 |
| `energy_cost_per_char` | 0.0005 | 每字额外消耗 |
| `energy_recovery_rate` | 0.02 | 每分钟恢复量 |
| `energy_recovery_on_at` | 0.3 | 被@时恢复量 |

### 时机控制

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `base_cooldown_seconds` | 45.0 | 基础冷却时间 |
| `min_reply_delay` | 1.0 | 最小回复延迟 |
| `max_reply_delay` | 10.0 | 最大回复延迟 |
| `heartbeat_interval_seconds` | 15.0 | 心跳间隔 |

### 学习系统

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_online_learning` | true | 启用在线学习 |
| `enable_offline_distillation` | true | 启用离线蒸馏 |
| `distillation_interval_hours` | 24 | 蒸馏间隔 |
| `strategy_memory_size` | 1000 | 策略容量 |

### 奖励函数

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `user_reply` | 0.5 | 用户回复奖励 |
| `user_at_follow` | 1.0 | 用户@跟进奖励 |
| `user_like` | 0.3 | 用户点赞奖励 |
| `ignore_penalty` | -0.2 | 被忽略惩罚 |
| `negative_penalty` | -1.0 | 负面反馈惩罚 |

## 命令

### 查询状态

```
/心流状态
```

显示当前群的心流状态面板：

```
📊 心流状态面板
━━━━━━━━━━━━━━━
💠 心流状态: flow
⚡ 能量值: 0.72
🔄 连续回复: 2
━━━━━━━━━━━━━━━
📈 群活跃度: 0.65
💬 每分钟消息: 3.2
👥 活跃用户: 5
━━━━━━━━━━━━━━━
⏱️ 冷却剩余: 0.0s
📚 策略命中率: 45.2%
🎯 决策统计: 128
📨 反馈收集: 89
```

### 调试命令（管理员）

```
/心流调试 <action> [value]
```

| Action | 说明 | 示例 |
|--------|------|------|
| `state` | 设置状态 | `/心流调试 state active` |
| `energy` | 设置能量 | `/心流调试 energy 0.8` |
| `cooldown` | 重置冷却 | `/心流调试 cooldown` |
| `reset` | 完全重置 | `/心流调试 reset` |

## 架构

```
astrbot_plugin_group_chat/
├── main.py                    # 插件入口
├── _conf_schema.json          # 配置Schema
├── metadata.yaml              # 元数据
├── requirements.txt           # 依赖
├── core/                      # 核心层
│   ├── state_machine.py       # 心流状态机
│   ├── energy_system.py       # 能量系统
│   └── decision_engine.py     # 决策引擎
├── perception/                # 感知层
│   ├── context_analyzer.py    # 上下文分析
│   ├── activity_meter.py      # 活跃度计量
│   └── topic_tracker.py       # 话题追踪
├── learning/                  # 学习层
│   ├── strategy_store.py      # 策略库
│   ├── offline_distiller.py   # 离线蒸馏
│   └── online_learner.py      # 在线学习
├── execution/                 # 执行层
│   ├── response_generator.py  # 回复生成
│   ├── timing_controller.py   # 时机控制
│   └── feedback_collector.py  # 反馈收集
└── storage/                   # 存储层
    └── persistence.py         # 持久化
```

## 数据流

```
消息到达 → 权限检查 → 上下文分析 → 状态机处理 → 决策引擎
                                              ↓
                                         时机控制
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                                 生成回复              跳过
                                    ↓
                                 发送消息
                                    ↓
                              等待反馈 → 更新学习模型
```

## 数据存储

数据存储在 `data/plugins/astrbot_plugin_group_chat/` 目录下：

| 文件 | 内容 |
|------|------|
| `group_states.json` | 群组状态 |
| `strategies.json` | 策略记忆库 |
| `learning_data.json` | Q表和学习数据 |
| `group_umo.json` | 群组UMO映射 |
| `conversation_history.json` | 对话历史 |

## 理论基础

本插件基于心流理论设计，参考了以下研究成果：

- **心流理论**：Csikszentmihalyi 提出的挑战-技能平衡模型
- **集体心流**：群体层面的心流状态特征
- **主动对话系统**：预期性、主动性、规划性三要素
- **PRINCIPLES**：离线策略蒸馏方法

详见：`心流理论在人机交互对话系统中的应用研究文献综述.md`

## 常见问题

### Q: 机器人回复太频繁？

调低以下参数：
- `energy_recovery_rate`：降低能量恢复速度
- `base_cooldown_seconds`：增加冷却时间
- `max_reply_streak`：降低连续回复上限

### Q: 机器人不主动说话？

检查以下配置：
- `heartbeat_interval_seconds`：确保心跳间隔合理
- `observer_to_flow_threshold`：降低进入沉浸态的阈值
- 确保群组在白名单中（如果使用白名单模式）

### Q: 如何让机器人更聪明？

1. 开启在线学习和离线蒸馏
2. 增加策略库容量
3. 调整奖励函数参数

## 版本历史

### v2.0.0 (当前版本)

- 完全重构架构
- 实现四态心流状态机
- 三路学习机制
- 能量系统
- 读空气功能

### v1.0.x

- 基础的主动对话功能
- 意愿计算
- 读空气

## 贡献

欢迎提交 Issue 和 Pull Request！

---

**⭐ 如果这个插件对你有帮助，请给项目一个 Star！**
