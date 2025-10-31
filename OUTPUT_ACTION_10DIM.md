# 输出 Action 维度修改说明

## 问题
虽然 DP 模型训练时使用 25 维 action（10 维机器人控制 + 15 维触觉嵌入），但在实际输出和评估时，我们只需要前 10 维用于机器人控制。

## 解决方案

### 1. Policy 输出修改
**文件**: `reactive_diffusion_policy/policy/kinedex_image_policy.py`

**修改内容**:
- 在 `__init__` 中添加 `self.output_action_dim = 10`（当 action_dim=25 时）
- 在 `predict_action` 中，对完整的 25 维预测进行 unnormalize 后，只提取前 10 维作为输出
- 返回结果中：
  - `action`: 10 维机器人控制 action
  - `action_pred`: 10 维完整预测
  - `action_pred_full`: 25 维完整预测（用于调试）

**代码片段**:
```python
# Extract robot action and tactile embedding separately
action_pred_robot = action_pred[..., :self.output_action_dim]

# Extract tactile embedding if action dimension includes it
action_pred_tactile = None
if Da > self.output_action_dim:
    action_pred_tactile = action_pred[..., self.output_action_dim:]

result = {
    'action': action_pred_robot[:,start:end],  # 10-dim robot control
    'action_pred': action_pred_robot,  # 10-dim full trajectory
    'action_pred_full': action_pred,  # 25-dim full prediction
    'action_pred_tactile': action_pred_tactile,  # 15-dim tactile embedding
    'action_tactile': action_pred_tactile[:,start:end]  # 15-dim tactile (n_action_steps)
}
```

### 2. Training Workspace 评估修改
**文件**: `reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`

**修改内容**:
- 在计算 MSE 和 L1 loss 前，将 ground truth 的 25 维 action 截取为前 10 维
- 确保 pred_action（已经是 10 维）和 gt_action_robot（截取的 10 维）维度匹配

**代码片段**:
```python
# Extract only robot action dimensions for comparison
gt_action_robot = gt_action[..., :10]
all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action_robot))
```

## 数据流程

### 训练阶段
```
Dataset 输出: 25-dim action (10 robot + 15 tactile)
    ↓
Normalize: 25-dim normalized action
    ↓
DP Model 训练: 学习预测 25-dim action
    ↓
Loss 计算: 在 25-dim 上计算 loss
```

### 推理/评估阶段
```
DP Model 预测: 25-dim action
    ↓
Unnormalize: 25-dim denormalized action
    ↓
提取前 10 维: action_pred_robot = action_pred[..., :10]
    ↓
输出: 10-dim robot control action
```

### 评估指标计算
```
Ground Truth: 25-dim action → 截取前 10 维
Prediction: 已经是 10-dim
    ↓
计算 MSE/L1: 在 10-dim 上比较
```

## 优势

1. **训练时利用触觉信息**: 模型在训练时学习预测包含触觉的 25 维 action，可能帮助学习更好的表示
2. **推理时只输出控制量**: 实际部署时只使用前 10 维进行机器人控制
3. **灵活性**: 保留 `action_pred_full` 用于调试或分析触觉预测质量
4. **向后兼容**: 对于非 kinedex 任务（action_dim ≠ 25），仍然输出全部维度

## 注意事项

1. **维度硬编码**: 当前实现假设 action_dim=25 时，前 10 维是机器人控制，后 15 维是触觉嵌入。如果改变维度划分，需要修改代码
2. **评估一致性**: 确保所有评估脚本都使用前 10 维进行比较
3. **触觉预测输出**: 模型预测的触觉部分（后 15 维）现在可通过 `action_pred_tactile` 获取，可用于：
   - 分析触觉预测质量
   - 可视化触觉信息
   - 触觉相关的下游任务

## 输出字段详解

| 字段名 | 形状 | 说明 | 用途 |
|--------|------|------|------|
| `action` | `[B, n_action_steps, 10]` | 机器人控制 action | 实际执行 |
| `action_pred` | `[B, horizon, 10]` | 完整机器人轨迹 | 评估指标 |
| `action_pred_full` | `[B, horizon, 25]` | 完整预测 | 调试/分析 |
| `action_pred_tactile` | `[B, horizon, 15]` | 触觉嵌入轨迹 | 触觉分析 |
| `action_tactile` | `[B, n_action_steps, 15]` | 触觉嵌入（执行步数） | 触觉可视化 |

## 相关文件

- `reactive_diffusion_policy/policy/kinedex_image_policy.py` - Policy 输出修改
- `reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py` - 训练评估修改
- `ACTION_TACTILE_CONCAT_CHANGES.md` - 触觉数据拼接说明
