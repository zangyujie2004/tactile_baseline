# Action 拼接触觉数据修改说明

## 修改目标
将 15 维的触觉嵌入数据 (`left_gripper1_marker_offset_emb`) 拼接到 10 维的 action 上，形成 25 维的完整 action，然后在 DP 中一起进行量化和加噪声。

## 修改内容

### 1. 配置文件修改
**文件**: `reactive_diffusion_policy/config/task/kinedex.yaml`

**修改前**:
```yaml
action:
  shape: [10] # (3+6+1) (x, y, z, 6d rotation, gripper_width)
  left_gripper1_marker_offset_emb:
    shape: [ 15 ]
    type: low_dim
```

**修改后**:
```yaml
action:
  shape: [25] # (3+6+1+15) (x, y, z, 6d rotation, gripper_width, tactile_emb_15d)
```

**说明**: 
- 将 action shape 从 10 改为 25
- 移除了 action 下的 `left_gripper1_marker_offset_emb` 子键（因为现在触觉数据直接包含在 action 的 25 维中）
- 触觉数据仍然保留在 obs 中用于读取

### 2. Dataset 初始化修改
**文件**: `reactive_diffusion_policy/dataset/kinedex_dataset.py`

**位置**: `__init__` 方法中计算 dataset-level action min/max 的部分

**功能**: 
- 检测是否需要拼接触觉数据（通过对比配置中的 action shape 和 replay_buffer 中的实际 action shape）
- 如果需要，从 replay_buffer 读取触觉数据并拼接到 action
- 计算完整 25 维 action 的 min/max 用于量化

**关键代码**:
```python
# Get base action (10 dims)
base_action_vals = replay_buffer['action'][:].astype(np.float32)

# Check if we need to append tactile embeddings to action
tactile_keys = [k for k in shape_meta.get('obs', {}).keys() if 'marker_offset_emb' in k]
if tactile_keys and action_shape_cfg > base_action_vals.shape[1]:
    tactile_key = tactile_keys[0]
    tactile_vals = replay_buffer[tactile_key][:].astype(np.float32)
    full_action_vals = np.concatenate([base_action_vals, tactile_vals], axis=-1)
```

### 3. Dataset __getitem__ 修改
**文件**: `reactive_diffusion_policy/dataset/kinedex_dataset.py`

**位置**: `__getitem__` 方法中读取 action 之后

**功能**:
- 从 replay_buffer 读取基础 10 维 action
- 检测是否需要拼接触觉数据
- 从 data 中读取触觉嵌入并拼接到 action
- 处理时间延迟（确保触觉数据也应用相同的 latency 处理）

**关键代码**:
```python
# Concatenate tactile embedding to action if specified in shape_meta
action_target_dim = self.shape_meta['action']['shape'][0]
if action.shape[-1] < action_target_dim:
    tactile_keys = [k for k in self.shape_meta.get('obs', {}).keys() if 'marker_offset_emb' in k]
    if tactile_keys:
        tactile_key = tactile_keys[0]
        tactile_data = data[tactile_key][:].astype(np.float32)
        if self.n_latency_steps > 0:
            tactile_data = tactile_data[self.n_latency_steps:]
        action = np.concatenate([action, tactile_data], axis=-1)
```

### 4. Normalizer 修改
**文件**: `reactive_diffusion_policy/dataset/kinedex_dataset.py`

**位置**: `get_normalizer` 方法中处理 action 的部分

**功能**:
- 为 normalizer 计算统计信息时，也需要拼接触觉数据
- 确保 normalizer 能正确处理 25 维 action

**关键代码**:
```python
# Get base action from replay buffer
base_action = self.replay_buffer['action'][:]

# Check if we need to append tactile embeddings to action
action_target_dim = self.shape_meta['action']['shape'][0]
if base_action.shape[-1] < action_target_dim:
    tactile_keys = [k for k in self.shape_meta.get('obs', {}).keys() if 'marker_offset_emb' in k]
    if tactile_keys:
        tactile_key = tactile_keys[0]
        tactile_data = self.replay_buffer[tactile_key][:]
        action_all = np.concatenate([base_action, tactile_data], axis=-1)
```

## 数据流程

1. **训练时**:
   ```
   replay_buffer['action'] (10 dims) + replay_buffer['left_gripper1_marker_offset_emb'] (15 dims)
   → concatenate → 25 dims action
   → quantize (if enabled) → 25 dims quantized integers
   → dequantize → 25 dims floats
   → add noise (if enabled) → 25 dims noisy floats
   → normalize → pass to DP model
   ```

2. **推理时**:
   - DP 模型预测 25 维 action
   - 前 10 维用于机器人控制 (x, y, z, 6d rotation, gripper)
   - 后 15 维是触觉嵌入（可用于其他目的或忽略）

## 量化和噪声

由于之前已经实现了量化和噪声功能，现在这些功能会自动应用到完整的 25 维 action 上：

- **量化**: 15-bit 量化会应用到所有 25 个维度
- **噪声**: 高斯噪声会加到所有 25 个维度

## 配置启用方式

在训练配置中添加：
```yaml
task:
  dataset:
    action_quant_bits: 15        # 启用 15-bit 量化
    action_noise_std: 0.01       # 噪声标准差
```

或通过命令行：
```bash
python train.py task=kinedex \
    task.dataset.action_quant_bits=15 \
    task.dataset.action_noise_std=0.01
```

## 测试

运行测试脚本验证修改：
```bash
python test_action_concat.py
```

预期输出应显示：
- Action 维度为 25
- Normalizer 处理 25 维数据
- 量化后的 action_quantized 也是 25 维

## 注意事项

1. **向后兼容**: 代码会自动检测配置中的 action shape，只有当 shape > 实际 replay_buffer action 维度时才会拼接触觉数据
2. **时间对齐**: 触觉数据和 action 使用相同的 latency 处理，确保时间对齐
3. **Normalizer**: action normalizer 现在会对所有 25 维进行归一化
4. **量化范围**: 每个维度使用各自的 min/max 进行量化，不同维度的量化范围可能不同

## 相关文件

- `reactive_diffusion_policy/config/task/kinedex.yaml` - 配置文件
- `reactive_diffusion_policy/dataset/kinedex_dataset.py` - 数据集实现
- `test_action_concat.py` - 测试脚本
