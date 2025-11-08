#!/usr/bin/env python3
"""
示例：如何使用 Policy 输出的分离字段

展示如何获取和使用：
1. 机器人控制 action (10 维)
2. 触觉嵌入预测 (15 维)
"""

import torch
import numpy as np

def example_policy_output_usage():
    """
    模拟 policy.predict_action() 的输出并展示如何使用
    """
    
    # 模拟 policy 输出
    batch_size = 4
    horizon = 16
    n_action_steps = 8
    
    result = {
        'action': torch.randn(batch_size, n_action_steps, 10),  # 机器人控制
        'action_pred': torch.randn(batch_size, horizon, 10),    # 完整机器人轨迹
        'action_pred_full': torch.randn(batch_size, horizon, 25),  # 完整 25 维
        'action_pred_tactile': torch.randn(batch_size, horizon, 15),  # 触觉轨迹
        'action_tactile': torch.randn(batch_size, n_action_steps, 15)  # 触觉（执行步数）
    }
    
    print("=" * 60)
    print("Policy 输出示例")
    print("=" * 60)
    
    # 1. 获取机器人控制 action（用于实际执行）
    robot_action = result['action']
    print(f"\n1. 机器人控制 action:")
    print(f"   形状: {robot_action.shape}")
    print(f"   用途: 发送给机器人执行")
    print(f"   维度: (x, y, z, 6d_rotation, gripper_width)")
    
    # 2. 获取触觉预测（用于分析）
    tactile_pred = result['action_pred_tactile']
    print(f"\n2. 触觉嵌入预测:")
    print(f"   形状: {tactile_pred.shape}")
    print(f"   用途: 分析模型对触觉的预测能力")
    print(f"   维度: PCA 降维后的 15 维触觉标记偏移")
    
    # 3. 获取执行步数的触觉
    tactile_action = result['action_tactile']
    print(f"\n3. 触觉嵌入（执行步数）:")
    print(f"   形状: {tactile_action.shape}")
    print(f"   用途: 对应机器人执行步数的触觉信息")
    
    # 4. 完整预测（调试用）
    full_pred = result['action_pred_full']
    print(f"\n4. 完整预测:")
    print(f"   形状: {full_pred.shape}")
    print(f"   用途: 包含机器人和触觉的完整预测")
    
    # 验证维度关系
    assert full_pred[..., :10].shape == result['action_pred'].shape
    assert full_pred[..., 10:].shape == result['action_pred_tactile'].shape
    print(f"\n✓ 维度验证通过:")
    print(f"  full_pred[:, :, :10] == action_pred")
    print(f"  full_pred[:, :, 10:] == action_pred_tactile")
    
    return result


def example_evaluation_usage(gt_action_25dim, pred_result):
    """
    示例：如何在评估中使用分离的输出
    
    Args:
        gt_action_25dim: Ground truth action (25 维)
        pred_result: Policy predict_action() 的输出
    """
    print("\n" + "=" * 60)
    print("评估示例")
    print("=" * 60)
    
    # 1. 评估机器人控制 action（前 10 维）
    gt_robot = gt_action_25dim[..., :10]
    pred_robot = pred_result['action_pred']
    
    robot_mse = torch.nn.functional.mse_loss(pred_robot, gt_robot)
    print(f"\n1. 机器人 action MSE: {robot_mse.item():.6f}")
    print(f"   GT 形状: {gt_robot.shape}")
    print(f"   Pred 形状: {pred_robot.shape}")
    
    # 2. 评估触觉预测（后 15 维）
    gt_tactile = gt_action_25dim[..., 10:]
    pred_tactile = pred_result['action_pred_tactile']
    
    tactile_mse = torch.nn.functional.mse_loss(pred_tactile, gt_tactile)
    print(f"\n2. 触觉嵌入 MSE: {tactile_mse.item():.6f}")
    print(f"   GT 形状: {gt_tactile.shape}")
    print(f"   Pred 形状: {pred_tactile.shape}")
    
    # 3. 完整 action 评估
    full_mse = torch.nn.functional.mse_loss(pred_result['action_pred_full'], gt_action_25dim)
    print(f"\n3. 完整 action MSE: {full_mse.item():.6f}")
    
    return {
        'robot_mse': robot_mse.item(),
        'tactile_mse': tactile_mse.item(),
        'full_mse': full_mse.item()
    }


def example_robot_control_usage(pred_result):
    """
    示例：如何在实际机器人控制中使用输出
    """
    print("\n" + "=" * 60)
    print("机器人控制示例")
    print("=" * 60)
    
    # 获取机器人控制 action
    robot_action = pred_result['action']  # [B, n_action_steps, 10]
    
    # 假设 batch_size = 1（单个预测）
    robot_action = robot_action[0]  # [n_action_steps, 10]
    
    print(f"\n机器人执行 {robot_action.shape[0]} 步动作:")
    for step_idx in range(min(3, robot_action.shape[0])):  # 只打印前3步
        action_step = robot_action[step_idx]
        print(f"\n  步骤 {step_idx + 1}:")
        print(f"    位置 (x, y, z): {action_step[:3].numpy()}")
        print(f"    旋转 (6d): {action_step[3:9].numpy()}")
        print(f"    夹爪宽度: {action_step[9].item():.4f}")
    
    # 可选：使用触觉信息进行可视化或监控
    tactile_info = pred_result['action_tactile'][0]  # [n_action_steps, 15]
    print(f"\n对应的触觉信息:")
    print(f"  形状: {tactile_info.shape}")
    print(f"  可用于: 可视化、异常检测、触觉反馈等")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# Policy 输出字段使用示例")
    print("#" * 60)
    
    # 1. 基本输出示例
    result = example_policy_output_usage()
    
    # 2. 评估示例
    batch_size, horizon = 4, 16
    gt_action_25dim = torch.randn(batch_size, horizon, 25)
    metrics = example_evaluation_usage(gt_action_25dim, result)
    
    # 3. 机器人控制示例
    example_robot_control_usage(result)
    
    print("\n" + "#" * 60)
    print("# 示例完成")
    print("#" * 60)
    print("\n总结:")
    print("  ✓ action: 用于机器人控制 (10 维)")
    print("  ✓ action_pred: 用于评估 (10 维)")
    print("  ✓ action_pred_tactile: 用于触觉分析 (15 维)")
    print("  ✓ action_tactile: 用于触觉可视化 (15 维)")
    print("  ✓ action_pred_full: 用于调试 (25 维)")
    print()
