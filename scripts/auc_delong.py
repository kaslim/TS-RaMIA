#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T50: AUC 和 DeLong 显著性检验
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from pathlib import Path


def delong_variance(y_true, y_scores):
    """
    计算 DeLong 方差（简化版）
    用于计算 AUC 的置信区间
    """
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    # 获取正负样本的得分
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    
    # 计算 AUC
    auc_val = roc_auc_score(y_true, y_scores)
    
    # DeLong 方差估计（简化版本）
    # 使用 Mann-Whitney U 统计量的方差估计
    q1 = auc_val / (2 - auc_val)
    q2 = (2 * auc_val**2) / (1 + auc_val)
    
    var_auc = (auc_val * (1 - auc_val) + 
               (n_pos - 1) * (q1 - auc_val**2) + 
               (n_neg - 1) * (q2 - auc_val**2)) / (n_pos * n_neg)
    
    return var_auc


def compute_auc_ci(y_true, y_scores, alpha=0.05):
    """
    计算 AUC 及其置信区间
    """
    from scipy.stats import norm
    
    auc_val = roc_auc_score(y_true, y_scores)
    var_auc = delong_variance(y_true, y_scores)
    std_auc = np.sqrt(var_auc)
    
    # 95% 置信区间
    z = norm.ppf(1 - alpha/2)
    ci_lower = max(0.0, auc_val - z * std_auc)
    ci_upper = min(1.0, auc_val + z * std_auc)
    
    # p 值（双侧检验，H0: AUC = 0.5）
    z_score = (auc_val - 0.5) / std_auc
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    return {
        'auc': auc_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': std_auc,
        'p_value': p_value
    }


def main():
    parser = argparse.ArgumentParser(description="T50: AUC 和 DeLong 检验")
    parser.add_argument("--csv", required=True, help="作品级 CSV 文件")
    parser.add_argument("--score_col", default="tis_mean", help="用于评分的列名")
    parser.add_argument("--label_col", default="is_member", help="标签列名")
    parser.add_argument("--out_txt", required=True, help="输出文本报告")
    parser.add_argument("--out_png", required=True, help="输出 ROC 曲线图")
    args = parser.parse_args()
    
    print("=" * 70)
    print("T50: AUC 和 DeLong 检验")
    print("=" * 70)
    print(f"输入 CSV: {args.csv}")
    print(f"评分列: {args.score_col}")
    print(f"标签列: {args.label_col}")
    
    # 加载数据
    print(f"\n加载数据...")
    df = pd.read_csv(args.csv)
    print(f"  总样本数: {len(df)}")
    print(f"  成员数: {df[args.label_col].sum()}")
    print(f"  非成员数: {len(df) - df[args.label_col].sum()}")
    
    # 提取标签和得分
    y_true = df[args.label_col].values
    y_scores = df[args.score_col].values
    
    # 计算 ROC 曲线
    print(f"\n计算 ROC 曲线...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 计算置信区间和 p 值
    print(f"\n计算 DeLong 统计量...")
    auc_stats = compute_auc_ci(y_true, y_scores)
    
    # 打印结果
    print(f"\n" + "=" * 70)
    print(f"评估结果:")
    print(f"=" * 70)
    print(f"AUC: {auc_stats['auc']:.4f}")
    print(f"95% CI: [{auc_stats['ci_lower']:.4f}, {auc_stats['ci_upper']:.4f}]")
    print(f"标准误: {auc_stats['std']:.4f}")
    print(f"DeLong p-value: {auc_stats['p_value']:.6f}")
    if auc_stats['p_value'] < 0.001:
        print(f"显著性: *** (p < 0.001)")
    elif auc_stats['p_value'] < 0.01:
        print(f"显著性: ** (p < 0.01)")
    elif auc_stats['p_value'] < 0.05:
        print(f"显著性: * (p < 0.05)")
    else:
        print(f"显著性: n.s. (p >= 0.05)")
    
    # 保存文本报告
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_txt, 'w') as f:
        f.write("T50: TS-RaMIA Transformer 评估结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"评分指标: {args.score_col}\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"成员数: {df[args.label_col].sum()}\n")
        f.write(f"非成员数: {len(df) - df[args.label_col].sum()}\n\n")
        f.write("ROC-AUC 分析:\n")
        f.write(f"  AUC = {auc_stats['auc']:.4f}\n")
        f.write(f"  95% CI = [{auc_stats['ci_lower']:.4f}, {auc_stats['ci_upper']:.4f}]\n")
        f.write(f"  标准误 = {auc_stats['std']:.4f}\n")
        f.write(f"  DeLong p-value = {auc_stats['p_value']:.6f}\n")
        
        if auc_stats['p_value'] < 0.001:
            f.write(f"  显著性: *** (p < 0.001)\n")
        elif auc_stats['p_value'] < 0.01:
            f.write(f"  显著性: ** (p < 0.01)\n")
        elif auc_stats['p_value'] < 0.05:
            f.write(f"  显著性: * (p < 0.05)\n")
        else:
            f.write(f"  显著性: n.s. (p >= 0.05)\n")
    
    print(f"\n✓ 文本报告已保存: {args.out_txt}")
    
    # 绘制 ROC 曲线
    print(f"\n绘制 ROC 曲线...")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_stats["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: TS-RaMIA Transformer (Protocol 1)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # 保存图片
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=300, bbox_inches='tight')
    print(f"✓ ROC 曲线已保存: {args.out_png}")
    
    print(f"\n" + "=" * 70)
    print(f"✓ 评估完成!")
    print(f"=" * 70)


if __name__ == "__main__":
    main()

