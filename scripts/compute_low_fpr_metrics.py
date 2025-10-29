#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T50: 低 FPR 区间指标计算
计算 TPR@FPR、Advantage@FPR、Partial AUC 等关键指标
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import json


def compute_tpr_at_fpr(fpr, tpr, target_fpr):
    """计算指定 FPR 下的 TPR"""
    # 找到最接近 target_fpr 的索引
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx], fpr[idx]


def compute_advantage_at_fpr(fpr, tpr, target_fpr):
    """
    计算 Advantage@FPR
    Advantage = TPR - FPR (在指定 FPR 下)
    """
    tpr_val, actual_fpr = compute_tpr_at_fpr(fpr, tpr, target_fpr)
    advantage = tpr_val - actual_fpr
    return advantage, tpr_val, actual_fpr


def compute_partial_auc(fpr, tpr, fpr_max=0.01):
    """
    计算 Partial AUC (0 到 fpr_max 区间)
    标准化到 [0, 1] 区间
    """
    # 截取 FPR <= fpr_max 的部分
    mask = fpr <= fpr_max
    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]
    
    # 如果没有到达 fpr_max，插值
    if fpr_partial[-1] < fpr_max:
        # 线性插值
        idx = np.searchsorted(fpr, fpr_max)
        if idx < len(fpr):
            alpha = (fpr_max - fpr[idx-1]) / (fpr[idx] - fpr[idx-1])
            tpr_interp = tpr[idx-1] + alpha * (tpr[idx] - tpr[idx-1])
            fpr_partial = np.append(fpr_partial, fpr_max)
            tpr_partial = np.append(tpr_partial, tpr_interp)
    
    # 计算 partial AUC
    pauc = auc(fpr_partial, tpr_partial)
    
    # 标准化：max possible AUC in [0, fpr_max] is fpr_max
    # 标准化到 [0, 1]: (actual - min) / (max - min)
    # min = 0 (diagonal), max = fpr_max (perfect classifier in this region)
    pauc_normalized = pauc / fpr_max
    
    return pauc, pauc_normalized, fpr_partial, tpr_partial


def main():
    parser = argparse.ArgumentParser(description="T50: 低 FPR 区间指标计算")
    parser.add_argument("--csv", required=True, help="作品级 CSV 文件")
    parser.add_argument("--score_col", default="tis_mean", help="评分列名")
    parser.add_argument("--label_col", default="is_member", help="标签列名")
    parser.add_argument("--out_json", required=True, help="输出 JSON 报告")
    parser.add_argument("--out_plot", default=None, help="输出低 FPR 区间 ROC 图（可选）")
    args = parser.parse_args()
    
    print("=" * 70)
    print("T50: 低 FPR 区间指标计算")
    print("=" * 70)
    print(f"输入 CSV: {args.csv}")
    print(f"评分列: {args.score_col}")
    print(f"标签列: {args.label_col}")
    
    # 加载数据
    print(f"\n加载数据...")
    df = pd.read_csv(args.csv)
    y_true = df[args.label_col].values
    y_scores = df[args.score_col].values
    
    print(f"  总样本数: {len(df)}")
    print(f"  成员数: {y_true.sum()}")
    print(f"  非成员数: {(y_true == 0).sum()}")
    
    # 计算 ROC 曲线
    print(f"\n计算 ROC 曲线...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # 定义要计算的 FPR 点
    fpr_targets = [0.001, 0.005, 0.01, 0.05, 0.1]  # 0.1%, 0.5%, 1%, 5%, 10%
    
    print(f"\n计算低 FPR 指标...")
    results = {
        "tpr_at_fpr": {},
        "advantage_at_fpr": {},
        "partial_auc": {}
    }
    
    # 计算各 FPR 点的指标
    for target_fpr in fpr_targets:
        # TPR@FPR
        tpr_val, actual_fpr = compute_tpr_at_fpr(fpr, tpr, target_fpr)
        results["tpr_at_fpr"][f"{target_fpr*100:.1f}%"] = {
            "target_fpr": target_fpr,
            "actual_fpr": float(actual_fpr),
            "tpr": float(tpr_val)
        }
        
        # Advantage@FPR
        adv, tpr_val, actual_fpr = compute_advantage_at_fpr(fpr, tpr, target_fpr)
        results["advantage_at_fpr"][f"{target_fpr*100:.1f}%"] = {
            "target_fpr": target_fpr,
            "actual_fpr": float(actual_fpr),
            "advantage": float(adv),
            "tpr": float(tpr_val)
        }
    
    # 计算 Partial AUC
    pauc_ranges = [0.001, 0.01, 0.05]  # 0-0.1%, 0-1%, 0-5%
    for fpr_max in pauc_ranges:
        pauc, pauc_norm, fpr_p, tpr_p = compute_partial_auc(fpr, tpr, fpr_max)
        results["partial_auc"][f"0-{fpr_max*100:.1f}%"] = {
            "fpr_max": fpr_max,
            "pauc": float(pauc),
            "pauc_normalized": float(pauc_norm),
            "n_points": len(fpr_p)
        }
    
    # 打印结果
    print(f"\n{'=' * 70}")
    print("低 FPR 指标结果:")
    print(f"{'=' * 70}")
    
    print(f"\n1. TPR @ FPR:")
    for key, val in results["tpr_at_fpr"].items():
        print(f"   TPR@{key}FPR = {val['tpr']:.4f} (实际 FPR={val['actual_fpr']:.5f})")
    
    print(f"\n2. Advantage @ FPR (TPR - FPR):")
    for key, val in results["advantage_at_fpr"].items():
        print(f"   Adv@{key}FPR = {val['advantage']:.4f} (TPR={val['tpr']:.4f}, FPR={val['actual_fpr']:.5f})")
    
    print(f"\n3. Partial AUC:")
    for key, val in results["partial_auc"].items():
        print(f"   pAUC({key}FPR) = {val['pauc']:.6f} (标准化: {val['pauc_normalized']:.4f})")
    
    # 保存 JSON 报告
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ JSON 报告已保存: {args.out_json}")
    
    # 绘制低 FPR 区间 ROC 图（可选）
    if args.out_plot:
        print(f"\n绘制低 FPR 区间 ROC 曲线...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 0-1% FPR
        ax1 = axes[0]
        mask1 = fpr <= 0.01
        ax1.plot(fpr[mask1], tpr[mask1], 'b-', lw=2, label='ROC curve')
        ax1.plot([0, 0.01], [0, 0.01], 'k--', lw=1, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve (FPR: 0-1%)')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 0.01])
        ax1.set_ylim([0, max(tpr[mask1])*1.1 if len(tpr[mask1]) > 0 else 0.1])
        
        # 右图: 0-10% FPR
        ax2 = axes[1]
        mask2 = fpr <= 0.1
        ax2.plot(fpr[mask2], tpr[mask2], 'b-', lw=2, label='ROC curve')
        ax2.plot([0, 0.1], [0, 0.1], 'k--', lw=1, label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve (FPR: 0-10%)')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
        ax2.set_xlim([0, 0.1])
        ax2.set_ylim([0, max(tpr[mask2])*1.1 if len(tpr[mask2]) > 0 else 0.1])
        
        plt.tight_layout()
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_plot, dpi=150, bbox_inches='tight')
        print(f"✓ 低 FPR 区间 ROC 图已保存: {args.out_plot}")
        plt.close()
    
    print(f"\n{'=' * 70}")
    print("✓ 低 FPR 指标计算完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

