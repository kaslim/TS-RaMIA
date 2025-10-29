#!/usr/bin/env python3
"""
绘制学术论文级别的ROC曲线
使用规范的命名: Baseline (mean NLL), StructTail-64, StructTail+Fusion
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

# 设置学术论文风格
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
})

# 学术论文配色方案 (色盲友好)
COLORS = {
    'baseline': '#0173B2',      # 蓝色
    'structtail64': '#DE8F05',  # 橙色
    'fusion': '#029E73',        # 绿色
    'random': '#333333',        # 深灰色
}

# 线型
LINESTYLES = {
    'baseline': '-',       # 实线
    'structtail64': '--',  # 虚线
    'fusion': '-.',        # 点划线
    'random': ':',         # 点线
}

def plot_roc_main(df, output_dir):
    """绘制主ROC曲线"""
    print("绘制主ROC曲线...")
    
    # 准备数据
    y_true = df['is_member'].values
    scores_baseline = df['tis_mean'].values
    scores_structtail64 = df['tis_topk64_mean'].values
    scores_fusion = df['meta_score'].values if 'meta_score' in df.columns else df['tis_topk64_mean'].values
    
    # 计算ROC曲线
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true, scores_baseline)
    fpr_structtail64, tpr_structtail64, _ = roc_curve(y_true, scores_structtail64)
    fpr_fusion, tpr_fusion, _ = roc_curve(y_true, scores_fusion)
    
    # 计算AUC
    auc_baseline = roc_auc_score(y_true, scores_baseline)
    auc_structtail64 = roc_auc_score(y_true, scores_structtail64)
    auc_fusion = roc_auc_score(y_true, scores_fusion)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 绘制ROC曲线
    ax.plot(fpr_baseline, tpr_baseline, 
            linestyle=LINESTYLES['baseline'],
            color=COLORS['baseline'],
            linewidth=2.5,
            label=f'Baseline (mean NLL), AUC={auc_baseline:.3f}',
            alpha=0.9)
    
    ax.plot(fpr_structtail64, tpr_structtail64,
            linestyle=LINESTYLES['structtail64'],
            color=COLORS['structtail64'],
            linewidth=2.5,
            label=f'StructTail-64, AUC={auc_structtail64:.3f}',
            alpha=0.9)
    
    ax.plot(fpr_fusion, tpr_fusion,
            linestyle=LINESTYLES['fusion'],
            color=COLORS['fusion'],
            linewidth=2.5,
            label=f'StructTail+Fusion, AUC={auc_fusion:.3f}',
            alpha=0.9)
    
    # 绘制随机分类器基线
    ax.plot([0, 1], [0, 1], 
            linestyle=LINESTYLES['random'],
            color=COLORS['random'],
            linewidth=1.5,
            label='Random',
            alpha=0.6)
    
    # 设置坐标轴
    ax.set_xlabel('False Positive Rate', fontweight='normal')
    ax.set_ylabel('True Positive Rate', fontweight='normal')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # 网格
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    # 图例
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, 
              edgecolor='black', fancybox=False, shadow=False)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存
    output_pdf = output_dir / 'roc_main_academic.pdf'
    output_png = output_dir / 'roc_main_academic.png'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    
    print(f"  ✓ 保存: {output_pdf}")
    print(f"  ✓ 保存: {output_png}")
    print(f"  AUC - Baseline: {auc_baseline:.4f}")
    print(f"  AUC - StructTail-64: {auc_structtail64:.4f}")
    print(f"  AUC - StructTail+Fusion: {auc_fusion:.4f}")

def plot_roc_lowfpr(df, output_dir):
    """绘制低FPR区域的ROC曲线"""
    print("\n绘制低FPR区域ROC曲线...")
    
    # 准备数据
    y_true = df['is_member'].values
    scores_baseline = df['tis_mean'].values
    scores_structtail64 = df['tis_topk64_mean'].values
    scores_fusion = df['meta_score'].values if 'meta_score' in df.columns else df['tis_topk64_mean'].values
    
    # 计算ROC曲线
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true, scores_baseline)
    fpr_structtail64, tpr_structtail64, _ = roc_curve(y_true, scores_structtail64)
    fpr_fusion, tpr_fusion, _ = roc_curve(y_true, scores_fusion)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 只显示FPR 0-5%区域
    mask_baseline = fpr_baseline <= 0.05
    mask_structtail64 = fpr_structtail64 <= 0.05
    mask_fusion = fpr_fusion <= 0.05
    
    # 绘制ROC曲线（百分比）
    ax.plot(fpr_baseline[mask_baseline] * 100, 
            tpr_baseline[mask_baseline] * 100,
            linestyle=LINESTYLES['baseline'],
            color=COLORS['baseline'],
            linewidth=2.5,
            label='Baseline (mean NLL)',
            alpha=0.9)
    
    ax.plot(fpr_structtail64[mask_structtail64] * 100,
            tpr_structtail64[mask_structtail64] * 100,
            linestyle=LINESTYLES['structtail64'],
            color=COLORS['structtail64'],
            linewidth=2.5,
            label='StructTail-64',
            alpha=0.9)
    
    ax.plot(fpr_fusion[mask_fusion] * 100,
            tpr_fusion[mask_fusion] * 100,
            linestyle=LINESTYLES['fusion'],
            color=COLORS['fusion'],
            linewidth=2.5,
            label='StructTail+Fusion',
            alpha=0.9)
    
    # 随机分类器基线
    ax.plot([0, 5], [0, 5],
            linestyle=LINESTYLES['random'],
            color=COLORS['random'],
            linewidth=1.5,
            label='Random',
            alpha=0.6)
    
    # 标记关键点 (TPR @ 1% FPR)
    if any(fpr_fusion <= 0.01):
        tpr_at_1fpr = tpr_fusion[np.where(fpr_fusion <= 0.01)[0][-1]]
        ax.axvline(1, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
        ax.text(1.2, tpr_at_1fpr * 100, f'{tpr_at_1fpr*100:.1f}%',
                fontsize=9, color=COLORS['fusion'], fontweight='normal')
    
    # 设置坐标轴
    ax.set_xlabel('False Positive Rate (%)', fontweight='normal')
    ax.set_ylabel('True Positive Rate (%)', fontweight='normal')
    ax.set_xlim([-0.1, 5.1])
    ax.set_ylim([-1, 50])
    
    # 网格
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    # 图例
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
              edgecolor='black', fancybox=False, shadow=False)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存
    output_pdf = output_dir / 'roc_lowfpr_academic.pdf'
    output_png = output_dir / 'roc_lowfpr_academic.png'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    
    print(f"  ✓ 保存: {output_pdf}")
    print(f"  ✓ 保存: {output_png}")

def main():
    print("=" * 80)
    print("绘制学术论文级别ROC曲线")
    print("=" * 80)
    
    # 路径设置
    workspace = Path("/home/yons/文档/AAAI")
    data_path = workspace / "best_results/T53v2_breakthrough/piece_level_win_multi_calibrated.csv"
    output_dir = workspace / "paper/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查数据文件
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return 1
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  数据形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    
    # 检查必需的列
    required_cols = ['is_member', 'tis_mean', 'tis_topk64_mean']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必需的列: {missing_cols}")
        return 1
    
    # 绘制图形
    try:
        plot_roc_main(df, output_dir)
        plot_roc_lowfpr(df, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ 所有图形绘制完成！")
        print("=" * 80)
        print(f"\n输出目录: {output_dir}")
        print("  - roc_main_academic.pdf/png")
        print("  - roc_lowfpr_academic.pdf/png")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

