#!/usr/bin/env python3
"""
B6: EVT 尾部概率分数
在非成员分布上拟合 GPD，将 tail-k 映射为尾部 p-value
"""

import pandas as pd
import numpy as np
import json
import argparse
from scipy import stats
from sklearn.metrics import roc_auc_score

def fit_gpd_tail(data, threshold_percentile=90):
    """
    拟合 GPD (Generalized Pareto Distribution) 到尾部
    
    Args:
        data: 数据数组
        threshold_percentile: 尾部阈值百分位 (默认90%，即top 10%)
    
    Returns:
        threshold, shape, scale, loc
    """
    threshold = np.percentile(data, threshold_percentile)
    exceedances = data[data > threshold] - threshold
    
    if len(exceedances) < 10:
        # 数据太少，返回 None
        return None, None, None, None
    
    # 拟合 GPD
    try:
        shape, loc, scale = stats.genpareto.fit(exceedances)
        return threshold, shape, scale, loc
    except:
        return None, None, None, None

def compute_tail_pvalue(score, threshold, shape, scale, loc):
    """计算尾部 p-value (生存函数)"""
    if threshold is None or score <= threshold:
        # 低于阈值，用经验 CDF
        return 0.5  # 中性值
    
    exceedance = score - threshold
    
    # GPD 生存函数: P(X > x)
    try:
        p_tail = 1 - stats.genpareto.cdf(exceedance, shape, loc, scale)
        return float(p_tail)
    except:
        return 0.5

def main():
    parser = argparse.ArgumentParser(description='B6: EVT 尾部概率')
    parser.add_argument('--piece_csv', required=True, help='piece-level CSV (含 tail-k 特征)')
    parser.add_argument('--score_cols', nargs='+', help='要转换的 tail-k 列 (如 tail64_t1.0_mean)')
    parser.add_argument('--threshold_percentile', type=float, default=90, help='尾部阈值百分位')
    parser.add_argument('--out_csv', required=True, help='输出 CSV (添加 EVT p-value 列)')
    parser.add_argument('--out_report', required=True, help='输出报告 JSON')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("B6: EVT 尾部概率分数")
    print("=" * 80)
    
    # 加载数据
    print(f"\n加载: {args.piece_csv}")
    df = pd.read_csv(args.piece_csv)
    
    if 'is_member' not in df.columns:
        raise ValueError("CSV 缺少 is_member 列")
    
    print(f"  总 pieces: {len(df)}")
    print(f"  成员: {df['is_member'].sum()}")
    print(f"  非成员: {(df['is_member'] == 0).sum()}")
    
    # 自动识别 score_cols
    if args.score_cols is None:
        # 找所有 tail*_mean 列
        args.score_cols = [col for col in df.columns if col.startswith('tail') and col.endswith('_mean')]
        print(f"\n自动识别 {len(args.score_cols)} 个 tail-k 特征")
    
    print(f"\n处理 {len(args.score_cols)} 个特征:")
    for col in args.score_cols:
        print(f"  - {col}")
    
    # 非成员数据
    non_member_mask = df['is_member'] == 0
    
    evt_params = {}
    
    # 对每个特征拟合 GPD
    for col in args.score_cols:
        print(f"\n拟合 EVT: {col}")
        
        non_member_data = df.loc[non_member_mask, col].values
        non_member_data = non_member_data[~np.isnan(non_member_data)]
        
        if len(non_member_data) < 20:
            print(f"  ⚠️  数据不足，跳过")
            continue
        
        threshold, shape, scale, loc = fit_gpd_tail(
            non_member_data, 
            threshold_percentile=args.threshold_percentile
        )
        
        if threshold is None:
            print(f"  ⚠️  拟合失败，使用经验 CDF")
            # 使用经验 CDF
            evt_col = col.replace('_mean', '_evt_pval')
            df[evt_col] = df[col].apply(
                lambda x: 1 - stats.percentileofscore(non_member_data, x) / 100.0
            )
        else:
            print(f"  ✓ 阈值: {threshold:.4f}")
            print(f"  ✓ shape: {shape:.4f}, scale: {scale:.4f}")
            
            evt_params[col] = {
                'threshold': float(threshold),
                'shape': float(shape),
                'scale': float(scale),
                'loc': float(loc)
            }
            
            # 计算所有样本的 p-value
            evt_col = col.replace('_mean', '_evt_pval')
            df[evt_col] = df[col].apply(
                lambda x: compute_tail_pvalue(x, threshold, shape, scale, loc)
            )
        
        # 攻击分数 = 1 - p_value (p-value 越小越可疑)
        attack_col = col.replace('_mean', '_evt_score')
        df[attack_col] = 1 - df[evt_col]
        
        print(f"  ✓ 创建: {evt_col}, {attack_col}")
    
    # 保存
    df.to_csv(args.out_csv, index=False)
    print(f"\n✓ 保存: {args.out_csv}")
    
    # 计算每个 EVT 分数的 AUC
    print(f"\n{'='*80}")
    print("EVT 分数 AUC (初步):")
    print(f"{'='*80}")
    
    evt_score_cols = [col for col in df.columns if '_evt_score' in col]
    auc_results = {}
    
    for col in evt_score_cols:
        valid_mask = df[col].notna()
        if valid_mask.sum() > 0:
            auc = roc_auc_score(
                df.loc[valid_mask, 'is_member'],
                df.loc[valid_mask, col]
            )
            auc_results[col] = float(auc)
            print(f"  {col}: {auc:.4f}")
    
    # 生成报告
    report = {
        "task": "B6_evt_tail_prob",
        "date": pd.Timestamp.now().isoformat(),
        "config": {
            "threshold_percentile": args.threshold_percentile,
            "score_cols": args.score_cols
        },
        "evt_params": evt_params,
        "auc_results": auc_results,
        "output": {
            "csv": args.out_csv,
            "n_evt_features": len(evt_score_cols)
        }
    }
    
    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ 报告: {args.out_report}")
    
    print(f"\n{'='*80}")
    print("✅ B6 EVT 完成")
    print(f"{'='*80}")
    
    if auc_results:
        best_col = max(auc_results, key=auc_results.get)
        best_auc = auc_results[best_col]
        print(f"\n最佳 EVT 分数: {best_col}")
        print(f"  AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()

