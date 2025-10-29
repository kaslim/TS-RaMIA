#!/usr/bin/env python3
"""
T50d: 长度匹配聚合 - 去除长度偏置
实现三种方法：分层重采样 (Stratified)、逆概率加权 (IPW)、计数裁剪 (Clip)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json

def stratified_resampling(df, bins, seed=1337):
    """方法 A: 分层重采样"""
    np.random.seed(seed)
    
    # 创建 bins
    df['n_seg_bin'] = pd.cut(df['n_segments'], bins=bins, include_lowest=True, duplicates='drop')
    
    balanced_dfs = []
    for bin_val in df['n_seg_bin'].unique():
        if pd.isna(bin_val):
            continue
        
        bin_df = df[df['n_seg_bin'] == bin_val]
        members = bin_df[bin_df['is_member'] == 1]
        non_members = bin_df[bin_df['is_member'] == 0]
        
        # 对齐到较小的那一方
        n_min = min(len(members), len(non_members))
        if n_min == 0:
            continue
        
        # 随机采样
        if len(members) > n_min:
            members = members.sample(n=n_min, random_state=seed)
        if len(non_members) > n_min:
            non_members = non_members.sample(n=n_min, random_state=seed)
        
        balanced_dfs.append(pd.concat([members, non_members]))
    
    result = pd.concat(balanced_dfs, ignore_index=True)
    result = result.drop(columns=['n_seg_bin'])
    return result

def ipw_weighting(df, bins):
    """方法 B: 逆概率加权 (IPW)"""
    df = df.copy()
    
    # 创建 bins
    df['n_seg_bin'] = pd.cut(df['n_segments'], bins=bins, include_lowest=True, duplicates='drop')
    
    # 计算每个 bin 的成员/非成员比例
    weights = []
    for idx, row in df.iterrows():
        bin_val = row['n_seg_bin']
        if pd.isna(bin_val):
            weights.append(1.0)
            continue
        
        bin_df = df[df['n_seg_bin'] == bin_val]
        n_members = (bin_df['is_member'] == 1).sum()
        n_non_members = (bin_df['is_member'] == 0).sum()
        
        if n_members == 0 or n_non_members == 0:
            weights.append(1.0)
            continue
        
        # 稀有类别给更高权重
        if row['is_member'] == 1:
            weight = min(n_members, n_non_members) / n_members
        else:
            weight = min(n_members, n_non_members) / n_non_members
        
        weights.append(weight)
    
    df['ipw_weight'] = weights
    df = df.drop(columns=['n_seg_bin'])
    
    # 加权聚合（这里返回带权重的数据，实际 AUC 计算时使用 sample_weight）
    return df

def count_clipping(df, k_clip):
    """方法 C: 计数裁剪"""
    # 找到成员和非成员的 k 分位数，取较小值
    k_member = df[df['is_member'] == 1]['n_segments'].quantile(0.95)
    k_non_member = df[df['is_member'] == 0]['n_segments'].quantile(0.95)
    k = min(int(k_clip), int(min(k_member, k_non_member)))
    
    print(f"  裁剪阈值: {k} segments (成员 p95={k_member:.1f}, 非成员 p95={k_non_member:.1f})")
    
    # 只保留 n_segments <= k 的作品
    result = df[df['n_segments'] <= k].copy()
    return result

def compute_ks_test(df):
    """计算 K-S 检验 p-value"""
    members = df[df['is_member'] == 1]['n_segments']
    non_members = df[df['is_member'] == 0]['n_segments']
    
    ks_stat, p_value = stats.ks_2samp(members, non_members)
    return ks_stat, p_value

def main():
    parser = argparse.ArgumentParser(description='T50d: 长度匹配聚合')
    parser.add_argument('--scores', required=True, help='Piece-level CSV 文件')
    parser.add_argument('--split_json', required=True, help='Split JSON (用于验证)')
    parser.add_argument('--method', default='all', choices=['all', 'strata', 'ipw', 'clip'],
                       help='聚合方法')
    parser.add_argument('--bins', nargs='+', type=int, 
                       default=[1, 4, 8, 12, 16, 20, 30, 50, 100],
                       help='分层 bins')
    parser.add_argument('--k_clip', type=int, default=15, help='裁剪阈值')
    parser.add_argument('--out_csv', required=True, help='输出 CSV')
    parser.add_argument('--seed', type=int, default=1337, help='随机种子')
    
    args = parser.parse_args()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("T50d: 长度匹配聚合（去除长度偏置）")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    # 加载数据
    df = pd.read_csv(args.scores)
    print(f"📊 加载数据: {len(df)} 个作品")
    print(f"  成员: {(df['is_member']==1).sum()}")
    print(f"  非成员: {(df['is_member']==0).sum()}")
    print("")
    
    # 原始 K-S 检验
    ks_orig, p_orig = compute_ks_test(df)
    print(f"📈 原始数据 K-S 检验:")
    print(f"  成员平均 n_segments: {df[df['is_member']==1]['n_segments'].mean():.2f}")
    print(f"  非成员平均 n_segments: {df[df['is_member']==0]['n_segments'].mean():.2f}")
    print(f"  K-S statistic: {ks_orig:.4f}")
    print(f"  p-value: {p_orig:.4g}")
    if p_orig < 0.2:
        print(f"  ⚠️  显著长度偏置 (p < 0.2)")
    print("")
    
    results = {}
    
    # 方法 A: 分层重采样
    if args.method in ['all', 'strata']:
        print("🔄 方法 A: 分层重采样")
        df_strata = stratified_resampling(df, args.bins, args.seed)
        ks_strata, p_strata = compute_ks_test(df_strata)
        
        print(f"  保留作品: {len(df_strata)} ({len(df_strata)*100/len(df):.1f}%)")
        print(f"  成员: {(df_strata['is_member']==1).sum()}")
        print(f"  非成员: {(df_strata['is_member']==0).sum()}")
        print(f"  成员平均 n_segments: {df_strata[df_strata['is_member']==1]['n_segments'].mean():.2f}")
        print(f"  非成员平均 n_segments: {df_strata[df_strata['is_member']==0]['n_segments'].mean():.2f}")
        print(f"  K-S p-value: {p_strata:.4g} {'✓' if p_strata >= 0.2 else '⚠️'}")
        print("")
        
        results['strata'] = df_strata
    
    # 方法 B: IPW
    if args.method in ['all', 'ipw']:
        print("🔄 方法 B: 逆概率加权 (IPW)")
        df_ipw = ipw_weighting(df, args.bins)
        
        print(f"  权重范围: [{df_ipw['ipw_weight'].min():.3f}, {df_ipw['ipw_weight'].max():.3f}]")
        print(f"  权重平均: {df_ipw['ipw_weight'].mean():.3f}")
        print(f"  (K-S 检验在加权数据上需特殊处理，此处跳过)")
        print("")
        
        results['ipw'] = df_ipw
    
    # 方法 C: 计数裁剪
    if args.method in ['all', 'clip']:
        print("🔄 方法 C: 计数裁剪")
        df_clip = count_clipping(df, args.k_clip)
        ks_clip, p_clip = compute_ks_test(df_clip)
        
        print(f"  保留作品: {len(df_clip)} ({len(df_clip)*100/len(df):.1f}%)")
        print(f"  成员: {(df_clip['is_member']==1).sum()}")
        print(f"  非成员: {(df_clip['is_member']==0).sum()}")
        print(f"  成员平均 n_segments: {df_clip[df_clip['is_member']==1]['n_segments'].mean():.2f}")
        print(f"  非成员平均 n_segments: {df_clip[df_clip['is_member']==0]['n_segments'].mean():.2f}")
        print(f"  K-S p-value: {p_clip:.4g} {'✓' if p_clip >= 0.2 else '⚠️'}")
        print("")
        
        results['clip'] = df_clip
    
    # 选择主方法（默认 strata）
    if 'strata' in results:
        primary_method = 'strata'
        primary_df = results['strata']
    elif 'clip' in results:
        primary_method = 'clip'
        primary_df = results['clip']
    elif 'ipw' in results:
        primary_method = 'ipw'
        primary_df = results['ipw']
    else:
        raise ValueError("No method selected")
    
    # 保存结果
    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    primary_df.to_csv(output_path, index=False)
    print(f"✅ 主方法 ({primary_method}) 结果已保存: {output_path}")
    
    # 保存比较报告
    summary = {
        'original': {
            'n_pieces': len(df),
            'n_members': int((df['is_member']==1).sum()),
            'n_non_members': int((df['is_member']==0).sum()),
            'mean_n_seg_member': float(df[df['is_member']==1]['n_segments'].mean()),
            'mean_n_seg_non_member': float(df[df['is_member']==0]['n_segments'].mean()),
            'ks_statistic': float(ks_orig),
            'ks_pvalue': float(p_orig)
        }
    }
    
    for method, df_result in results.items():
        if method == 'ipw':
            summary[method] = {
                'n_pieces': len(df_result),
                'has_weights': True,
                'weight_min': float(df_result['ipw_weight'].min()),
                'weight_max': float(df_result['ipw_weight'].max())
            }
        else:
            ks_stat, p_val = compute_ks_test(df_result)
            summary[method] = {
                'n_pieces': len(df_result),
                'n_members': int((df_result['is_member']==1).sum()),
                'n_non_members': int((df_result['is_member']==0).sum()),
                'mean_n_seg_member': float(df_result[df_result['is_member']==1]['n_segments'].mean()),
                'mean_n_seg_non_member': float(df_result[df_result['is_member']==0]['n_segments'].mean()),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(p_val),
                'passed': bool(p_val >= 0.2)
            }
    
    summary_path = output_path.parent / 'length_match_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ 比较报告已保存: {summary_path}")
    
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ T50d 完成！主方法: {primary_method}")
    print(f"   输出: {output_path}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == '__main__':
    main()

