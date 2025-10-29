#!/usr/bin/env python3
"""
B5 后处理: 聚合 + 融合多温度特征
对多个 k 和 T 的组合进行 z-score 融合 (max/mean/geometric)
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from scipy import stats

def load_splits(split_json_path):
    """加载数据划分"""
    with open(split_json_path) as f:
        splits = json.load(f)
    
    if 'train_set' in splits:
        train_set = set(splits['train_set'])
    elif 'train' in splits:
        train_set = set([item['midi_filename'] for item in splits['train']])
    else:
        raise ValueError("无法识别 split JSON 格式")
    
    return train_set

def z_score_normalize(scores, reference_scores=None):
    """Z-score 标准化"""
    if reference_scores is not None:
        # 用参考分数计算均值和标准差
        mean = np.mean(reference_scores)
        std = np.std(reference_scores)
    else:
        mean = np.mean(scores)
        std = np.std(scores)
    
    if std == 0:
        return scores - mean
    return (scores - mean) / std

def geometric_mean(values):
    """几何均值"""
    # 平移到正数域
    min_val = values.min()
    if min_val <= 0:
        shifted = values - min_val + 1
    else:
        shifted = values
    
    return stats.gmean(shifted, axis=1)

def main():
    parser = argparse.ArgumentParser(description='B5 融合多温度特征')
    parser.add_argument('--scores_jsonl', required=True, help='B5 输出的 JSONL')
    parser.add_argument('--split_json', required=True, help='数据划分 JSON')
    parser.add_argument('--out_csv', required=True, help='输出融合后 CSV')
    parser.add_argument('--out_report', required=True, help='输出报告 JSON')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("B5: 多温度尾部融合")
    print("=" * 80)
    
    # 加载评分
    print(f"\n加载评分: {args.scores_jsonl}")
    samples = []
    with open(args.scores_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    
    df = pd.DataFrame(samples)
    print(f"  总 samples: {len(df)}")
    print(f"  unique pieces: {df['piece_id'].nunique()}")
    
    # 识别所有 tail-k@T 列
    score_cols = [col for col in df.columns if col.startswith('tail')]
    print(f"\n找到 {len(score_cols)} 个 tail-k@T 特征:")
    for col in sorted(score_cols):
        print(f"  - {col}")
    
    # 聚合到 piece-level
    print(f"\n聚合到 piece-level...")
    
    agg_funcs = {}
    for col in score_cols:
        agg_funcs[col] = ['mean', 'median', 'max', 'min']
    
    piece_df = df.groupby('piece_id').agg(agg_funcs)
    
    # 扁平化列名
    piece_df.columns = ['_'.join(col).strip() for col in piece_df.columns.values]
    piece_df = piece_df.reset_index()
    
    # 添加 n_segments (segment 计数)
    n_segments = df.groupby('piece_id').size().reset_index(name='n_segments')
    piece_df = piece_df.merge(n_segments, on='piece_id')
    
    # 添加 is_member
    train_set = load_splits(args.split_json)
    piece_df['is_member'] = piece_df['piece_id'].apply(lambda x: 1 if x in train_set else 0)
    
    print(f"  聚合后 pieces: {len(piece_df)}")
    print(f"  成员: {piece_df['is_member'].sum()}")
    print(f"  非成员: {(piece_df['is_member'] == 0).sum()}")
    
    # Z-score 标准化 (用非成员作为参考)
    print(f"\nZ-score 标准化 (非成员参考)...")
    
    non_member_mask = piece_df['is_member'] == 0
    
    # 收集所有均值特征
    mean_cols = [col for col in piece_df.columns if col.endswith('_mean') and col.startswith('tail')]
    
    z_scores = {}
    for col in mean_cols:
        ref_scores = piece_df.loc[non_member_mask, col].values
        all_scores = piece_df[col].values
        z_col = col.replace('_mean', '_z')
        piece_df[z_col] = z_score_normalize(all_scores, ref_scores)
        z_scores[z_col] = piece_df[z_col].values
    
    print(f"  标准化了 {len(mean_cols)} 个特征")
    
    # 融合: max, mean, geometric
    print(f"\n融合策略...")
    
    # 构建 z-score 矩阵
    z_matrix = np.column_stack([piece_df[col].values for col in sorted(z_scores.keys())])
    
    piece_df['fusion_max'] = z_matrix.max(axis=1)
    piece_df['fusion_mean'] = z_matrix.mean(axis=1)
    piece_df['fusion_gmean'] = geometric_mean(z_matrix)
    
    print(f"  ✓ fusion_max (最大 z-score)")
    print(f"  ✓ fusion_mean (平均 z-score)")
    print(f"  ✓ fusion_gmean (几何均值)")
    
    # 保存
    piece_df.to_csv(args.out_csv, index=False)
    print(f"\n✓ 保存: {args.out_csv}")
    
    # 生成报告
    report = {
        "task": "B5_aggregate_fusion",
        "date": pd.Timestamp.now().isoformat(),
        "input": {
            "scores_jsonl": args.scores_jsonl,
            "n_samples": len(df),
            "n_pieces": len(piece_df)
        },
        "features": {
            "n_tail_features": len(score_cols),
            "tail_features": sorted(score_cols),
            "fusion_methods": ["max", "mean", "gmean"]
        },
        "output": {
            "csv": args.out_csv,
            "n_features_total": len(piece_df.columns)
        }
    }
    
    with open(args.out_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ 报告: {args.out_report}")
    
    print(f"\n{'='*80}")
    print("✅ B5 融合完成")
    print(f"{'='*80}")
    
    # 显示融合特征统计
    print(f"\n融合特征统计 (成员 vs 非成员):")
    for fusion_col in ['fusion_max', 'fusion_mean', 'fusion_gmean']:
        member_mean = piece_df[piece_df['is_member'] == 1][fusion_col].mean()
        non_member_mean = piece_df[piece_df['is_member'] == 0][fusion_col].mean()
        print(f"  {fusion_col}:")
        print(f"    成员:   {member_mean:.4f}")
        print(f"    非成员: {non_member_mean:.4f}")
        print(f"    差异:   {member_mean - non_member_mean:.4f}")

if __name__ == "__main__":
    main()

