#!/usr/bin/env python3
"""
T50e: 条件校准 - 残差化 + CDF 标准化
消除混杂变量对评分的影响
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def residualize(df, score_cols, confound_cols, label_col='is_member'):
    """方法1: 残差化 - 在非成员上拟合，得到残差"""
    print("📊 方法1: 残差化")
    
    # 只在非成员上拟合
    non_members = df[df[label_col] == 0].copy()
    
    # 准备混杂变量
    X_nm = non_members[confound_cols].values
    
    # 对每个评分列进行残差化
    residual_cols = []
    for score_col in score_cols:
        print(f"  处理: {score_col}")
        
        # 拟合回归模型（用浅层梯度提升树）
        y_nm = non_members[score_col].values
        
        # 处理 NaN
        valid_idx = ~(np.isnan(X_nm).any(axis=1) | np.isnan(y_nm))
        if valid_idx.sum() < 10:
            print(f"    ⚠️ 有效样本太少，跳过")
            continue
        
        X_nm_valid = X_nm[valid_idx]
        y_nm_valid = y_nm[valid_idx]
        
        # 标准化混杂变量
        scaler = StandardScaler()
        X_nm_scaled = scaler.fit_transform(X_nm_valid)
        
        # 拟合模型
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=1337
        )
        model.fit(X_nm_scaled, y_nm_valid)
        
        # 在全部数据上预测
        X_all = df[confound_cols].values
        valid_all = ~np.isnan(X_all).any(axis=1)
        
        predictions = np.full(len(df), np.nan)
        if valid_all.sum() > 0:
            X_all_scaled = scaler.transform(X_all[valid_all])
            predictions[valid_all] = model.predict(X_all_scaled)
        
        # 计算残差
        residuals = df[score_col].values - predictions
        
        # 保存
        residual_col = f"{score_col}_resid"
        df[residual_col] = residuals
        residual_cols.append(residual_col)
        
        # 计算相关性下降
        orig_corr = np.abs([df[df[label_col]==0][score_col].corr(df[df[label_col]==0][conf]) 
                           for conf in confound_cols])
        resid_corr = np.abs([df[df[label_col]==0][residual_col].corr(df[df[label_col]==0][conf]) 
                            for conf in confound_cols])
        
        orig_corr_mean = np.nanmean(orig_corr)
        resid_corr_mean = np.nanmean(resid_corr)
        
        if orig_corr_mean > 0:
            reduction = (orig_corr_mean - resid_corr_mean) / orig_corr_mean * 100
            print(f"    原始相关性: {orig_corr_mean:.4f} → 残差相关性: {resid_corr_mean:.4f} (↓{reduction:.1f}%)")
    
    return df, residual_cols

def cdf_normalize(df, score_cols, confound_cols, label_col='is_member', n_bins=10):
    """方法2: CDF 标准化 - 简化版全局 CDF"""
    print("\n📊 方法2: CDF 标准化 (全局)")
    
    # 只在非成员上计算 CDF
    non_members = df[df[label_col] == 0].copy()
    
    cdf_cols = []
    for score_col in score_cols:
        print(f"  处理: {score_col}")
        
        # 使用全局 CDF（简化版）
        nm_scores = non_members[score_col].dropna().values
        
        if len(nm_scores) < 10:
            print(f"    ⚠️ 非成员样本太少，跳过")
            continue
        
        # 计算每个样本的分位数位置
        cdf_scores = []
        for score in df[score_col].values:
            if np.isnan(score):
                cdf_scores.append(np.nan)
            else:
                cdf = stats.percentileofscore(nm_scores, score, kind='rank') / 100.0
                cdf_scores.append(cdf)
        
        cdf_col = f"{score_col}_cdf"
        df[cdf_col] = cdf_scores
        cdf_cols.append(cdf_col)
        
        print(f"    CDF 范围: [{np.nanmin(cdf_scores):.3f}, {np.nanmax(cdf_scores):.3f}]")
    
    return df, cdf_cols

def main():
    parser = argparse.ArgumentParser(description='T50e: 条件校准')
    parser.add_argument('--csv_in', required=True, help='输入 CSV')
    parser.add_argument('--label_col', default='is_member', help='标签列')
    parser.add_argument('--confounds', nargs='+', required=True, help='混杂变量列')
    parser.add_argument('--methods', nargs='+', default=['residual', 'cdf'],
                       choices=['residual', 'cdf'], help='校准方法')
    parser.add_argument('--csv_out', required=True, help='输出 CSV')
    
    args = parser.parse_args()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("T50e: 条件校准（残差化 + CDF 标准化）")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    # 加载数据
    df = pd.read_csv(args.csv_in)
    print(f"📊 加载数据: {len(df)} 个作品")
    print(f"  成员: {(df[args.label_col]==1).sum()}")
    print(f"  非成员: {(df[args.label_col]==0).sum()}")
    print("")
    
    # 检查混杂变量
    print(f"📋 混杂变量:")
    valid_confounds = []
    for conf in args.confounds:
        if conf in df.columns:
            print(f"  ✓ {conf}")
            valid_confounds.append(conf)
        else:
            print(f"  ✗ {conf} (缺失)")
    
    if not valid_confounds:
        print("\n❌ 没有有效的混杂变量！")
        return
    
    print("")
    
    # 选择要校准的评分列（TIS 和 PPL 相关）
    score_cols = [col for col in df.columns if any(x in col for x in ['tis_', 'ppl_']) 
                  and not any(x in col for x in ['_resid', '_cdf'])]
    
    # 限制数量，避免太多列
    priority_scores = ['tis_win_p95_mean', 'tis_mean', 'tis_win_max_max', 
                      'ppl_fwd_mean', 'ppl_rev_mean']
    score_cols = [col for col in priority_scores if col in df.columns]
    
    print(f"🎯 将校准的评分列 ({len(score_cols)} 个):")
    for col in score_cols:
        print(f"  • {col}")
    print("")
    
    # 应用校准方法
    if 'residual' in args.methods:
        df, resid_cols = residualize(df, score_cols, valid_confounds, args.label_col)
    
    if 'cdf' in args.methods:
        df, cdf_cols = cdf_normalize(df, score_cols, valid_confounds, args.label_col)
    
    # 保存结果
    df.to_csv(args.csv_out, index=False)
    print(f"\n✅ 校准结果已保存: {args.csv_out}")
    print(f"   新增列数: {len([c for c in df.columns if '_resid' in c or '_cdf' in c])}")
    
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ T50e 完成！")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == '__main__':
    main()

