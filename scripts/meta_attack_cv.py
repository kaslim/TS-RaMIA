#!/usr/bin/env python3
"""
T53: 元攻击器融合 - 整合多个弱特征为强分类器
使用 5-fold CV，Logistic Regression / SVM
"""

import argparse
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def select_features(df, exclude_patterns=['piece_id', 'split', 'is_member']):
    """选择特征列"""
    feature_cols = []
    for col in df.columns:
        # 排除特定列
        if any(pat in col for pat in exclude_patterns):
            continue
        # 只保留数值列
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    return feature_cols

def compute_low_fpr_metrics(y_true, y_score, fpr_thresholds=[0.001, 0.005, 0.01, 0.05, 0.1]):
    """计算低 FPR 指标"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    metrics = {}
    for target_fpr in fpr_thresholds:
        # 找到最接近目标 FPR 的点
        idx = np.argmin(np.abs(fpr - target_fpr))
        actual_fpr = fpr[idx]
        actual_tpr = tpr[idx]
        
        metrics[f'TPR@{target_fpr*100:.1f}%FPR'] = {
            'tpr': float(actual_tpr),
            'fpr': float(actual_fpr),
            'advantage': float(actual_tpr - actual_fpr)
        }
    
    # Partial AUC (0-1%)
    mask = fpr <= 0.01
    if mask.sum() > 1:
        pauc = np.trapz(tpr[mask], fpr[mask])
        pauc_norm = pauc / 0.01  # 标准化
        metrics['pAUC(0-1%)'] = {
            'value': float(pauc),
            'normalized': float(pauc_norm)
        }
    
    return metrics

def train_and_evaluate(X, y, model_type='logreg', calibration='isotonic', n_folds=5, random_state=1337):
    """训练并评估模型"""
    
    # 准备存储结果
    fold_results = []
    all_y_true = []
    all_y_scores = []
    
    # 5-fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_idx+1}/{n_folds}...", end=" ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 选择模型
        if model_type == 'logreg':
            base_model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=random_state
            )
        elif model_type == 'svm':
            base_model = SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 校准
        if calibration:
            model = CalibratedClassifierCV(
                base_model,
                method=calibration,
                cv=3
            )
        else:
            model = base_model
        
        # 训练
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_scores = model.predict_proba(X_test_scaled)[:, 1]
        
        # 评估
        auc = roc_auc_score(y_test, y_scores)
        low_fpr = compute_low_fpr_metrics(y_test, y_scores)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'auc': float(auc),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'low_fpr': low_fpr
        })
        
        all_y_true.extend(y_test)
        all_y_scores.extend(y_scores)
        
        print(f"AUC={auc:.4f}, TPR@1%={low_fpr['TPR@1.0%FPR']['tpr']*100:.1f}%")
    
    # 总体指标
    overall_auc = roc_auc_score(all_y_true, all_y_scores)
    overall_low_fpr = compute_low_fpr_metrics(all_y_true, all_y_scores)
    
    return fold_results, overall_auc, overall_low_fpr, all_y_true, all_y_scores

def main():
    parser = argparse.ArgumentParser(description='T53: 元攻击器融合')
    parser.add_argument('--csv', required=True, help='输入 CSV（piece-level 特征）')
    parser.add_argument('--label_col', default='is_member', help='标签列')
    parser.add_argument('--models', nargs='+', default=['logreg'], 
                       choices=['logreg', 'svm'], help='模型类型')
    parser.add_argument('--calibration', default='isotonic', 
                       choices=['isotonic', 'sigmoid', 'none'], help='校准方法')
    parser.add_argument('--folds', type=int, default=5, help='CV 折数')
    parser.add_argument('--out_json', required=True, help='输出 JSON 报告')
    parser.add_argument('--out_png', required=True, help='输出 ROC 图')
    parser.add_argument('--out_lowfpr', required=True, help='输出低 FPR 指标')
    parser.add_argument('--seed', type=int, default=1337, help='随机种子')
    
    args = parser.parse_args()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("T53: 元攻击器融合（冲击 AUC 0.7）")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    # 加载数据
    df = pd.read_csv(args.csv)
    print(f"📊 加载数据: {len(df)} 个作品")
    print(f"  成员: {(df[args.label_col]==1).sum()}")
    print(f"  非成员: {(df[args.label_col]==0).sum()}")
    print("")
    
    # 选择特征
    feature_cols = select_features(df)
    print(f"🎯 特征选择: {len(feature_cols)} 个特征")
    
    # 优先选择 TIS/PPL 相关特征
    priority_features = [c for c in feature_cols if any(x in c for x in ['tis', 'ppl', 'nll'])]
    if len(priority_features) > 50:
        priority_features = priority_features[:50]  # 限制特征数
    
    print(f"   优先特征: {len(priority_features)} 个")
    for i, feat in enumerate(priority_features[:10]):
        print(f"     {i+1}. {feat}")
    if len(priority_features) > 10:
        print(f"     ... 还有 {len(priority_features)-10} 个")
    print("")
    
    # 准备数据
    X = df[priority_features].fillna(0).values
    y = df[args.label_col].values
    
    # 训练和评估每个模型
    results = {}
    best_auc = 0
    best_y_true = None
    best_y_scores = None
    
    for model_type in args.models:
        print(f"🔄 训练模型: {model_type.upper()}")
        
        calibration = args.calibration if args.calibration != 'none' else None
        
        fold_results, overall_auc, overall_low_fpr, y_true, y_scores = train_and_evaluate(
            X, y, 
            model_type=model_type,
            calibration=calibration,
            n_folds=args.folds,
            random_state=args.seed
        )
        
        # 计算均值和标准差
        aucs = [r['auc'] for r in fold_results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        results[model_type] = {
            'fold_results': fold_results,
            'mean_auc': float(mean_auc),
            'std_auc': float(std_auc),
            'overall_auc': float(overall_auc),
            'overall_low_fpr': overall_low_fpr
        }
        
        print(f"  ✓ 平均 AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"    总体 AUC: {overall_auc:.4f}")
        print(f"    TPR@1%FPR: {overall_low_fpr['TPR@1.0%FPR']['tpr']*100:.1f}%")
        print("")
        
        # 记录最佳
        if overall_auc > best_auc:
            best_auc = overall_auc
            best_y_true = y_true
            best_y_scores = y_scores
    
    # 保存 JSON 报告
    summary = {
        'n_samples': len(df),
        'n_features': len(priority_features),
        'n_folds': args.folds,
        'calibration': args.calibration,
        'models': results,
        'best_model': max(results.items(), key=lambda x: x[1]['overall_auc'])[0],
        'best_auc': float(best_auc)
    }
    
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ JSON 报告已保存: {args.out_json}")
    
    # 保存低 FPR 指标
    best_model = summary['best_model']
    with open(args.out_lowfpr, 'w') as f:
        json.dump(results[best_model]['overall_low_fpr'], f, indent=2)
    print(f"✅ 低 FPR 指标已保存: {args.out_lowfpr}")
    
    # 绘制 ROC 曲线
    if best_y_true is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(best_y_true, best_y_scores)
        plt.plot(fpr, tpr, linewidth=2, label=f'Meta-Attack (AUC={best_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'T53: Meta-Attack ROC ({best_model.upper()})', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=150)
        plt.close()
        print(f"✅ ROC 曲线已保存: {args.out_png}")
    
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ T53 完成！最佳模型: {best_model.upper()}, AUC={best_auc:.4f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == '__main__':
    main()

