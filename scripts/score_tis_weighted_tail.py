#!/usr/bin/env python3
"""
T52: 置信度加权 + 尾部聚合
实现 token 级加权和 top-k 尾部统计来放大稀疏记忆信号
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class TokenDataset(Dataset):
    def __init__(self, jsonl_path):
        self.items = []
        with open(jsonl_path) as f:
            for line_no, line in enumerate(f, 1):
                item = json.loads(line)
                item['line_no'] = line_no
                self.items.append(item)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            'ids': torch.tensor(item['ids'], dtype=torch.long),
            'piece_id': item.get('piece_id', f'unknown_{idx}'),
            'seg_idx': item.get('seg_idx', 0),
        }

@torch.no_grad()
def compute_weighted_ppl(model, input_ids, device, weight_mode='invprob', epsilon=1e-6, 
                        topk_list=[32, 64, 128], win_size=256, win_stride=128):
    """
    计算加权 PPL 和尾部聚合统计
    
    weight_mode:
        - 'invprob': w = 1 / (p_true + epsilon)
        - 'entropy': w = -Σ p log p
        - 'uniform': w = 1 (baseline)
    """
    model.eval()
    input_ids = input_ids.unsqueeze(0).to(device)
    
    # Forward pass
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits
    
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # 获取概率分布
    probs = F.softmax(shift_logits, dim=-1)
    
    # 计算每个 token 的 NLL
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_nlls = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # 有效 token mask
    valid_mask = shift_labels.view(-1) != -100
    valid_nlls = token_nlls[valid_mask].cpu().numpy()
    n_tokens = valid_mask.sum().item()
    
    # 计算 token 级权重
    if weight_mode == 'invprob':
        # w = 1 / (p_true + epsilon)
        true_probs = probs.view(-1, probs.size(-1))[torch.arange(probs.view(-1, probs.size(-1)).size(0)), 
                                                      shift_labels.view(-1)]
        weights = 1.0 / (true_probs[valid_mask] + epsilon)
        weights = weights.cpu().numpy()
        
    elif weight_mode == 'entropy':
        # w = entropy = -Σ p log p
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropies = -(probs * log_probs).sum(dim=-1)
        weights = entropies.view(-1)[valid_mask].cpu().numpy()
        
    else:  # uniform
        weights = np.ones(len(valid_nlls))
    
    # 标准化权重
    weights = weights / weights.sum() * len(weights)
    
    # 全局加权 PPL
    weighted_nll = float((valid_nlls * weights).sum() / weights.sum())
    weighted_ppl = math.exp(weighted_nll) if math.isfinite(weighted_nll) else float('inf')
    
    # 全局均匀 PPL (baseline)
    uniform_nll = float(valid_nlls.mean())
    uniform_ppl = math.exp(uniform_nll) if math.isfinite(uniform_nll) else float('inf')
    
    # Top-k 尾部统计（最大 NLL）
    topk_stats = {}
    for k in topk_list:
        if len(valid_nlls) >= k:
            topk_nlls = np.sort(valid_nlls)[-k:]
            topk_mean_nll = float(topk_nlls.mean())
            topk_mean_ppl = math.exp(topk_mean_nll) if math.isfinite(topk_mean_nll) else float('inf')
            topk_stats[f'topk{k}'] = {
                'nll': topk_mean_nll,
                'ppl': topk_mean_ppl
            }
    
    # 窗口统计（加权）
    window_weighted_ppls = []
    window_topk_ppls = {k: [] for k in topk_list}
    
    if n_tokens >= win_size:
        for start in range(0, len(valid_nlls) - win_size + 1, win_stride):
            end = start + win_size
            win_nlls = valid_nlls[start:end]
            win_weights = weights[start:end]
            
            # 窗口加权 PPL
            win_weighted_nll = float((win_nlls * win_weights).sum() / win_weights.sum())
            win_weighted_ppl = math.exp(win_weighted_nll) if math.isfinite(win_weighted_nll) else float('inf')
            if math.isfinite(win_weighted_ppl):
                window_weighted_ppls.append(win_weighted_ppl)
            
            # 窗口 top-k
            for k in topk_list:
                if len(win_nlls) >= k:
                    win_topk_nlls = np.sort(win_nlls)[-k:]
                    win_topk_nll = float(win_topk_nlls.mean())
                    win_topk_ppl = math.exp(win_topk_nll) if math.isfinite(win_topk_nll) else float('inf')
                    if math.isfinite(win_topk_ppl):
                        window_topk_ppls[k].append(win_topk_ppl)
    
    # 窗口极值
    win_stats = {}
    if window_weighted_ppls:
        win_stats['weighted'] = {
            'min': float(np.min(window_weighted_ppls)),
            'max': float(np.max(window_weighted_ppls)),
            'p95': float(np.percentile(window_weighted_ppls, 95)),
            'mean': float(np.mean(window_weighted_ppls)),
            'median': float(np.median(window_weighted_ppls))
        }
    
    for k, ppls in window_topk_ppls.items():
        if ppls:
            win_stats[f'topk{k}'] = {
                'min': float(np.min(ppls)),
                'max': float(np.max(ppls)),
                'p95': float(np.percentile(ppls, 95)),
                'mean': float(np.mean(ppls)),
                'median': float(np.median(ppls))
            }
    
    return {
        'n_tokens': n_tokens,
        'uniform_ppl': uniform_ppl,
        'uniform_nll': uniform_nll,
        'weighted_ppl': weighted_ppl,
        'weighted_nll': weighted_nll,
        'topk_stats': topk_stats,
        'win_stats': win_stats
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--train_manifest', required=True)
    parser.add_argument('--val_manifest', required=True)
    parser.add_argument('--output_jsonl', required=True)
    parser.add_argument('--weight_mode', default='invprob', choices=['invprob', 'entropy', 'uniform'])
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--topk', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--win_size', type=int, default=256)
    parser.add_argument('--win_stride', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1337)
    
    args = parser.parse_args()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("T52: 置信度加权 + 尾部聚合")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\n权重模式: {args.weight_mode}")
    print(f"Top-k: {args.topk}")
    print(f"窗口: size={args.win_size}, stride={args.win_stride}")
    print("")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    print("✓ 模型加载完成")
    
    # 加载数据
    print(f"\n加载数据:")
    print(f"  训练集: {args.train_manifest}")
    train_dataset = TokenDataset(args.train_manifest)
    print(f"  验证集: {args.val_manifest}")
    val_dataset = TokenDataset(args.val_manifest)
    
    # 合并数据集
    all_items = []
    for item in train_dataset.items:
        item['split'] = 'train'
        all_items.append(item)
    for item in val_dataset.items:
        item['split'] = 'validation'
        all_items.append(item)
    
    print(f"\n总样本数: {len(all_items)}")
    
    # 评分
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    with open(output_path, 'w') as fout:
        for item in tqdm(all_items, desc="评分中"):
            ids = torch.tensor(item['ids'], dtype=torch.long)
            
            # 前向
            fwd_stats = compute_weighted_ppl(
                model, ids, device,
                weight_mode=args.weight_mode,
                epsilon=args.epsilon,
                topk_list=args.topk,
                win_size=args.win_size,
                win_stride=args.win_stride
            )
            
            # 反向
            ids_rev = ids.flip(0)
            rev_stats = compute_weighted_ppl(
                model, ids_rev, device,
                weight_mode=args.weight_mode,
                epsilon=args.epsilon,
                topk_list=args.topk,
                win_size=args.win_size,
                win_stride=args.win_stride
            )
            
            # 组织结果
            result = {
                'piece_id': item.get('piece_id', 'unknown'),
                'seg_idx': item.get('seg_idx', 0),
                'split': item['split'],
                'n_tokens': fwd_stats['n_tokens']
            }
            
            # 前向统计
            for k, v in fwd_stats.items():
                if k != 'n_tokens':
                    result[f'fwd_{k}'] = v
            
            # 反向统计
            for k, v in rev_stats.items():
                if k != 'n_tokens':
                    result[f'rev_{k}'] = v
            
            # TIS（加权版本）
            if fwd_stats['weighted_ppl'] > 0 and rev_stats['weighted_ppl'] > 0:
                result['tis_weighted'] = math.log(rev_stats['weighted_ppl']) - math.log(fwd_stats['weighted_ppl'])
            else:
                result['tis_weighted'] = 0.0
            
            # TIS（均匀版本，baseline）
            if fwd_stats['uniform_ppl'] > 0 and rev_stats['uniform_ppl'] > 0:
                result['tis_uniform'] = math.log(rev_stats['uniform_ppl']) - math.log(fwd_stats['uniform_ppl'])
            else:
                result['tis_uniform'] = 0.0
            
            # Top-k TIS
            for k in args.topk:
                fwd_topk = fwd_stats['topk_stats'].get(f'topk{k}', {})
                rev_topk = rev_stats['topk_stats'].get(f'topk{k}', {})
                
                if fwd_topk and rev_topk:
                    fwd_ppl = fwd_topk['ppl']
                    rev_ppl = rev_topk['ppl']
                    if fwd_ppl > 0 and rev_ppl > 0:
                        result[f'tis_topk{k}'] = math.log(rev_ppl) - math.log(fwd_ppl)
            
            fout.write(json.dumps(result) + '\n')
            results.append(result)
    
    print(f"\n✅ 评分完成！")
    print(f"   输出: {output_path}")
    print(f"   样本数: {len(results)}")
    
    # 快速统计
    if results:
        print(f"\n📊 快速统计:")
        tis_weighted = [r.get('tis_weighted', 0) for r in results]
        tis_uniform = [r.get('tis_uniform', 0) for r in results]
        print(f"  TIS (weighted): mean={np.mean(tis_weighted):.3f}, std={np.std(tis_weighted):.3f}")
        print(f"  TIS (uniform):  mean={np.mean(tis_uniform):.3f}, std={np.std(tis_uniform):.3f}")
        
        for k in args.topk:
            tis_topk = [r.get(f'tis_topk{k}', 0) for r in results if f'tis_topk{k}' in r]
            if tis_topk:
                print(f"  TIS (top-{k}):    mean={np.mean(tis_topk):.3f}, std={np.std(tis_topk):.3f}")
    
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ T52 完成！")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == '__main__':
    main()

