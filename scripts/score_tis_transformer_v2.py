#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T49: 样本级 PPL/NLL 评估（修复版）
修复 batch-level PPL 的致命问题，改为逐样本精确计算
"""

import argparse
import json
import math
import hashlib
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM


class TokenDataset(Dataset):
    """从 JSONL 加载 token 数据"""
    def __init__(self, jsonl_path):
        self.items = []
        with open(jsonl_path) as f:
            for line_no, line in enumerate(f, 1):
                item = json.loads(line)
                item['line_no'] = line_no
                self.items.append(item)
        print(f"  加载了 {len(self.items)} 个样本")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        ids = torch.tensor(item['ids'], dtype=torch.long)
        return {
            'ids': ids,
            'line_no': item['line_no'],
            'piece_id': item.get('piece_id', f'unknown_{idx}'),
            'seg_idx': item.get('seg_idx', 0),
            'seq_len': len(ids)
        }


def collate_fn(batch):
    """批次整理：填充到相同长度"""
    max_len = max(item['seq_len'] for item in batch)
    
    batch_ids = []
    batch_ids_rev = []
    batch_masks = []
    batch_info = []
    
    for item in batch:
        ids = item['ids']
        seq_len = item['seq_len']
        
        # 填充
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, dtype=torch.long)
            ids_padded = torch.cat([ids, padding])
            mask = torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)])
        else:
            ids_padded = ids
            mask = torch.ones(seq_len)
        
        # 反转序列（保持填充在右侧）
        ids_rev = ids.flip(0)
        if seq_len < max_len:
            ids_rev_padded = torch.cat([ids_rev, padding])
        else:
            ids_rev_padded = ids_rev
        
        batch_ids.append(ids_padded)
        batch_ids_rev.append(ids_rev_padded)
        batch_masks.append(mask)
        batch_info.append({
            'line_no': item['line_no'],
            'seq_len': seq_len,
            'piece_id': item['piece_id'],
            'seg_idx': item['seg_idx']
        })
    
    return {
        'input_ids': torch.stack(batch_ids),
        'input_ids_rev': torch.stack(batch_ids_rev),
        'attention_mask': torch.stack(batch_masks),
        'info': batch_info
    }


def compute_perplexity_per_sample(model, input_ids, attention_mask, device):
    """
    样本级精确 PPL 计算（修复版）
    返回每个样本的 (ppl, nll, n_tokens)
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab]
        
        # Shift for causal LM: 预测 token_i 基于 token_<i
        shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
        shift_labels = input_ids[..., 1:].contiguous()   # [batch, seq_len-1]
        shift_mask = attention_mask[..., 1:].contiguous() # [batch, seq_len-1]
        
        # 计算每个 token 的 NLL
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1)
        )
        losses = losses.view(input_ids.size(0), -1)  # [batch, seq_len-1]
        
        # 每个样本独立计算
        sample_results = []
        for i in range(input_ids.size(0)):
            mask = shift_mask[i]
            valid_losses = losses[i][mask == 1]
            
            if len(valid_losses) == 0:
                # 空序列（理论上不应该出现）
                nll = float('inf')
                ppl = float('inf')
                n_tokens = 0
            else:
                nll = valid_losses.mean().item()
                ppl = math.exp(nll)
                n_tokens = len(valid_losses)
            
            sample_results.append({
                'ppl': ppl,
                'nll': nll,
                'n_tokens': n_tokens
            })
        
        return sample_results


def main():
    parser = argparse.ArgumentParser(description="T49: 样本级 PPL/NLL 评估（修复版）")
    parser.add_argument("--model_dir", required=True, help="模型目录")
    parser.add_argument("--manifest", required=True, help="训练集 JSONL")
    parser.add_argument("--manifest_val", required=True, help="验证集 JSONL")
    parser.add_argument("--out", required=True, help="输出 JSONL 文件")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--seed", type=int, default=1337, help="随机种子")
    args = parser.parse_args()
    
    print("=" * 70)
    print("T49: 样本级 PPL/NLL 评估（修复版）")
    print("=" * 70)
    print(f"模型目录: {args.model_dir}")
    print(f"训练集: {args.manifest}")
    print(f"验证集: {args.manifest_val}")
    print(f"输出文件: {args.out}")
    print(f"批次大小: {args.batch_size}")
    print(f"随机种子: {args.seed}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    print(f"✓ 模型加载成功")
    
    # 获取模型版本
    model_rev = f"local:finetuned_maestro@{Path(args.model_dir).stat().st_mtime:.0f}"
    
    # 创建输出目录
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理训练集和验证集
    manifests = [
        (args.manifest, "train"),
        (args.manifest_val, "validation")
    ]
    
    all_scores = []
    
    for manifest_path, split_name in manifests:
        print(f"\n处理 {split_name} 集...")
        dataset = TokenDataset(manifest_path)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        print(f"  共 {len(dataloader)} 个批次")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"打分 {split_name}")):
            input_ids = batch['input_ids'].to(device)
            input_ids_rev = batch['input_ids_rev'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 计算前向 PPL（逐样本）
            results_fwd = compute_perplexity_per_sample(model, input_ids, attention_mask, device)
            
            # 计算反向 PPL（逐样本）
            results_rev = compute_perplexity_per_sample(model, input_ids_rev, attention_mask, device)
            
            # 保存每个样本的结果
            for i, info in enumerate(batch['info']):
                ppl_fwd = results_fwd[i]['ppl']
                nll_fwd = results_fwd[i]['nll']
                n_tokens_fwd = results_fwd[i]['n_tokens']
                
                ppl_rev = results_rev[i]['ppl']
                nll_rev = results_rev[i]['nll']
                n_tokens_rev = results_rev[i]['n_tokens']
                
                # 计算 TIS
                if ppl_fwd > 0 and ppl_rev > 0 and math.isfinite(ppl_fwd) and math.isfinite(ppl_rev):
                    tis = math.log(ppl_rev) - math.log(ppl_fwd)
                else:
                    tis = 0.0
                
                # 使用 piece_id
                piece_id = info['piece_id']
                seg_idx = info['seg_idx']
                uid = f"{piece_id}#{seg_idx:04d}"
                
                # SHA256
                sha256_str = hashlib.sha256(f"{piece_id}:{seg_idx}".encode()).hexdigest()
                
                # 构造记录（新增字段：nll_fwd, nll_rev, n_tokens）
                record = {
                    "uid": uid,
                    "piece_id": piece_id,
                    "seg_idx": seg_idx,
                    "split": split_name,
                    "seed": args.seed,
                    "model_rev": model_rev,
                    "seq_len": info['seq_len'],
                    "n_tokens": n_tokens_fwd,  # 新字段：实际 token 数
                    "ppl_fwd": ppl_fwd,
                    "ppl_rev": ppl_rev,
                    "nll_fwd": nll_fwd,        # 新字段：前向 NLL
                    "nll_rev": nll_rev,        # 新字段：反向 NLL
                    "tis": tis,
                    "path_tokens": f"{manifest_path}:#{info['line_no']}",
                    "sha256": sha256_str,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                
                all_scores.append(record)
    
    # 写入所有结果
    print(f"\n写入结果...")
    with open(output_path, 'w') as f:
        for record in all_scores:
            f.write(json.dumps(record) + "\n")
    
    print(f"\n✓ 打分完成！")
    print(f"  总样本数: {len(all_scores)}")
    print(f"  输出文件: {args.out}")
    
    # 快速统计
    train_scores = [r for r in all_scores if r['split'] == 'train']
    val_scores = [r for r in all_scores if r['split'] == 'validation']
    
    if train_scores:
        avg_ppl_fwd_train = sum(r['ppl_fwd'] for r in train_scores if math.isfinite(r['ppl_fwd'])) / len(train_scores)
        avg_tis_train = sum(r['tis'] for r in train_scores if math.isfinite(r['tis'])) / len(train_scores)
        print(f"\n训练集统计:")
        print(f"  样本数: {len(train_scores)}")
        print(f"  平均 PPL_fwd: {avg_ppl_fwd_train:.4f}")
        print(f"  平均 TIS: {avg_tis_train:.4f}")
    
    if val_scores:
        avg_ppl_fwd_val = sum(r['ppl_fwd'] for r in val_scores if math.isfinite(r['ppl_fwd'])) / len(val_scores)
        avg_tis_val = sum(r['tis'] for r in val_scores if math.isfinite(r['tis'])) / len(val_scores)
        print(f"\n验证集统计:")
        print(f"  样本数: {len(val_scores)}")
        print(f"  平均 PPL_fwd: {avg_ppl_fwd_val:.4f}")
        print(f"  平均 TIS: {avg_tis_val:.4f}")
    
    print(f"\n{'=' * 70}")
    print("✓ T49 任务完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

