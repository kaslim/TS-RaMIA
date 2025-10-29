#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T50: TS-RaMIA Transformer 样本级打分
计算前向/反向困惑度和时序反转信号 (TIS)
"""

import argparse
import json
import math
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


class TokenDataset(Dataset):
    """Token JSONL 数据集"""
    
    def __init__(self, manifest_path, max_items_per_piece=None):
        self.manifest_path = manifest_path
        self.items = []
        
        # 读取所有样本
        with open(manifest_path) as f:
            for line_no, line in enumerate(f, 1):
                item = json.loads(line)
                item['line_no'] = line_no
                self.items.append(item)
        
        # 如果指定了每个作品的最大样本数，进行采样
        if max_items_per_piece is not None:
            # 这里简化处理：直接均匀采样
            import random
            random.seed(1337)
            if len(self.items) > max_items_per_piece * 100:  # 假设平均每作品有多个段
                self.items = random.sample(self.items, max_items_per_piece * 100)
        
        print(f"  加载了 {len(self.items)} 个样本")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        ids = torch.tensor(item['ids'], dtype=torch.long)
        return {
            'ids': ids,
            'line_no': item['line_no'],
            'seq_len': len(ids),
            'piece_id': item.get('piece_id', f'unknown_{idx}'),
            'seg_idx': item.get('seg_idx', 0)
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


def compute_perplexity(model, input_ids, attention_mask):
    """计算困惑度"""
    with torch.no_grad():
        # 使用 labels 触发语言模型损失计算
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        # outputs.loss 是平均 NLL (按 token)
        loss = outputs.loss.item()
        ppl = math.exp(loss)
        
        return ppl, loss


def main():
    parser = argparse.ArgumentParser(description="T50: TS-RaMIA Transformer 打分")
    parser.add_argument("--model_dir", required=True, help="模型目录")
    parser.add_argument("--manifest", required=True, help="Token JSONL 文件")
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"], 
                       help="数据集划分")
    parser.add_argument("--max_items_per_piece", type=int, default=None, 
                       help="每个作品最大样本数（用于快速评估）")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--seed", type=int, default=1337, help="随机种子")
    parser.add_argument("--output", default="results/protocol_randomized/transformer_scores.jsonl",
                       help="输出 JSONL 文件")
    args = parser.parse_args()
    
    print("=" * 70)
    print("T50: TS-RaMIA Transformer 样本级打分")
    print("=" * 70)
    print(f"模型目录: {args.model_dir}")
    print(f"数据清单: {args.manifest}")
    print(f"数据集划分: {args.split}")
    print(f"批次大小: {args.batch_size}")
    print(f"随机种子: {args.seed}")
    print(f"输出文件: {args.output}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    print(f"✓ 模型加载成功")
    
    # 获取模型版本信息
    model_rev = f"local:finetuned_maestro@{Path(args.model_dir).stat().st_mtime:.0f}"
    
    # 加载数据
    print(f"\n加载数据...")
    dataset = TokenDataset(args.manifest, args.max_items_per_piece)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    print(f"✓ 数据加载完成，共 {len(dataloader)} 个批次")
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 判断是否追加模式
    write_mode = 'a' if output_path.exists() else 'w'
    
    # 打分
    print(f"\n开始打分...")
    scores_written = 0
    
    with open(output_path, write_mode) as f:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="打分进度")):
            input_ids = batch['input_ids'].to(device)
            input_ids_rev = batch['input_ids_rev'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 计算前向困惑度
            ppl_fwd, nll_fwd = compute_perplexity(model, input_ids, attention_mask)
            
            # 计算反向困惑度
            ppl_rev, nll_rev = compute_perplexity(model, input_ids_rev, attention_mask)
            
            # 保存每个样本的结果
            for i, info in enumerate(batch['info']):
                # 计算 TIS
                tis = math.log(ppl_rev) - math.log(ppl_fwd)
                
                # 使用 piece_id 和 seg_idx
                piece_id = info.get('piece_id', f"{args.split}_{info['line_no']:06d}")
                seg_idx = info.get('seg_idx', 0)
                
                # 生成 UID（piece#seg格式）
                uid = f"{piece_id}#{seg_idx:04d}"
                
                # 生成路径引用
                path_tokens = f"{args.manifest}:#{info['line_no']}"
                
                # 生成 SHA256（可选，这里简化为基于 piece_id + seg_idx）
                sha256_str = hashlib.sha256(f"{piece_id}:{seg_idx}".encode()).hexdigest()
                
                # 构造记录
                record = {
                    "uid": uid,
                    "piece_id": piece_id,
                    "seg_idx": seg_idx,
                    "split": args.split,
                    "seed": args.seed,
                    "model_rev": model_rev,
                    "seq_len": info['seq_len'],
                    "ppl_fwd": ppl_fwd,
                    "ppl_rev": ppl_rev,
                    "tis": tis,
                    "path_tokens": path_tokens,
                    "sha256": sha256_str,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                
                # 写入 JSONL
                f.write(json.dumps(record) + "\n")
                scores_written += 1
    
    print(f"\n✓ 打分完成！")
    print(f"  写入样本数: {scores_written}")
    print(f"  输出文件: {args.output}")
    print(f"  平均 ppl_fwd: {ppl_fwd:.4f}")
    print(f"  平均 ppl_rev: {ppl_rev:.4f}")
    print(f"  平均 TIS: {tis:.4f}")


if __name__ == "__main__":
    main()

