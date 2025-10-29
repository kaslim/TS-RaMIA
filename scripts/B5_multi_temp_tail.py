#!/usr/bin/env python3
"""
B5: 多温度尾部融合
对不同 k 和温度 T 的组合计算 tail-k PPL，然后融合
k ∈ {32, 48, 64, 96, 128}; T ∈ {0.7, 1.0, 1.3}
验收: TPR@1%FPR ≥ 4%, AUC ≥ 0.74
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import math

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path) as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 兼容不同字段名
        token_ids = item.get('token_ids') or item.get('ids')
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'piece_id': item['piece_id'],
            'seg_idx': item.get('seg_idx', 0)
        }

def collate_fn(batch):
    max_len = max([item['input_ids'].size(0) for item in batch])
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        length = item['input_ids'].size(0)
        input_ids[i, :length] = item['input_ids']
        attention_mask[i, :length] = 1
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'piece_ids': [item['piece_id'] for item in batch],
        'seg_indices': [item['seg_idx'] for item in batch]
    }

def compute_multi_temp_tail(model, input_ids, attention_mask, device, k_values, temperatures):
    """
    计算多个 k 和温度组合的 tail-k PPL
    
    Returns:
        scores: dict with keys like "tail32_t0.7", "tail64_t1.0", etc.
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous().to(device)
        shift_mask = attention_mask[:, 1:].contiguous().to(device)
        
        scores = {}
        
        # 对每个温度
        for temp in temperatures:
            # 应用温度缩放
            if temp != 1.0:
                scaled_logits = shift_logits / temp
            else:
                scaled_logits = shift_logits
            
            # 计算 token-level NLL
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_nlls = loss_fct(
                scaled_logits.view(-1, scaled_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            # 只保留有效 token
            valid_nlls = token_nlls[shift_mask.bool()].cpu().numpy()
            
            if len(valid_nlls) == 0:
                # 无有效 token
                for k in k_values:
                    scores[f'tail{k}_t{temp}'] = float('inf')
                continue
            
            # 对每个 k 值计算 tail-k
            for k in k_values:
                if len(valid_nlls) >= k:
                    topk_nlls = np.sort(valid_nlls)[-k:]
                    topk_mean = topk_nlls.mean()
                    topk_ppl = math.exp(topk_mean)
                else:
                    # 不足 k 个，使用全部
                    topk_ppl = math.exp(valid_nlls.mean())
                
                scores[f'tail{k}_t{temp}'] = float(topk_ppl)
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='B5: 多温度尾部融合')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--train_manifest', required=True, help='训练集 JSONL')
    parser.add_argument('--val_manifest', required=True, help='验证集 JSONL')
    parser.add_argument('--output_jsonl', required=True, help='输出 JSONL')
    parser.add_argument('--k_values', nargs='+', type=int, default=[32, 48, 64, 96, 128], help='k 值列表')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.7, 1.0, 1.3], help='温度列表')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--seed', type=int, default=1337, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    # 合并数据集
    print("加载数据集...")
    train_dataset = JSONLDataset(args.train_manifest)
    val_dataset = JSONLDataset(args.val_manifest)
    
    combined_data = list(train_dataset.data) + list(val_dataset.data)
    print(f"  总样本数: {len(combined_data)}")
    
    # 创建 DataLoader
    class CombinedDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            # 兼容不同字段名
            token_ids = item.get('token_ids') or item.get('ids')
            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'piece_id': item['piece_id'],
                'seg_idx': item.get('seg_idx', 0)
            }
    
    dataset = CombinedDataset(combined_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 评分
    print(f"\n开始评分: k={args.k_values}, T={args.temperatures}")
    print(f"  共 {len(args.k_values)} × {len(args.temperatures)} = {len(args.k_values) * len(args.temperatures)} 种组合")
    
    results = []
    
    with open(args.output_jsonl, 'w') as outf:
        for batch in tqdm(dataloader, desc="评分中"):
            batch_scores = compute_multi_temp_tail(
                model,
                batch['input_ids'],
                batch['attention_mask'],
                device,
                args.k_values,
                args.temperatures
            )
            
            # 为批次中的每个样本写入结果
            for i in range(len(batch['piece_ids'])):
                record = {
                    'piece_id': batch['piece_ids'][i],
                    'seg_idx': batch['seg_indices'][i],
                    **{k: v if isinstance(v, float) else v[i] if isinstance(v, (list, np.ndarray)) else v 
                       for k, v in batch_scores.items()}
                }
                outf.write(json.dumps(record) + '\n')
                results.append(record)
    
    print(f"\n✅ 完成！输出: {args.output_jsonl}")
    print(f"   共 {len(results)} 条记录")
    
    # 显示示例
    if results:
        print("\n示例记录:")
        print(json.dumps(results[0], indent=2))

if __name__ == "__main__":
    main()

