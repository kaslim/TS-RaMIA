#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer 模型微调脚本
在 MAESTRO 数据集上微调预训练的音乐生成模型
"""

import argparse
import json
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datetime import datetime


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JSONLDataset(Dataset):
    """从 JSONL 文件加载数据的 Dataset (直接读取 token IDs)"""
    
    def __init__(self, jsonl_path, max_len):
        self.items = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                self.items.append(item["ids"])
        self.max_len = max_len
        print(f"  加载 {len(self.items)} 个训练片段")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        # 直接使用 token IDs
        ids = torch.tensor(self.items[idx][:self.max_len], dtype=torch.long)
        
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids)
        }


class NPZManifestDS(Dataset):
    """从 NPZ 文件清单加载数据的 Dataset"""
    
    def __init__(self, manifest, max_len):
        self.items = [json.loads(l) for l in open(manifest)]
        self.max_len = max_len
        print(f"  加载 {len(self.items)} 个训练片段")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        # 加载 npz 文件
        arr = np.load(self.items[idx]["npz"])["ids"].astype(np.int64)
        # 截断到最大长度（保险起见，tokenize 阶段应该已经处理）
        ids = torch.from_numpy(arr[:self.max_len])
        
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids)
        }


def collate(batch):
    """批次整理函数：左对齐 padding 到批次最长"""
    maxL = max(x["input_ids"].shape[0] for x in batch)
    
    def pad(t):
        return torch.nn.functional.pad(t, (0, maxL - t.shape[0]), value=0)
    
    input_ids = torch.stack([pad(x["input_ids"]) for x in batch])
    labels = torch.stack([pad(x["labels"]) for x in batch])
    attn = torch.stack([pad(x["attention_mask"]) for x in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn
    }


def main():
    ap = argparse.ArgumentParser(description="Transformer 模型微调")
    ap.add_argument("--base_model", required=True, help="基础模型路径或名称")
    ap.add_argument("--train_manifest", required=True, help="训练集清单")
    ap.add_argument("--val_manifest", required=True, help="验证集清单")
    ap.add_argument("--output_dir", required=True, help="输出目录")
    ap.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    ap.add_argument("--batch_size", type=int, default=8, help="批次大小")
    ap.add_argument("--epochs", type=int, default=2, help="训练轮数")
    ap.add_argument("--lr", type=float, default=5e-5, help="学习率")
    ap.add_argument("--seed", type=int, default=1337, help="随机种子")
    ap.add_argument("--grad_ckpt", type=int, default=1, help="是否使用梯度检查点")
    args = ap.parse_args()
    
    print("=" * 70)
    print("Transformer 模型微调")
    print("=" * 70)
    print(f"基础模型: {args.base_model}")
    print(f"训练清单: {args.train_manifest}")
    print(f"验证清单: {args.val_manifest}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大长度: {args.max_length}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"随机种子: {args.seed}")
    print(f"梯度检查点: {'是' if args.grad_ckpt else '否'}")
    print(f"开始时间: {datetime.now().isoformat()}")
    
    # 设置随机种子
    print("\n设置随机种子...")
    set_seed(args.seed)
    print(f"✓ 随机种子设置为 {args.seed}")
    
    # 加载模型
    print(f"\n加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    print(f"✓ 模型加载成功")
    
    # 检查词汇表大小
    vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"  词汇表大小: {vocab_size}")
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查 is_dummy 门禁
    print("\n检查 tokenization 统计 (强门禁)...")
    stats_path = "reports/tokenize_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        is_dummy = stats.get("is_dummy", True)
        print(f"  is_dummy: {is_dummy}")
        if is_dummy:
            print("\n" + "=" * 70)
            print("✗ 门禁拒绝：检测到模拟数据 (is_dummy=True)")
            print("=" * 70)
            print("T40 要求真实 tokenization 产物。")
            print("请先完成 T35 真实 tokenization。")
            print("=" * 70)
            raise ValueError("Training blocked: is_dummy=True")
        else:
            print("  ✓ 门禁放行：真实数据 (is_dummy=False)")
    else:
        print("  ⚠️  未找到统计文件，假设为真实数据")
    
    # 加载数据集 - 使用 JSONLDataset
    print(f"\n加载数据集...")
    print("训练集:")
    train_ds = JSONLDataset(args.train_manifest, args.max_length)
    print("验证集:")
    val_ds = JSONLDataset(args.val_manifest, args.max_length)
    
    # 配置训练参数
    print(f"\n配置训练参数...")
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",  # 新版本使用 eval_strategy
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        gradient_checkpointing=bool(args.grad_ckpt),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )
    
    # 创建 Trainer
    print(f"\n创建 Trainer...")
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate
    )
    
    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    trainer.train()
    
    # 保存模型
    print(f"\n保存微调后的模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    print("\n" + "=" * 70)
    print("✓ 训练完成!")
    print("=" * 70)
    print(f"完成时间: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

