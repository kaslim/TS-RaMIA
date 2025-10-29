#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAESTRO 数据集官方分割固化
按照 CSV 中的 split 列生成 train/validation/test 分割
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib


def compute_file_hash(file_path):
    """计算文件的 SHA256 哈希"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MAESTRO 数据集分割固化")
    parser.add_argument("--csv_path", type=str, required=True, help="MAESTRO CSV 文件路径")
    parser.add_argument("--root", type=str, required=True, help="MAESTRO 数据根目录")
    parser.add_argument("--output_path", type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument("--stats_path", type=str, required=True, help="统计信息输出路径")
    parser.add_argument("--require_midi", type=int, default=1, help="是否要求 MIDI 文件存在")
    parser.add_argument("--require_audio", type=int, default=0, help="是否要求音频文件存在")
    return parser.parse_args()


def validate_file_exists(root, filename, file_type="MIDI"):
    """验证文件是否存在"""
    file_path = Path(root) / filename
    if not file_path.exists():
        return False, f"{file_type} 文件不存在: {filename}"
    return True, None


def process_maestro_split(csv_path, root, require_midi=True, require_audio=False):
    """
    处理 MAESTRO 数据集分割
    
    Args:
        csv_path: CSV 文件路径
        root: 数据根目录
        require_midi: 是否要求 MIDI 文件存在
        require_audio: 是否要求音频文件存在
    
    Returns:
        dict: 包含 train/validation/test 分割的字典
        dict: 统计信息
    """
    print("=" * 70)
    print("MAESTRO 数据集官方分割固化")
    print("=" * 70)
    print(f"CSV 路径: {csv_path}")
    print(f"数据根目录: {root}")
    print(f"开始时间: {datetime.now().isoformat()}")
    print()
    
    # 读取 CSV
    print("[1/4] 读取 CSV 文件...")
    df = pd.read_csv(csv_path)
    print(f"✓ 读取成功，共 {len(df)} 行")
    print(f"  列: {list(df.columns)}")
    
    # 检查必需的列
    required_cols = ["split", "midi_filename"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必需的列: {missing_cols}")
    
    # 移除没有 split 标注的行
    df = df.dropna(subset=["split"])
    print(f"  有效行数（有 split 标注）: {len(df)}")
    
    # 初始化分割字典
    splits = {
        "train": [],
        "validation": [],
        "test": []
    }
    
    # 统计信息
    stats = {
        "total_files": len(df),
        "splits": {},
        "skipped": {"midi": 0, "audio": 0},
        "timestamp": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "csv_sha256": compute_file_hash(csv_path)
    }
    
    print("\n[2/4] 处理各个分割...")
    for split_name in ["train", "validation", "test"]:
        df_split = df[df["split"] == split_name]
        print(f"\n  处理 {split_name}...")
        print(f"    CSV 中的数量: {len(df_split)}")
        
        processed = 0
        skipped = 0
        
        for idx, row in df_split.iterrows():
            midi_filename = str(row["midi_filename"])
            
            # 验证 MIDI 文件
            if require_midi:
                exists, error = validate_file_exists(root, midi_filename, "MIDI")
                if not exists:
                    print(f"    ⚠️  跳过: {midi_filename} - {error}")
                    skipped += 1
                    stats["skipped"]["midi"] += 1
                    continue
            
            # 验证音频文件（如果需要）
            if require_audio and "audio_filename" in df.columns:
                audio_filename = str(row["audio_filename"])
                exists, error = validate_file_exists(root, audio_filename, "Audio")
                if not exists:
                    print(f"    ⚠️  跳过: {midi_filename} - {error}")
                    skipped += 1
                    stats["skipped"]["audio"] += 1
                    continue
            
            # 创建条目
            entry = {
                "uid": midi_filename,  # 使用 MIDI 相对路径作为 UID
                "split": split_name,
                "midi_filename": midi_filename,
            }
            
            # 添加可选字段
            if "audio_filename" in df.columns and pd.notna(row["audio_filename"]):
                entry["audio_filename"] = str(row["audio_filename"])
            if "canonical_composer" in df.columns and pd.notna(row["canonical_composer"]):
                entry["composer"] = str(row["canonical_composer"])
            if "canonical_title" in df.columns and pd.notna(row["canonical_title"]):
                entry["title"] = str(row["canonical_title"])
            if "year" in df.columns and pd.notna(row["year"]):
                entry["year"] = int(row["year"])
            if "duration" in df.columns and pd.notna(row["duration"]):
                entry["duration"] = float(row["duration"])
            
            splits[split_name].append(entry)
            processed += 1
        
        print(f"    ✓ 处理完成: {processed} 个文件")
        if skipped > 0:
            print(f"    ⚠️  跳过: {skipped} 个文件")
        
        stats["splits"][split_name] = {
            "count": len(splits[split_name]),
            "processed": processed,
            "skipped": skipped
        }
    
    # 验证互斥性
    print("\n[3/4] 验证分割互斥性...")
    train_uids = set(e["uid"] for e in splits["train"])
    val_uids = set(e["uid"] for e in splits["validation"])
    test_uids = set(e["uid"] for e in splits["test"])
    
    assert train_uids.isdisjoint(val_uids), "Train 和 Validation 有重叠"
    assert train_uids.isdisjoint(test_uids), "Train 和 Test 有重叠"
    assert val_uids.isdisjoint(test_uids), "Validation 和 Test 有重叠"
    print("  ✓ 所有分割互斥")
    
    # 计算统计信息
    total_processed = sum(len(splits[k]) for k in splits)
    stats["total_processed"] = total_processed
    stats["total_skipped"] = stats["total_files"] - total_processed
    
    print("\n[4/4] 生成统计摘要...")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  处理成功: {stats['total_processed']}")
    print(f"  跳过: {stats['total_skipped']}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Validation: {len(splits['validation'])}")
    print(f"  Test: {len(splits['test'])}")
    
    return splits, stats


def main():
    """主函数"""
    args = parse_args()
    
    # 验证 CSV 文件存在
    if not Path(args.csv_path).exists():
        raise FileNotFoundError(f"CSV 文件不存在: {args.csv_path}")
    
    # 验证根目录存在
    if not Path(args.root).exists():
        raise FileNotFoundError(f"根目录不存在: {args.root}")
    
    # 处理分割
    splits, stats = process_maestro_split(
        csv_path=args.csv_path,
        root=args.root,
        require_midi=bool(args.require_midi),
        require_audio=bool(args.require_audio)
    )
    
    # 保存分割文件
    print(f"\n保存分割文件到: {args.output_path}")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    print("✓ 分割文件已保存")
    
    # 保存统计信息
    print(f"\n保存统计信息到: {args.stats_path}")
    Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("✓ 统计信息已保存")
    
    print("\n" + "=" * 70)
    print("✓ MAESTRO 数据集分割固化完成")
    print("=" * 70)
    print(f"完成时间: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

