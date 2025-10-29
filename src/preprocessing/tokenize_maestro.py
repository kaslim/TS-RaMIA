#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T35: MAESTRO Tokenization - 使用 MidiTok 3.x + Symusic 0.5.8
论文级 REMI (无 BPE)
"""

import json
import argparse
import os
import sys
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score

def chunk_ids(ids, max_len=1024, stride=1024):
    """固定长度切片，避免 Trainer 内部 pad/对齐问题"""
    for i in range(0, len(ids) - max_len + 1, stride):
        yield ids[i:i+max_len]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--stats_path", required=True)
    args = ap.parse_args()

    splits = json.load(open(args.split_json))
    
    # 论文级 REMI 配置（无 BPE）
    cfg = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=False,
        use_programs=False
    )
    tok = REMI(cfg)
    
    print(f"✓ Tokenizer 创建成功")
    print(f"  词汇表大小: {len(tok)}")
    print(f"  配置: REMI 无 BPE")
    
    stats = {
        "is_dummy": False,
        "vocab_tag": "REMI-noBPE",
        "vocab_size": len(tok),
        "train_segments": 0,
        "val_segments": 0,
        "train_skipped": 0,
        "val_skipped": 0,
        "max_length": args.max_len,
        "stride": args.stride,
        "train_skip_reasons": {},
        "val_skip_reasons": {},
        "miditok_version": "3.0.6.post1",
        "symusic_version": "0.5.8"
    }

    def process(split_items, out_path, is_train):
        n = 0
        skipped_count = 0
        skip_reasons = {}
        
        os.makedirs(Path(out_path).parent, exist_ok=True)
        
        with open(out_path, "w") as fout:
            for idx, it in enumerate(split_items):
                midi_rel = it.get("midi_filename", it.get("midi_path", ""))
                midi_path = os.path.join(args.dataset_root, midi_rel)
                
                if not os.path.exists(midi_path):
                    skipped_count += 1
                    reason = "file_not_found"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue

                try:
                    # 使用 tick 时间模式 - 关键！
                    score = Score(midi_path, ttype="tick")
                    
                    # 检查空乐谱
                    if not score.tracks or sum(len(tr.notes) for tr in score.tracks) == 0:
                        skipped_count += 1
                        reason = "empty_or_no_notes"
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        continue
                    
                    # 编码 - MidiTok 3.x 返回列表
                    ts_list = tok(score)
                    
                    if not ts_list or len(ts_list) == 0:
                        skipped_count += 1
                        reason = "no_tokens_generated"
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        continue
                    
                    # 取第一个 TokSequence
                    first_ts = ts_list[0]
                    ids = first_ts.ids if hasattr(first_ts, 'ids') else first_ts
                    
                    if not ids or len(ids) < args.max_len:
                        skipped_count += 1
                        reason = "too_short"
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        continue
                    
                    # 固定长度切片
                    # 添加 piece_id（使用 midi_filename）
                    piece_id = it.get("midi_filename", it.get("uid", f"unknown_{idx}"))
                    for seg_idx, seg in enumerate(chunk_ids(ids, args.max_len, args.stride)):
                        fout.write(json.dumps({
                            "ids": list(seg),
                            "piece_id": piece_id,
                            "seg_idx": seg_idx
                        }) + "\n")
                        n += 1
                    
                    # 进度提示
                    if (idx + 1) % 100 == 0:
                        split_name = "train" if is_train else "val"
                        print(f"  [{split_name}] 处理 {idx + 1}/{len(split_items)}, 段数: {n}, 跳过: {skipped_count}")
                        
                except Exception as e:
                    skipped_count += 1
                    k = f"exc:{type(e).__name__}"
                    skip_reasons[k] = skip_reasons.get(k, 0) + 1
                    
        return n, skipped_count, skip_reasons

    print(f"\n处理训练集 ({len(splits['train'])} 个文件)...")
    tr_n, tr_skipped, tr_reasons = process(splits["train"], args.out_train, True)
    
    print(f"\n处理验证集 ({len(splits['validation'])} 个文件)...")
    va_n, va_skipped, va_reasons = process(splits["validation"], args.out_val, False)
    
    stats["train_segments"] = tr_n
    stats["val_segments"] = va_n
    stats["train_skipped"] = tr_skipped
    stats["val_skipped"] = va_skipped
    stats["train_skip_reasons"] = tr_reasons
    stats["val_skip_reasons"] = va_reasons

    os.makedirs("reports", exist_ok=True)
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Tokenization 完成!")
    print("=" * 70)
    print(f"训练段数: {tr_n}")
    print(f"验证段数: {va_n}")
    print(f"训练跳过: {tr_skipped} ({tr_reasons})")
    print(f"验证跳过: {va_skipped} ({va_reasons})")
    print(f"词汇表: {len(tok)} tokens (REMI-noBPE)")
    print(f"is_dummy: False ✓")
    print(f"统计文件: {args.stats_path}")

if __name__ == "__main__":
    main()
