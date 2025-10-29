# Quick Start Guide: TS-RaMIA

Get started with Membership Inference Attacks on music models in 15 minutes!

## 1. Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/kaslim/TS-RaMIA.git
cd TS-RaMIA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Your Data (3 minutes)

### Option A: Use Pre-computed Scores (Fastest)
If you have sample-level scores in CSV format:
```bash
# Copy your scores to: data/sample_scores.csv
# Required columns: sample_id, is_member, score1, score2, ...
```

### Option B: Compute Scores from Scratch
```bash
# 1. Download MAESTRO dataset
# https://magenta.tensorflow.org/datasets/maestro

# 2. Prepare data
python src/preprocessing/maestro_split.py \
    --maestro-dir /path/to/maestro \
    --output-dir ./data/processed

# 3. Convert MIDI to ABC
python notagen/data/2_data_preprocess.py \
    --input-dir ./data/processed/midi \
    --output-dir ./data/abc
```

## 3. Run Membership Inference Attack (5 minutes)

### Basic Attack (Top-64 Structural Tokens)
```bash
python scripts/score_tis_transformer.py \
    --model-dir ./models/transformer \
    --abc-dir ./data/abc \
    --split-json data/processed/split.json \
    --top-k 64 \
    --output-dir ./results
```

### Advanced Attack (Multi-Temperature Fusion)
```bash
# Step 1: Score with multiple temperatures
python scripts/B5_multi_temp_tail.py \
    --model-dir ./models/transformer \
    --abc-dir ./data/abc \
    --split-json data/processed/split.json \
    --temperatures 0.8,1.0,1.2,1.5 \
    --top-k-values 64 \
    --output-dir ./results

# Step 2: Aggregate and fuse
python scripts/B5_aggregate_fusion.py \
    --scores-dir ./results \
    --split-json data/processed/split.json \
    --fusion-method geometric_mean \
    --output-dir ./results/fusion
```

## 4. Evaluate Results (3 minutes)

### Aggregate to Piece-Level
```bash
python scripts/aggregate_piece_level_lenmatch.py \
    --sample-level ./results/fusion/sample_scores.csv \
    --split-json data/processed/split.json \
    --output-dir ./results
```

### Compute Metrics
```bash
python scripts/compute_low_fpr_metrics.py \
    --piece-level ./results/piece_level.csv \
    --output-dir ./results/metrics
```

### Generate ROC Curves
```bash
python scripts/plot_roc_academic.py \
    --piece-level ./results/piece_level.csv \
    --output-dir ./figures
```

## 5. Interpret Results (2 minutes)

Check the output files:
- `results/metrics/metrics.json` - AUC, TPR@k%FPR values
- `figures/roc_main_academic.png` - ROC curve visualization
- `results/piece_level.csv` - Per-piece predictions

Example output:
```
AUC: 0.925
TPR@1%FPR: 44.2%
TPR@5%FPR: 68.8%
```

## Common Issues & Solutions

### Issue: "Model not found"
**Solution**: Ensure transformer model is at `./models/transformer`. Download from HuggingFace if needed:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("gpt2")
model.save_pretrained("./models/transformer")
```

### Issue: "Out of memory"
**Solution**: Reduce batch size or process in smaller chunks:
```bash
python scripts/B5_multi_temp_tail.py \
    --batch-size 16 \  # Reduce from default 32
    # ... other args
```

### Issue: "ABC files not found"
**Solution**: Verify ABC conversion:
```bash
ls -la data/abc/
# Should contain .abc files
```

## Next Steps

1. **Read the full README** for detailed documentation
2. **Try custom configurations** in `configs/note_token_ids.json`
3. **Implement custom scoring** - see `scripts/score_tis_transformer.py` for template
4. **Compare multiple methods** - run all scoring variants and compare

## Performance Expectations

| Task | Time | Output |
|------|------|--------|
| Score 1000 samples | 15 sec | sample_scores.csv |
| Aggregate to pieces | 1 min | piece_level.csv |
| Compute metrics | 30 sec | metrics.json |
| Generate plots | 1 min | roc_curves.png |

## Advanced: Running Your Own Models

### Step 1: Train a Transformer
```bash
python src/train_transformer.py \
    --train-manifest data/processed/train.jsonl \
    --val-manifest data/processed/val.jsonl \
    --output-dir ./models/my_model \
    --epochs 10
```

### Step 2: Use Your Model for Attacks
```bash
python scripts/score_tis_transformer.py \
    --model-dir ./models/my_model \
    # ... rest of args
```

## Citation

If you use this code, please cite:
```bibtex
@article{ts-ramia2025,
  title={TS-RaMIA: Membership Inference Attacks on Music Models via Transcription Structure},
  author={Your Name},
  year={2025}
}
```

## Support

- **Documentation**: See README.md
- **Issues**: Open an issue on GitHub
- **Questions**: Check existing issues first

---

**Happy attacking!** 🎵🔓
