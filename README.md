# TS-RaMIA: Membership Inference Attacks for Symbolic Music Generation Models

A comprehensive framework for membership inference attacks (MIA) on music generation models, combining transcription structure analysis with advanced scoring and fusion techniques.

## Overview

TS-RaMIA presents a novel approach to membership inference attacks on music models by:
- **Structural Token Analysis**: Leveraging ABC notation structure tokens as natural leak channels
- **Top-k Tail Scoring**: Advanced statistical methods to detect membership signals in the tail distributions
- **Meta-Fusion Approach**: Combining multiple scoring dimensions through meta-learning

### Key Results
- **Main Attack (StructTail+Fusion)**: AUC = 0.925 with TPR = 44.2% @ 1% FPR
- **Baseline (mean NLL)**: AUC = 0.679
- **StructTail-64**: AUC = 0.794

## Features

### 1. Data Processing & Tokenization
- MAESTRO dataset handling and train/val/test splitting
- ABC notation to XML conversion and vice versa
- Token-level structure masking for privacy analysis

### 2. Scoring Modules

#### TIS (Token Importance Score)
- `score_tis_transformer.py`: Base transformer-based scoring
- `score_tis_transformer_v2.py`: Enhanced version with multi-view evaluation
- `score_tis_weighted_tail.py`: Weighted tail distribution analysis
- `score_tis_transformer_windowed.py`: Windowed aggregation for robustness

#### Multi-Temperature Tail Fusion
- `B5_multi_temp_tail.py`: Multi-temperature top-k scoring
- `B5_aggregate_fusion.py`: Flexible aggregation and fusion strategies
- Supports max, mean, and geometric mean fusion methods

#### EVT-based Approaches
- `B6_evt_tail_prob.py`: Extreme Value Theory for tail probability estimation

#### Meta-Learning Fusion
- `meta_attack_cv.py`: Cross-validation meta-attacker combining multiple signals

### 3. Debiasing & Calibration
- `calibrate_scores.py`: Conditional calibration for score normalization
- `aggregate_piece_level_lenmatch.py`: Length-matched aggregation with bias correction
- Debiasing pipelines for distributional shift handling

### 4. Evaluation Tools
- `compute_low_fpr_metrics.py`: Low FPR region analysis (TPR@k%FPR)
- `auc_delong.py`: DeLong confidence intervals for AUC
- `plot_roc_academic.py`: Publication-quality ROC curve generation

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- scikit-learn, pandas, numpy
- music21 (for ABC/XML conversion)

### Setup

```bash
# Clone the repository
git clone https://github.com/kaslim/TS-RaMIA.git
cd TS-RaMIA

# Create environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

```bash
# Prepare MAESTRO dataset
python src/preprocessing/maestro_split.py --maestro-dir /path/to/maestro \
                                           --output-dir ./data/processed

# Convert MIDI to ABC (requires music21)
python NotaGen/data/2_data_preprocess.py --input-dir /path/to/midi \
                                         --output-dir ./data/abc
```

### 2. Tokenization

```bash
# Tokenize ABC files using miditok
python src/preprocessing/tokenize_maestro.py --abc-dir ./data/abc \
                                              --output-dir ./data/tokens
```

### 3. Model Training (Optional)

```bash
# Fine-tune transformer on MAESTRO
python src/train_transformer.py \
    --train-manifest data/processed/train.jsonl \
    --val-manifest data/processed/val.jsonl \
    --output-dir ./models/transformer \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 2e-4
```

### 4. Membership Inference Attack

#### Basic TIS Scoring (StructTail-64)
```bash
python scripts/score_tis_transformer.py \
    --model-dir ./models/transformer \
    --abc-dir ./data/abc \
    --split-json data/processed/split.json \
    --top-k 64 \
    --output-dir ./results
```

#### Advanced Multi-Temperature Fusion (StructTail+Fusion)
```bash
python scripts/B5_multi_temp_tail.py \
    --model-dir ./models/transformer \
    --abc-dir ./data/abc \
    --split-json data/processed/split.json \
    --temperatures 0.8,1.0,1.2,1.5 \
    --top-k-values 32,64,96,128 \
    --output-dir ./results

python scripts/B5_aggregate_fusion.py \
    --scores-dir ./results \
    --split-json data/processed/split.json \
    --output-dir ./results/aggregated
```

#### Meta-Fusion Approach
```bash
python scripts/meta_attack_cv.py \
    --raw-scores results/raw_scores.csv \
    --split-json data/processed/split.json \
    --output-dir ./results/meta_fusion
```

### 5. Debiasing & Evaluation

```bash
# Aggregate to piece-level with length matching
python scripts/aggregate_piece_level_lenmatch.py \
    --sample-level results/sample_scores.csv \
    --split-json data/processed/split.json \
    --output-dir ./results

# Compute debiased metrics
python scripts/compute_low_fpr_metrics.py \
    --piece-level results/piece_level.csv \
    --output-dir ./results/metrics

# Generate ROC curves
python scripts/plot_roc_academic.py \
    --piece-level results/piece_level.csv \
    --output-dir ./figures
```

## Architecture

### Directory Structure
```
TS-RaMIA/
├── src/
│   ├── preprocessing/
│   │   ├── maestro_split.py          # MAESTRO dataset handling
│   │   ├── tokenize_maestro.py       # Tokenization pipeline
│   │   └── abc_structure_utils.py    # ABC structure masking
│   └── train_transformer.py           # Model training
├── scripts/
│   ├── score_tis_*.py                # TIS scoring variants
│   ├── B5_*.py                       # Multi-temp fusion
│   ├── B6_*.py                       # EVT approaches
│   ├── meta_attack_cv.py             # Meta-fusion
│   ├── aggregate_piece_level_*.py    # Aggregation & debiasing
│   ├── calibrate_scores.py           # Score calibration
│   ├── auc_delong.py                 # Statistical testing
│   └── plot_roc_academic.py          # Visualization
├── notagen/
│   ├── inference/                    # NotaGen inference modules
│   ├── data/                         # Data preprocessing
│   ├── clamp2/                       # CLAMP2 compatibility
│   └── requirements.txt
├── configs/
│   └── note_token_ids.json           # Token configuration
├── schemas/
│   ├── maestro_split.schema.json     # Data schema
│   └── transformer_score.schema.json # Score schema
└── README.md
```

### Key Concepts

#### Token Importance Score (TIS)
```
TIS = mean(log_probs[structural_tokens])
TIS_topk = mean(top-k_largest(log_probs[structural_tokens]))
```

#### Multi-Temperature Tail Fusion
Combines multiple temperature-scaled scores:
```
score_T = argmax_fusion(TIS_topk @ temperature T)
fusion_score = f(scores_{0.8}, scores_{1.0}, scores_{1.2}, scores_{1.5})
```

#### Debiasing Pipeline
1. **Length Matching**: Create balanced pairs by sample length
2. **Conditional Calibration**: Normalize scores within length bins
3. **View Aggregation**: Combine multiple evaluation views

## Usage Examples

### Complete Pipeline
```bash
#!/bin/bash

# Setup
DATA_DIR="./data"
MODEL_DIR="./models/transformer"
RESULTS_DIR="./results"
mkdir -p $RESULTS_DIR

# Step 1: Prepare data
python src/preprocessing/maestro_split.py \
    --maestro-dir /path/to/maestro \
    --output-dir $DATA_DIR/processed

# Step 2: Score all samples
python scripts/B5_multi_temp_tail.py \
    --model-dir $MODEL_DIR \
    --abc-dir $DATA_DIR/abc \
    --split-json $DATA_DIR/processed/split.json \
    --temperatures 0.8,1.0,1.2,1.5 \
    --top-k-values 64 \
    --output-dir $RESULTS_DIR

# Step 3: Fuse scores
python scripts/B5_aggregate_fusion.py \
    --scores-dir $RESULTS_DIR \
    --split-json $DATA_DIR/processed/split.json \
    --fusion-method geometric_mean \
    --output-dir $RESULTS_DIR/fusion

# Step 4: Aggregate to piece-level
python scripts/aggregate_piece_level_lenmatch.py \
    --sample-level $RESULTS_DIR/fusion/sample_scores.csv \
    --split-json $DATA_DIR/processed/split.json \
    --output-dir $RESULTS_DIR

# Step 5: Evaluate
python scripts/compute_low_fpr_metrics.py \
    --piece-level $RESULTS_DIR/piece_level.csv \
    --output-dir $RESULTS_DIR/metrics

python scripts/plot_roc_academic.py \
    --piece-level $RESULTS_DIR/piece_level.csv \
    --output-dir ./figures
```

## Performance Benchmarks

### Attack Performance (MAESTRO, Length-Matched)
| Method | AUC | TPR@1%FPR | TPR@5%FPR |
|--------|-----|-----------|-----------|
| Baseline (mean NLL) | 0.679 | 1.46% | 9.71% |
| StructTail-64 | 0.794 | 14.63% | 28.29% |
| StructTail+Fusion | 0.925 | 44.20% | 68.75% |

### Computational Cost
- Single sample scoring: ~50ms (transformer)
- Batch of 1000 samples: ~15 seconds
- Full MAESTRO (1267 samples): ~20 minutes
- Debiasing pipeline: ~5 minutes

## Configuration

### Score Configuration
Edit `configs/note_token_ids.json` to customize:
- Token vocabulary
- Structural token masks
- Temperature scales
- Top-k ranges

### Model Configuration
Transformer model parameters (in `src/train_transformer.py`):
- Hidden size: 768
- Number of layers: 12
- Attention heads: 12
- Dropout: 0.1

## Advanced Usage

### Custom Fusion Strategies
```python
from scripts.B5_aggregate_fusion import compute_fusion

# Define custom fusion
scores_fusion = compute_fusion(
    scores_dict={'T_0.8': s1, 'T_1.0': s2, 'T_1.2': s3, 'T_1.5': s4},
    method='geometric_mean',  # or 'max', 'mean', custom function
    weights=None  # or specify custom weights
)
```

### Low-FPR Analysis
```python
from scripts.compute_low_fpr_metrics import compute_low_fpr_metrics

metrics = compute_low_fpr_metrics(
    fpr_array, tpr_array,
    fpr_targets=[0.01, 0.05, 0.10]
)
```

### Statistical Testing
```python
from scripts.auc_delong import compute_delong_ci

auc, ci_lower, ci_upper = compute_delong_ci(y_true, y_score)
print(f"AUC: {auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
```

## NotaGen Integration

For Music Generation MIA attacks using NotaGen:

```bash
# Inference on ABC files
python notagen/inference/inference.py \
    --weights-path ./models/notagen_pretrain.pth \
    --abc-file ./data/abc/sample.abc \
    --output-path ./results/generation.abc
```

## Dataset

### MAESTRO v3.0.0
- 1276 MIDI pieces
- Automated splits (train/val/test)
- Metadata: composer, year, difficulty
- Pre-configured in `src/preprocessing/maestro_split.py`

### ABC Notation
- Symbolic representation used for structure analysis
- Automatic conversion from MIDI
- Structure tokens extracted for membership signals

## Citation

If you use TS-RaMIA in your research, please cite:

```bibtex
@inproceedings{liuts,
  title={TS-RaMIA: Membership Inference Attacks for Symbolic Music Generation Models},
  author={Liu, Yuxuan and Sang, Rui and Zhang, Peihong and Li, Zhixin and Zhang, Kunyang and He, Shengyuan and Li, Ye and Xu, Kaiyi and Li, Shengchen},
  booktitle={1st International Workshop on Emerging AI Technologies for Music}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or feedback, please open an issue on GitHub or contact the authors.

---

**Last Updated**: October 2025  
**Maintainer**: [GitHub Account](https://github.com/kaslim)
