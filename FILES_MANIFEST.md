# TS-RaMIA Files Manifest

## Directory Structure

```
TS-RaMIA/
├── README.md                          # Main documentation
├── QUICKSTART.md                      # 15-minute quick start guide
├── GITHUB_SETUP.md                    # GitHub upload instructions
├── DEPLOYMENT_CHECKLIST.md            # Pre-upload checklist
├── FILES_MANIFEST.md                  # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── src/                               # Core source code
│   ├── preprocessing/
│   │   ├── maestro_split.py          # MAESTRO dataset splitting
│   │   ├── tokenize_maestro.py       # Tokenization pipeline
│   │   └── abc_structure_utils.py    # ABC notation utilities
│   └── train_transformer.py           # Transformer model training
│
├── scripts/                           # Experiment and evaluation scripts
│   ├── score_tis_transformer.py       # Base TIS scoring (Top-k)
│   ├── score_tis_transformer_v2.py    # Enhanced TIS scoring
│   ├── score_tis_weighted_tail.py     # Weighted tail scoring
│   ├── B5_multi_temp_tail.py          # Multi-temperature scoring
│   ├── B5_aggregate_fusion.py         # Fusion strategies
│   ├── B6_evt_tail_prob.py            # Extreme Value Theory approach
│   ├── meta_attack_cv.py              # Meta-learning fusion
│   ├── aggregate_piece_level_lenmatch.py  # Aggregation & debiasing
│   ├── calibrate_scores.py            # Score calibration
│   ├── auc_delong.py                  # DeLong statistical tests
│   └── plot_roc_academic.py           # Publication-quality plotting
│
├── notagen/                           # NotaGen integration modules
│   ├── inference/
│   │   ├── config.py                 # Inference configuration
│   │   ├── inference.py              # Inference pipeline
│   │   └── utils.py                  # Helper utilities
│   ├── data/
│   │   ├── 1_batch_xml2abc.py        # XML to ABC conversion
│   │   ├── 2_data_preprocess.py      # Data preprocessing
│   │   ├── 3_batch_abc2xml.py        # ABC to XML conversion
│   │   ├── abc2xml.py                # ABC conversion utilities
│   │   ├── xml2abc.py                # XML conversion utilities
│   │   └── README.md                 # Data module documentation
│   ├── clamp2/
│   │   ├── config.py                 # CLAMP2 configuration
│   │   ├── extract_clamp2.py         # CLAMP2 extraction
│   │   ├── statistics.py             # Statistics computation
│   │   └── utils.py                  # Helper functions
│   ├── README.md                     # NotaGen module overview
│   └── requirements.txt              # NotaGen dependencies
│
├── configs/                           # Configuration files
│   └── note_token_ids.json           # Token configuration
│
└── schemas/                           # JSON schemas
    ├── maestro_split.schema.json     # MAESTRO split schema
    └── transformer_score.schema.json # Score output schema
```

## File Count Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Python Code | 21 | Core algorithms and utilities |
| Documentation | 4 | README, guides, manifests |
| Configuration | 1 | Token settings |
| Schemas | 2 | Data format validation |
| Other | 11 | Supporting files |
| **Total** | **39** | **Complete framework** |

## Core Modules Description

### Data Processing (src/preprocessing/)

**maestro_split.py**
- Handles MAESTRO dataset loading and splitting
- Creates train/val/test splits
- Generates protocol JSON files
- Features: deterministic splitting, stratified sampling

**tokenize_maestro.py**
- Converts ABC/MIDI to token sequences
- Uses miditok library
- Saves as NPZ and JSONL formats
- Features: structure masking, batch processing

**abc_structure_utils.py**
- ABC notation parsing and manipulation
- Structure token identification
- Masking utilities for privacy analysis
- Features: header/body separation, structure extraction

### Scoring Modules (scripts/)

**score_tis_transformer.py**
- Base Token Importance Score implementation
- Loads pretrained transformer
- Computes mean and top-k scores
- Features: batching, GPU support, progress tracking

**B5_multi_temp_tail.py**
- Multi-temperature scoring extension
- Generates scores at T ∈ {0.8, 1.0, 1.2, 1.5}
- Top-k aggregation across temperatures
- Features: flexible k values, temperature ranges

**B5_aggregate_fusion.py**
- Combines multiple scoring signals
- Supports: max, mean, geometric mean fusion
- Conditional normalization
- Features: cross-validation, uncertainty quantification

**B6_evt_tail_prob.py**
- Extreme Value Theory for tail probability
- Gumbel distribution fitting
- P-value estimation
- Features: robust estimation, low-FPR optimization

**meta_attack_cv.py**
- Meta-learning based fusion
- Logistic regression on score combinations
- Cross-validation framework
- Features: score combination learning, statistical testing

### Aggregation & Evaluation (scripts/)

**aggregate_piece_level_lenmatch.py**
- Aggregates sample scores to piece level
- Length-matched pair generation
- Debiasing pipeline
- Features: stratified sampling, bias correction

**calibrate_scores.py**
- Conditional score calibration
- Length-bin normalization
- Isotonic regression
- Features: probability calibration, robustness

**auc_delong.py**
- DeLong method for AUC confidence intervals
- Statistical significance testing
- Comparison between methods
- Features: non-parametric, exact computation

**compute_low_fpr_metrics.py**
- Low False Positive Rate analysis
- TPR@k%FPR computation
- Partial AUC calculation
- Features: multiple FPR targets, uncertainty bands

**plot_roc_academic.py**
- Publication-quality ROC curve generation
- Academic style formatting
- Color-blind friendly palettes
- Features: multiple methods, low-FPR zoom

### Data Modules (notagen/)

**inference/**
- NotaGen model inference pipeline
- Score generation from ABC files
- Character-level NLL computation
- Features: batch processing, streaming

**data/**
- ABC ↔ XML bidirectional conversion
- MIDI to ABC conversion
- Batch processing utilities
- Features: error handling, format validation

**clamp2/**
- CLAMP2 compatibility layer
- Character and structure extraction
- Statistical analysis
- Features: hierarchical structure, metadata

## Dependencies

### Core (requirements.txt)
- torch ≥ 1.9.0
- transformers ≥ 4.20.0
- pandas ≥ 1.3.0
- scikit-learn ≥ 0.24.0

### Music Processing
- music21 ≥ 7.3.0
- miditok ≥ 2.0.0
- pretty_midi ≥ 0.2.9

### Analysis & Visualization
- numpy ≥ 1.20.0
- scipy ≥ 1.7.0
- matplotlib ≥ 3.3.0
- seaborn ≥ 0.11.0

## Configuration Files

### note_token_ids.json
```json
{
  "vocabulary_size": 1000,
  "structural_tokens": [
    "BAR",
    "TIME",
    "KEY",
    "REST",
    "CLEF"
  ],
  "token_ranges": {
    "pitch": [60, 84],
    "duration": [1, 128]
  }
}
```

### Schemas
- **maestro_split.schema.json**: Validates MAESTRO split files
- **transformer_score.schema.json**: Validates score output format

## Code Organization Principles

1. **Modularity**: Each script is independent and can be run separately
2. **Reusability**: Common functions in utility modules
3. **Reproducibility**: Fixed random seeds, explicit parameters
4. **Documentation**: Docstrings for all functions and classes
5. **Robustness**: Error handling and input validation

## File Size Breakdown

- Python code: ~450 KB (12 scripts + modules)
- Configuration: ~2 KB
- Schemas: ~3 KB
- Documentation: ~100 KB
- Other: ~45 KB
- **Total: ~600 KB**

## Usage Patterns

### Single File Use
Each script can be used independently:
```bash
python scripts/score_tis_transformer.py --help
```

### Pipeline Usage
Scripts are designed to work in sequence:
```bash
score_tis_transformer.py
  ↓
B5_aggregate_fusion.py
  ↓
aggregate_piece_level_lenmatch.py
  ↓
compute_low_fpr_metrics.py
```

### Custom Integration
Import modules in your own code:
```python
from scripts.auc_delong import compute_delong_ci
from src.preprocessing.maestro_split import load_split
```

## Maintenance Notes

- All paths use relative imports
- No hardcoded system paths
- Compatible with Windows/Linux/Mac
- Python 3.8+ required
- GPU optional but recommended

## Next Steps for Users

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Read QUICKSTART.md for basic usage
4. Run example scripts with sample data
5. Adapt to your own models/datasets

---

**Last Updated**: October 2025  
**Framework Version**: 1.0.0  
**Status**: Production Ready
