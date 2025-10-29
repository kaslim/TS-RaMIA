# GitHub Setup Guide

Instructions for uploading TS-RaMIA to GitHub

## Prerequisites

- GitHub account (https://github.com)
- Git installed on your machine
- Repository created on GitHub: `https://github.com/kaslim/TS-RaMIA.git`

## Step-by-Step Upload

### 1. Initialize Git Repository (First Time Only)

```bash
cd /path/to/TS-RaMIA
git init
```

### 2. Add All Files

```bash
git add .
```

### 3. Commit Changes

```bash
git commit -m "Initial commit: TS-RaMIA membership inference attack framework"
```

### 4. Add Remote Repository

```bash
git remote add origin https://github.com/kaslim/TS-RaMIA.git
```

### 5. Push to GitHub

```bash
# If this is the first push, use -u flag to set upstream
git branch -M main
git push -u origin main

# For subsequent pushes
git push origin main
```

## Commands for Reference

### Check Status
```bash
git status
```

### View Commit History
```bash
git log --oneline
```

### Update from Remote
```bash
git pull origin main
```

### Create Release/Tag
```bash
git tag -a v1.0.0 -m "Version 1.0.0 Release"
git push origin v1.0.0
```

## Folder Structure on GitHub

```
TS-RaMIA/
├── README.md              # Main documentation
├── QUICKSTART.md          # Quick start guide
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
│
├── src/                   # Source code
│   ├── preprocessing/
│   │   ├── maestro_split.py
│   │   ├── tokenize_maestro.py
│   │   └── abc_structure_utils.py
│   └── train_transformer.py
│
├── scripts/               # Experiment scripts
│   ├── score_tis_transformer.py
│   ├── B5_multi_temp_tail.py
│   ├── B5_aggregate_fusion.py
│   ├── B6_evt_tail_prob.py
│   ├── meta_attack_cv.py
│   ├── aggregate_piece_level_lenmatch.py
│   ├── calibrate_scores.py
│   ├── auc_delong.py
│   └── plot_roc_academic.py
│
├── notagen/               # NotaGen integration
│   ├── inference/
│   ├── data/
│   ├── clamp2/
│   ├── README.md
│   └── requirements.txt
│
├── configs/               # Configuration files
│   └── note_token_ids.json
│
└── schemas/               # JSON schemas
    ├── maestro_split.schema.json
    └── transformer_score.schema.json
```

## GitHub Configuration

### Add Description
```
TS-RaMIA: Membership Inference Attacks on Music Models 
via Transcription Structure Analysis
```

### Add Topics
```
membership-inference privacy music deep-learning security
```

### Add License
- GitHub will auto-detect MIT License

## Working with Branches

### Create a Development Branch
```bash
git checkout -b develop
```

### Merge Back to Main
```bash
git checkout main
git merge develop
git push origin main
```

## Updating Code

### Make Changes and Push
```bash
# Make your code changes
nano scripts/score_tis_transformer.py

# Stage changes
git add scripts/score_tis_transformer.py

# Commit with meaningful message
git commit -m "Improve TIS scoring performance"

# Push to GitHub
git push origin main
```

## Common Issues

### "fatal: repository not found"
**Solution**: Check your GitHub URL and ensure you have access
```bash
git remote -v  # Verify remote URL
```

### "Permission denied (publickey)"
**Solution**: Set up SSH keys for GitHub
```bash
ssh-keygen -t ed25519 -C "your@email.com"
# Add public key to GitHub Settings > SSH keys
```

### "Your branch is ahead of origin/main"
**Solution**: Push your commits
```bash
git push origin main
```

## Collaboration

### For Contributors
```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/TS-RaMIA.git
cd TS-RaMIA

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Add your feature description"

# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub website
```

## Release Workflow

### Create a New Release
```bash
# Update version in code/docs if needed
# Create a tag
git tag -a v1.1.0 -m "Release version 1.1.0"

# Push tag to GitHub
git push origin v1.1.0

# GitHub will automatically create a release
```

## Maintenance

### Regular Updates
```bash
# Pull latest changes
git pull origin main

# Keep local fork up to date
git fetch upstream
git rebase upstream/main
```

## Useful Resources

- [GitHub Documentation](https://docs.github.com)
- [Git Basics](https://git-scm.com/book/en/v2)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)

---

**Setup Complete!** Your TS-RaMIA repository is now on GitHub.
