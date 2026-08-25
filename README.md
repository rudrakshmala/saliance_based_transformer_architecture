# Functionally Specialized Transformer Blocks (FSTB)

Research-grade experimental framework for evaluating **Functionally Specialized Transformer Blocks (FSTB)** against standard homogenous decoder-only transformers and memory-augmented baselines.

## Architecture Overview

```
                          ┌──────────────────────────────────────┐
                          │     Input Token Sequence (X)         │
                          └──────────────────┬───────────────────┘
                                             │
                          ┌──────────────────▼───────────────────┐
                          │   Stage A: Memory Selection          │
                          │   (Transformer Blocks 1–6)           │
                          └──────────────────┬───────────────────┘
                                             │
                          ┌──────────────────▼───────────────────┐
                          │   Stage B: Memory Encoding           │
                          │   (Transformer Blocks 7–12)          │
                          └──────────┬───────────────────┬───────┘
                                     │                   │
                     ┌───────────────▼────────┐  ┌───────▼────────┐
                     │ Dynamic Memory         │  │ Learnable      │
                     │ Controller Store       │  │ Routing Gate   │
                     └───────────────┬────────┘  └───────┬────────┘
                                     │                   │
                          ┌──────────▼───────────────────▼───────┐
                          │   Stage C: Memory Validation         │
                          │   (Transformer Blocks 13–18)         │
                          └──────────────────┬───────────────────┘
                                             │
                          ┌──────────────────▼───────────────────┐
                          │   Stage D: Response Generation       │
                          │   (Transformer Blocks 19–24)         │
                          └──────────────────┬───────────────────┘
                                             │
                          ┌──────────────────▼───────────────────┐
                          │     Output Next-Token Logits P(X)    │
                          └──────────────────────────────────────┘
```

## Quick Start

### 1. Run Unit & Integration Tests
```bash
python scripts/run_tests.py
```

### 2. Run Main Experiment Pipeline
```bash
python scripts/run_experiments.py
```

### 3. Open Generated Report & Figures
- Interactive HTML Dashboard: `results/dashboard.html`
- Layer-wise CKA Heatmap: `results/figures/cka_similarity_heatmap.png`
- Attention Entropy Profile: `results/figures/attention_entropy_by_layer.png`
- Ablation Study Matrix Chart: `results/figures/ablation_matrix_comparison.png`

## Directory Structure
- `fstb/models/`: Baseline Transformer, FSTB Model, Memory Interface, Dynamic Memory Controller, Routing Gate, Stage Heads, and Loss Function.
- `fstb/baselines/`: RAG, Vector Memory, Summarization Memory, and Memory Replay baselines.
- `fstb/data/`: Long-term Conversation, Contradiction, Temporal, Preference Evolution, and Multi-Session Project dataset generators.
- `fstb/evaluation/`: Metric suite, internal block specialization probing (CKA, attention entropy), statistical significance tests.
- `fstb/ablations/`: 10-condition ablation matrix runner.
- `fstb/visualization/`: Matplotlib/Seaborn plots and HTML visual dashboard generator.
- `docs/THESIS_DOCUMENTATION.md`: Complete theoretical model, math formulation, methodology, and results guide.
