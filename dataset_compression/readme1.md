# Dataset Compression Pipeline

Implementation based on `structure.md` specifications for compressing multi-lmentry benchmark datasets while maintaining high correlation with full dataset rankings.

## Pipeline Overview

The pipeline follows structure.md sections:

### Section 0: Global Setup (config.py)
- Reference model: ALIA-40b
- 3000 examples per (task, language)
- Fixed ratio grid: {1, 2, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100}%
- Multilingual embedding model: paraphrase-multilingual-MiniLM-L12-v2
- Similarity threshold: τ = 0.90
- Correlation thresholds: Spearman ≥ 0.90, Pearson ≥ 0.90

### Section 1: Data Extraction
- Load predictions from `predictions/<language>/<task>/ALIA-40b.json`
- Extract example IDs and scores
- Use "score" field only (ignore "certainty")

### Section 2: Random Sampling
- Multi-seed sampling: 10 seeds for r ≤ 10%, 5 seeds for r > 10%
- Aggregate correlations using median
- Evaluate Spearman (rank preservation) and Pearson (score preservation)

### Section 3: Clustering-based Sampling
- Encode prompts using multilingual sentence embeddings
- Cluster using cosine similarity threshold (τ = 0.90)
- Retain one representative per cluster (deduplicated pool)
- Apply random sampling on deduplicated pool

### Section 4: Subset Selection
- Traverse ratios in increasing order
- Select smallest ratio meeting both thresholds for two consecutive ratios (stability condition)
- Fallback: maximize Spearman with relaxed Pearson constraint

## Files

- **config.py**: Global configuration and fixed settings
- **samplers.py**: Sampling strategies (random, cluster_dedup with paraphrase deduplication)
- **eval_utils.py**: Score-based evaluation and correlation metrics
- **adaptive_sampling.py**: Main pipeline for finding optimal subsets
- **pipeline.py**: End-to-end compression pipeline with CLI

## Usage

```bash
# Single language
python -m dataset_compression --predictions predictions/ --output outputs/ --language en

# Multiple languages
python -m dataset_compression --predictions predictions/ --output outputs/ --languages en es ca de

# With GPU for embeddings
python -m dataset_compression --predictions predictions/ --output outputs/ --language en --device cuda
```

## Output Format

Results are saved to `<output_dir>/adaptive_results_<language>.json` with:
- Per-task results for each technique and ratio
- Multi-seed correlation values
- Aggregated correlations (median)
- Best configuration selection with stability/fallback method

## Sampling Techniques

### Random (`random`)
- Uniform random sampling with multiple seeds
- Baseline for comparison

### Cluster Dedup (`cluster_dedup`)
- Removes near-duplicate prompts using similarity clustering
- More efficient representation with higher compression ratios
- Steps:
  1. Encode prompts with multilingual SBERT
  2. Cluster by cosine similarity ≥ 0.90
  3. Keep one representative per cluster
  4. Random sample from deduplicated pool
