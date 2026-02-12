from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Reference model used for extracting example IDs
REFERENCE_MODEL = "ALIA-40b"

# Number of examples per (task, language)
N_EXAMPLES = 3000

# Fixed ratio grid (percentages converted to decimals)
RATIO_GRID: Tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.05, 0.07,
    0.10, 0.12, 0.15, 0.18, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.50,
    0.60, 0.70, 0.80, 1.00,
)

# Sampling techniques
SAMPLING_TECHNIQUES: Tuple[str, ...] = ("random", "cluster_dedup", "difficulty_easy", "difficulty_mid", "difficulty_hard","difficulty_stratified")

# Multilingual embedding model for clustering
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Similarity threshold for clustering (τ = 0.90)
SIMILARITY_THRESHOLD = 0.90

# Correlation thresholds
SPEARMAN_THRESHOLD = 0.90
PEARSON_THRESHOLD = 0.90

# Number of seeds per ratio
def get_num_seeds(ratio: float) -> int:
    """Return number of random seeds for a given ratio.
    
    - 10 seeds for r ≤ 10%
    - 5 seeds for r > 10%
    """
    return 10 if ratio <= 0.10 else 5


# Fixed random seeds for reproducibility
RANDOM_SEEDS: Tuple[int, ...] = (42, 123, 456, 789, 1011, 1314, 1516, 1718, 1920, 2122)


@dataclass
class CompressionConfig:
    """Configuration for the compression pipeline."""
    
    # Data paths
    predictions_root: Path  # predictions/<language>/<task>/<model>.json
    output_dir: Path        # outputs_compression/
    
    # Language and tasks
    language: str
    task_names: Optional[List[str]] = None  # None = all tasks
    
    # Sampling parameters
    ratios: Tuple[float, ...] = RATIO_GRID
    techniques: Tuple[str, ...] = SAMPLING_TECHNIQUES
    
    # Thresholds
    spearman_thr: float = SPEARMAN_THRESHOLD
    pearson_thr: float = PEARSON_THRESHOLD
    
    # Clustering
    similarity_threshold: float = SIMILARITY_THRESHOLD
    embedding_model: str = EMBEDDING_MODEL
  
    device: str = "cpu"
    embed_batch_size: int = 32
    
    # Reference model
    reference_model: str = REFERENCE_MODEL
