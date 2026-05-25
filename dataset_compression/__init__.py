"""
- config: Global configuration and fixed settings
- samplers: Sampling strategies (random, cluster_dedup)
- eval_utils: Score-based evaluation and correlation metrics
- adaptive_sampling: Main pipeline for finding optimal subsets
- pipeline: End-to-end compression pipeline with CLI
"""

__version__ = "2.0.0"

from .config import (
    CompressionConfig,
    RATIO_GRID,
    SAMPLING_TECHNIQUES,
    REFERENCE_MODEL,
    SPEARMAN_THRESHOLD,
    PEARSON_THRESHOLD,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
)
from .samplers import (
    TextEmbedder,
    SamplerConfig,
    sample_random_multi_seed,
    sample_cluster_dedup_multi_seed,
    cluster_by_similarity,
)
from .eval_utils import (
    load_predictions,
    extract_example_ids,
    compute_model_accuracy,
    spearman_correlation,
    pearson_correlation,
    aggregate_correlations,
)
from .adaptive_sampling import run_adaptive_sampling
from .pipeline import (
    run_compression_pipeline,
    run_multi_language_pipeline,
)

__all__ = [
    # Config
    "CompressionConfig",
    "RATIO_GRID",
    "SAMPLING_TECHNIQUES",
    "REFERENCE_MODEL",
    "SPEARMAN_THRESHOLD",
    "PEARSON_THRESHOLD",
    "SIMILARITY_THRESHOLD",
    "EMBEDDING_MODEL",
    # Samplers
    "TextEmbedder",
    "SamplerConfig",
    "sample_random_multi_seed",
    "sample_cluster_dedup_multi_seed",
    "cluster_by_similarity",
    # Eval utils
    "load_predictions",
    "extract_example_ids",
    "compute_model_accuracy",
    "spearman_correlation",
    "pearson_correlation",
    "aggregate_correlations",
    # Pipeline
    "run_adaptive_sampling",
    "run_compression_pipeline",
    "run_multi_language_pipeline",
]
