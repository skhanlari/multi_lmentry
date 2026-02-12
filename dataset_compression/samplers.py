"""Sampling strategies for dataset compression.
Each sampler takes example IDs and returns subsets for multiple seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Tuple
import math
import random
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    RANDOM_SEEDS,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    get_num_seeds,
)

logger = logging.getLogger(__name__)

Json = Dict[str, Any]

# Text embedder (Multilingual Sentence-BERT)
# ----------------------------------------------------------------------
class TextEmbedder:
    """
    Multilingual text embedder using Sentence-BERT style models.
    Falls back to TF-IDF if transformers are unavailable.
    """
    
    def __init__(
        self, 
        model_name: str = EMBEDDING_MODEL, 
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._backend = None
        self._model = None
        self._vectorizer = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._backend = "sentence_transformers"
            logger.info(f"Using sentence-transformers: {model_name}")
        except ImportError:
            try:
                from transformers import AutoModel, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name).to(device)
                self._model.eval()
                self._backend = "transformers"
                logger.info(f"Using transformers: {model_name}")
            except Exception as e:
                logger.warning(f"Transformers unavailable ({e}), falling back to TF-IDF")
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._vectorizer = TfidfVectorizer(max_features=4096)
                self._backend = "tfidf"
    
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if self._backend == "sentence_transformers":
            return self._model.encode(
                list(texts), 
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        if self._backend == "tfidf":
            mat = self._vectorizer.fit_transform(list(texts))
            return mat.toarray().astype(np.float32)
        
        # Transformer encoding with mean pooling
        import torch
        all_vecs = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i:i + batch_size])
                tokens = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self._model(**tokens)
                last_hidden = outputs.last_hidden_state
                attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
                
                # Mean pooling
                pooled = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1).clamp(min=1.0)
                all_vecs.append(pooled.detach().cpu().numpy())
        
        return np.vstack(all_vecs) if all_vecs else np.zeros((0, 1), dtype=np.float32)


# Sampler configuration
# ----------------------------------------------------------------------
@dataclass
class SamplerConfig:
    embedder: Optional[TextEmbedder] = None
    embed_batch_size: int = 32
    similarity_threshold: float = SIMILARITY_THRESHOLD


def _k_from_ratio(n: int, ratio: float) -> int:
    if n <= 0:
        return 0
    return max(1, min(n, int(math.floor(n * ratio))))


def cluster_by_similarity(
    embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[List[List[int]], List[int]]:
    """
    Cluster embeddings using cosine similarity threshold.
    
    Groups prompts where cosine similarity >= threshold.
    Returns clusters and one representative per cluster.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        threshold: Cosine similarity threshold (default 0.90)
    
    Returns:
        Tuple of (clusters, representatives):
        - clusters: List of lists, each containing indices in that cluster
        - representatives: List of representative indices (one per cluster)
    """
    n = len(embeddings)
    if n == 0:
        return [], []
    
    # Compute pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    assigned = [False] * n
    clusters: List[List[int]] = []
    representatives: List[int] = []
    
    for i in range(n):
        if assigned[i]:
            continue
        
        # Start new cluster with i as representative
        cluster = [i]
        assigned[i] = True
        
        # Add all similar unassigned items
        for j in range(i + 1, n):
            if not assigned[j] and similarities[i, j] >= threshold:
                cluster.append(j)
                assigned[j] = True
        
        clusters.append(cluster)
        representatives.append(i)
    
    return clusters, representatives


# Sampling Strategies
# ----------------------------------------------------------------------
def sample_random(
    example_ids: Sequence[str],
    ratio: float,
    seed: int,
) -> List[str]:
    """
    Random sampling baseline.
    
    Args:
        example_ids: List of example IDs
        ratio: Sampling ratio (0.0 - 1.0)
    
    Returns:
        List of selected example IDs
    """
    n = len(example_ids)
    k = _k_from_ratio(n, ratio)
    
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    
    selected_indices = sorted(indices[:k])
    return [example_ids[i] for i in selected_indices]


def sample_random_multi_seed(
    example_ids: Sequence[str],
    ratio: float,
) -> Dict[int, List[str]]:
    """
    Random sampling with multiple seeds.
    
    Args:
        example_ids: List of example IDs
        ratio: Sampling ratio
    
    Returns:
        Dict mapping seed -> list of selected example IDs
    """
    num_seeds = get_num_seeds(ratio)
    seeds = RANDOM_SEEDS[:num_seeds]
    
    return {seed: sample_random(example_ids, ratio, seed) for seed in seeds}


def sample_cluster_dedup(
    example_ids: Sequence[str],
    prompts: Sequence[str],
    ratio: float,
    seed: int,
    embedder: TextEmbedder,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    batch_size: int = 32,
) -> Tuple[List[str], List[int]]:
    """
    Clustering-based sampling with paraphrase deduplication.
    
    1. Encode prompts using multilingual sentence embeddings
    2. Cluster using similarity threshold
    3. Retain one representative per cluster (deduplicated pool)
    4. Apply random sampling on deduplicated pool
    
    Args:
        example_ids: List of example IDs
        prompts: List of prompt texts corresponding to example IDs
        ratio: Sampling ratio
        seed: Random seed
        embedder: Text embedder for computing embeddings
        similarity_threshold: Cosine similarity threshold (default 0.90)
        batch_size: Batch size for embedding
    
    Returns:
        Tuple of (selected_ids, representative_indices):
        - selected_ids: List of selected example IDs
        - representative_indices: Indices of cluster representatives (deduplicated pool)
    """
    n = len(example_ids)
    if n == 0:
        return [], []
    
    embeddings = embedder.encode(list(prompts), batch_size=batch_size)
    
    # Cluster by similarity
    clusters, representatives = cluster_by_similarity(embeddings, similarity_threshold)
    
    logger.info(
        f"  Clustering: {n} examples -> {len(clusters)} clusters "
        f"(deduplication ratio: {len(clusters)/n:.2%})"
    )
    
    # Create deduplicated pool (representative IDs)
    dedup_ids = [example_ids[i] for i in representatives]
    
    # Random sample from deduplicated pool
    k = _k_from_ratio(len(dedup_ids), ratio)
    
    rng = random.Random(seed)
    shuffled = list(range(len(dedup_ids)))
    rng.shuffle(shuffled)
    
    selected_indices = sorted(shuffled[:k])
    selected_ids = [dedup_ids[i] for i in selected_indices]
    
    return selected_ids, representatives


def sample_cluster_dedup_multi_seed(
    example_ids: Sequence[str],
    prompts: Sequence[str],
    ratio: float,
    embedder: TextEmbedder,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    batch_size: int = 32,
    precomputed_dedup: Optional[Tuple[List[str], List[int]]] = None,
) -> Tuple[Dict[int, List[str]], List[int], List[str]]:
    """
    Clustering-based sampling with multiple seeds.
    
    Clustering is performed once, then random sampling is applied
    multiple times with different seeds.
    
    Args:
        example_ids: List of example IDs
        prompts: List of prompt texts
        ratio: Sampling ratio
        embedder: Text embedder
        similarity_threshold: Similarity threshold
        batch_size: Embedding batch size
        precomputed_dedup: Optional pre-computed (dedup_ids, representatives)
                          to skip re-clustering
    
    Returns:
        Tuple of (seed_results, representatives, dedup_ids):
        - seed_results: Dict mapping seed -> list of selected IDs
        - representatives: Indices of cluster representatives
        - dedup_ids: List of deduplicated example IDs
    """
    n = len(example_ids)
    if n == 0:
        return {}, [], []
    
    # Use precomputed clustering if available
    if precomputed_dedup is not None:
        dedup_ids, representatives = precomputed_dedup
    else:
        embeddings = embedder.encode(list(prompts), batch_size=batch_size)
        clusters, representatives = cluster_by_similarity(embeddings, similarity_threshold)
        
        logger.info(
            f"  Clustering: {n} examples -> {len(clusters)} clusters "
            f"(deduplication ratio: {len(clusters)/n:.2%})"
        )

        dedup_ids = [example_ids[i] for i in representatives]
    
    # Random sample with multiple seeds
    num_seeds = get_num_seeds(ratio)
    seeds = RANDOM_SEEDS[:num_seeds]
    
    k = _k_from_ratio(len(dedup_ids), ratio)
    seed_results = {}
    
    for seed in seeds:
        rng = random.Random(seed)
        shuffled = list(range(len(dedup_ids)))
        rng.shuffle(shuffled)
        selected_indices = sorted(shuffled[:k])
        seed_results[seed] = [dedup_ids[i] for i in selected_indices]
    
    return seed_results, representatives, dedup_ids


def sample_difficulty_variance_multi_seed(
    example_ids: Sequence[str],
    predictions_by_model: Dict[str, Dict[str, Any]],
    ratio: float,
    mode: str = "mid",
) -> Tuple[Dict[int, List[str]], Optional[Dict[int, Dict[str, List[str]]]]]:
    """
    Difficulty-based sampling using inter-model variance.

    Modes:
        - easy
        - hard
        - mid
        - stratified

    Returns:
        Tuple of:
            - Dict[seed -> selected example_ids]
            - Dict[seed -> {bin_name -> example_ids}] for stratified mode, None otherwise
    """

    import numpy as np
    from .config import RANDOM_SEEDS, get_num_seeds

    n = len(example_ids)
    if n == 0:
        return {}

    # compute p_i (mean correctness across models)
    # ---------------------------------------------------
    difficulty_scores = {}
    for eid in example_ids:
        scores = []
        for model_preds in predictions_by_model.values():
            if eid in model_preds:
                entry = model_preds[eid]
                if isinstance(entry, dict) and "score" in entry:
                    scores.append(entry["score"])
        if scores:
            p = np.mean(scores)
            difficulty_scores[eid] = float(p)

    if not difficulty_scores:
        return {}

    # define ranking depending on mode
    # ---------------------------------------------------
    if mode == "easy":
        ranked = sorted(difficulty_scores.items(), key=lambda x: -x[1])

    elif mode == "hard":
        ranked = sorted(difficulty_scores.items(), key=lambda x: x[1])

    elif mode == "mid":
        ranked = sorted(difficulty_scores.items(), key=lambda x: abs(x[1] - 0.5))

    elif mode == "stratified":
        # bin-based
        hard_bin = [eid for eid, p in difficulty_scores.items() if p < 0.33]
        mid_bin = [eid for eid, p in difficulty_scores.items() if 0.33 <= p <= 0.66]
        easy_bin = [eid for eid, p in difficulty_scores.items() if p > 0.66]
        ranked = None
    else:
        raise ValueError(f"Unknown difficulty mode: {mode}")

    # determine k
    # ---------------------------------------------------
    k = max(1, int(np.floor(n * ratio)))

    num_seeds = get_num_seeds(ratio)
    seeds = RANDOM_SEEDS[:num_seeds]

    results = {}
    bin_info = {} if mode == "stratified" else None

    for seed in seeds:
        rng = random.Random(seed)

        if mode != "stratified":
            pool_size = min(len(ranked), 5 * k)
            pool = [eid for eid, _ in ranked[:pool_size]]
            rng.shuffle(pool)
            selected = sorted(pool[:k])
        else:
            # Proportional Stratified Sampling
            # Compute proportions based on bin sizes
            n_total = len(hard_bin) + len(mid_bin) + len(easy_bin)
            if n_total == 0:
                n_total = 1  # Avoid division by zero
            
            p_hard = len(hard_bin) / n_total
            p_mid = len(mid_bin) / n_total
            
            # Allocate proportionally
            k_hard = round(k * p_hard)
            k_mid = round(k * p_mid)
            k_easy = k - k_hard - k_mid  # Ensures sum equals k
            
            # Clamp to available bin sizes
            k_hard = min(k_hard, len(hard_bin))
            k_mid = min(k_mid, len(mid_bin))
            k_easy = min(k_easy, len(easy_bin))
            
            selected = []
            seed_bins = {"hard": [], "mid": [], "easy": []}

            # Sample from each bin
            hard_copy = hard_bin.copy()
            mid_copy = mid_bin.copy()
            easy_copy = easy_bin.copy()
            
            rng.shuffle(hard_copy)
            rng.shuffle(mid_copy)
            rng.shuffle(easy_copy)
            
            hard_selected = hard_copy[:k_hard]
            mid_selected = mid_copy[:k_mid]
            easy_selected = easy_copy[:k_easy]
            
            selected.extend(hard_selected)
            selected.extend(mid_selected)
            selected.extend(easy_selected)
            
            seed_bins["hard"] = hard_selected
            seed_bins["mid"] = mid_selected
            seed_bins["easy"] = easy_selected

            remaining = k - len(selected)
            # Redistribute remaining proportionally across bins that still have capacity
            if remaining > 0:
                bins = [
                    ("hard", hard_copy, k_hard),
                    ("mid", mid_copy, k_mid),
                    ("easy", easy_copy, k_easy),
                ]
                
                for name, bin_list, k_bin in bins:
                    capacity = len(bin_list) - k_bin
                    if capacity > 0 and remaining > 0:
                        add = min(capacity, remaining)
                        extra_selected = bin_list[k_bin:k_bin + add]
                        selected.extend(extra_selected)
                        seed_bins[name].extend(extra_selected)
                        remaining -= add

            selected = sorted(selected[:k])
            bin_info[seed] = seed_bins

        results[seed] = selected

    return results, bin_info


SAMPLER_MAP = {
    "random": sample_random,
    "cluster_dedup": sample_cluster_dedup,
}


def get_sampler(name: str):
    """Get sampler function by name."""
    name = name.lower().strip()
    if name not in SAMPLER_MAP:
        raise ValueError(
            f"Unknown sampling technique: {name}. "
            f"Available: {list(SAMPLER_MAP.keys())}"
        )
    return SAMPLER_MAP[name]
