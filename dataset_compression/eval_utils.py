"""
Evaluation utilities for dataset compression.

This module provides functions for:
- Loading prediction files with scores
- Computing model score vectors on subsets
- Computing Spearman and Pearson correlations

- Evaluation signal: prediction results using "score" only (ignore "certainty")
- Spearman: rank preservation
- Pearson: score preservation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import logging

import numpy as np
from scipy.stats import spearmanr, pearsonr

from .config import REFERENCE_MODEL

logger = logging.getLogger(__name__)

Json = Dict[str, Any]


def load_predictions(path: str) -> Dict[str, Any]:

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin-1') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load predictions from {path}: {e}")
        return {}


def extract_example_ids(predictions: Dict[str, Any]) -> List[str]:

    keys = list(predictions.keys())
    try:
        return sorted(keys, key=lambda k: int(k))
    except ValueError:
        return sorted(keys)


def extract_scores(
    predictions: Dict[str, Any],
    example_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:

    if example_ids is None:
        example_ids = extract_example_ids(predictions)
    
    scores = {}
    for eid in example_ids:
        if eid in predictions:
            entry = predictions[eid]
            if isinstance(entry, dict) and "score" in entry:
                scores[eid] = entry["score"]
            elif isinstance(entry, (int, float)):
                scores[eid] = entry
            else:
                scores[eid] = 0.0
        else:
            scores[eid] = 0.0
    
    return scores


def extract_prompts(predictions: Dict[str, Any]) -> Dict[str, str]:

    prompts = {}
    for eid, entry in predictions.items():
        if isinstance(entry, dict):
            # Try common keys for prompt text
            for key in ("input", "prompt", "question", "text"):
                if key in entry:
                    prompts[eid] = str(entry[key])
                    break
            else:
                prompts[eid] = str(entry)
        else:
            prompts[eid] = str(entry)
    
    return prompts

# Score computation for model evaluation

def compute_model_accuracy(
    predictions: Dict[str, Any],
    subset_ids: Optional[Sequence[str]] = None,
) -> float:

    scores = extract_scores(predictions, subset_ids)
    
    if not scores:
        return float("nan")
    
    values = list(scores.values())
    return sum(values) / len(values)


def compute_all_models_scores(
    predictions_dir: str,
    example_ids: Sequence[str],
    model_names: Sequence[str],
) -> Dict[str, float]:

    from pathlib import Path
    pred_dir = Path(predictions_dir)
    
    scores = {}
    for model in model_names:
        pred_path = pred_dir / f"{model}.json"
        if not pred_path.exists():
            logger.warning(f"Predictions not found: {pred_path}")
            scores[model] = float("nan")
            continue
        
        predictions = load_predictions(str(pred_path))
        scores[model] = compute_model_accuracy(predictions, example_ids)
    
    return scores

def spearman_correlation(
    subset_scores: Dict[str, float],
    full_scores: Dict[str, float],
) -> float:

    models = sorted(set(subset_scores.keys()) & set(full_scores.keys()))
    
    if len(models) < 2:
        return float("nan")
    
    a = [subset_scores[m] for m in models]
    b = [full_scores[m] for m in models]
    
    # Filter out NaN values
    valid = [(x, y) for x, y in zip(a, b) if not (np.isnan(x) or np.isnan(y))]
    if len(valid) < 2:
        return float("nan")
    
    a_valid, b_valid = zip(*valid)
    
    try:
        r, _ = spearmanr(a_valid, b_valid)
        return float(r) if not np.isnan(r) else float("nan")
    except Exception as e:
        logger.warning(f"Spearman correlation failed: {e}")
        return float("nan")


def pearson_correlation(
    subset_scores: Dict[str, float],
    full_scores: Dict[str, float],
) -> float:

    models = sorted(set(subset_scores.keys()) & set(full_scores.keys()))
    
    if len(models) < 2:
        return float("nan")
    
    a = [subset_scores[m] for m in models]
    b = [full_scores[m] for m in models]
    
    # Filter out NaN values
    valid = [(x, y) for x, y in zip(a, b) if not (np.isnan(x) or np.isnan(y))]
    if len(valid) < 2:
        return float("nan")
    
    a_valid, b_valid = zip(*valid)
    
    try:
        r, _ = pearsonr(a_valid, b_valid)
        return float(r) if not np.isnan(r) else float("nan")
    except Exception as e:
        logger.warning(f"Pearson correlation failed: {e}")
        return float("nan")


def aggregate_correlations(
    correlations: Sequence[float],
    method: str = "median",
) -> float:

    valid = [c for c in correlations if not np.isnan(c)]
    
    if not valid:
        return float("nan")
    
    if method == "median":
        return float(np.median(valid))
    else:
        return float(np.mean(valid))


def compute_correlations_for_subset(
    subset_ids: Sequence[str],
    full_scores: Dict[str, float],
    predictions_dir: str,
    model_names: Sequence[str],
) -> Tuple[float, float]:

    subset_scores = compute_all_models_scores(predictions_dir, subset_ids, model_names)
    
    spearman = spearman_correlation(subset_scores, full_scores)
    pearson = pearson_correlation(subset_scores, full_scores)
    
    return spearman, pearson
