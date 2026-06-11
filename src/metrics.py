"""Final evaluation metrics: IQM + bootstrap CI via rliable."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_iqm_ci(
    returns_dict: Dict[str, List[float]],
    n_bootstrap: int = 2000,
) -> Dict[str, Dict[str, float]]:
    """Compute IQM and 95% bootstrap CI for each method.

    Args:
        returns_dict: {"Standard PPO": [seed0_ret, seed1_ret, ...], "SMART-D": [...], ...}
        n_bootstrap:  number of bootstrap samples

    Returns:
        {"Standard PPO": {"iqm": ..., "ci_low": ..., "ci_high": ...}, ...}
    """
    try:
        from rliable import library as rly
        from rliable import metrics as rl_metrics

        # rliable expects shape (n_runs, n_episodes) per algorithm
        score_dict = {k: np.array(v, dtype=np.float32)[:, np.newaxis] for k, v in returns_dict.items()}
        aggregate_fn = lambda scores: np.array([rl_metrics.aggregate_iqm(scores)])
        results, cis = rly.get_interval_estimates(score_dict, aggregate_fn, reps=n_bootstrap)
        out = {}
        for k in returns_dict:
            out[k] = {
                "iqm": float(results[k][0]),
                "ci_low": float(cis[k][0, 0]),
                "ci_high": float(cis[k][1, 0]),
            }
        return out

    except Exception:
        # Fallback: plain mean + std-of-mean CI
        out = {}
        for k, vals in returns_dict.items():
            arr = np.array(vals, dtype=np.float32)
            # IQM: trim bottom/top 25%
            lo, hi = np.percentile(arr, 25), np.percentile(arr, 75)
            trimmed = arr[(arr >= lo) & (arr <= hi)]
            iqm = float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(arr))
            se = float(np.std(arr) / max(1, np.sqrt(len(arr))))
            out[k] = {"iqm": iqm, "ci_low": iqm - 1.96 * se, "ci_high": iqm + 1.96 * se}
        return out
