"""Reward shaping weight schedule and final reward computation."""
from __future__ import annotations

def shaped_weight_schedule(iteration: int, n_total: int) -> float:
    """Return shaped reward coefficient for the current iteration.

    First 50% of training: weight = 1.0
    Second 50%: linearly anneal from 1.0 to 0.0
    """
    if n_total <= 0:
        return 0.0
    frac = iteration / n_total
    if frac <= 0.5:
        return 1.0
    return max(0.0, 1.0 - (frac - 0.5) / 0.5)

def compute_final_reward(
    sparse_reward: float,
    info: dict,
    shaped_weight: float,
) -> float:
    """sparse_reward + shaped_weight * sum(shaped_r_by_agent)."""
    shaped = sum(float(r) for r in info.get("shaped_r_by_agent", [0.0, 0.0]))
    return float(sparse_reward) + shaped_weight * shaped