"""Multi-agent Generalised Advantage Estimation (GAE)."""
from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-agent advantages and returns via GAE.

    Args:
        rewards:     (T, n_envs)        shared reward per step
        values:      (T, n_envs, 2)     per-agent value estimates
        dones:       (T, n_envs)        episode-end flags
        last_values: (n_envs, 2)        bootstrap values after last step
        gamma, lam:  GAE hyperparams

    Returns:
        advantages: (T, n_envs, 2)
        returns:    (T, n_envs, 2)
    """
    T, n_envs = rewards.shape
    advantages = np.zeros((T, n_envs, 2), dtype=np.float32)

    # Broadcast shared reward to per-agent: (T, n_envs) -> (T, n_envs, 2)
    rewards_2 = rewards[:, :, np.newaxis]  # (T, n_envs, 1) broadcasts to 2

    gae = np.zeros((n_envs, 2), dtype=np.float32)
    next_v = last_values.copy()  # (n_envs, 2)

    for t in reversed(range(T)):
        nonterminal = (1.0 - dones[t])[:, np.newaxis]  # (n_envs, 1)
        delta = rewards_2[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
        next_v = values[t]

    returns = advantages + values
    return advantages, returns
