"""Minibatch PPO update loop."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from .controller import Controller
from .losses import (
    ppo_clip_loss,
    sd_loss,
    value_loss,
)

def update_step(
    controller: Controller,
    optimizer: torch.optim.Optimizer,
    # flattened tensors – all shape (N,) or (N, dim) where N = T*n_envs*2
    states: Tensor,        # (N, state_dim)
    subgoals: Tensor,      # (N, 16)  real subgoals
    actions: Tensor,       # (N,)     int
    old_log_probs: Tensor, # (N,)
    advantages: Tensor,    # (N,)
    returns: Tensor,       # (N,)
    # hyperparams
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    k_epochs: int = 4,
    minibatch_size: int = 256,
    # method-specific
    method: str = "SMART-D",
    lambda_sd: float = 0.0,
    iteration: int = 0,
    n_total: int = 1,
) -> Dict[str, float]:
    """Run K_EPOCHS of minibatch PPO updates.

    Returns dict of mean scalar metrics for logging.
    """
    N = states.shape[0]
    zero_sg = torch.zeros_like(subgoals)

    metrics: Dict[str, list] = {
        "loss": [], "L_ppo": [], "L_value": [],
        "entropy": [], "L_sd": []
    }

    for _ in range(k_epochs):
        perm = torch.randperm(N, device=states.device)
        for start in range(0, N, minibatch_size):
            idx = perm[start: start + minibatch_size]

            mb_states = states[idx]
            mb_sg_real = subgoals[idx]
            mb_actions = actions[idx]
            mb_old_lp = old_log_probs[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]

            # -- Teacher branch (with subgoal) --
            out_teacher = controller.forward(mb_states, mb_sg_real)
            dist_teacher = Categorical(logits=out_teacher["logits"])
            new_log_probs = dist_teacher.log_prob(mb_actions)
            entropy = dist_teacher.entropy().mean()

            L_ppo = ppo_clip_loss(new_log_probs, mb_old_lp, mb_adv, clip_eps)
            L_val = value_loss(out_teacher["value"], mb_ret)

            total = L_ppo + value_coef * L_val - entropy_coef * entropy
            L_sd_val = torch.tensor(0.0, device=states.device)

            if method != "SMART-D":
                raise ValueError(
                    f"[update] Execution terminated, please use SMART-D method."
                )

            if lambda_sd > 0:
                # -- Student branch (zero subgoal) --
                out_student = controller.forward(mb_states, zero_sg[idx])
                L_sd_val = sd_loss(out_teacher["logits"].detach(), out_student["logits"])
                total = total + lambda_sd * L_sd_val

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), max_grad_norm)
            optimizer.step()

            metrics["loss"].append(float(total.item()))
            metrics["L_ppo"].append(float(L_ppo.item()))
            metrics["L_value"].append(float(L_val.item()))
            metrics["entropy"].append(float(entropy.item()))
            metrics["L_sd"].append(float(L_sd_val.item()))

    return {k: float(np.mean(v)) for k, v in metrics.items()}
