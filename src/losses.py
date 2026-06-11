"""Loss functions for M1 / M4 (SD) / M5 (FM)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# ------------------------------------------------------------------
# Shared PPO losses
# ------------------------------------------------------------------
def ppo_clip_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_eps: float = 0.2,
) -> Tensor:
    ratio = (new_log_probs - old_log_probs).exp()
    s1 = ratio * advantages
    s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(s1, s2).mean()

def value_loss(new_values: Tensor, returns: Tensor) -> Tensor:
    return F.mse_loss(new_values, returns)

# ------------------------------------------------------------------
# Self-Distillation loss
# ------------------------------------------------------------------
def sd_loss(logits_teacher: Tensor, logits_student: Tensor) -> Tensor:
    """KL( softmax(teacher) || log_softmax(student) ).

    Teacher logits must already be detached by the caller.
    """
    teacher_probs = F.softmax(logits_teacher, dim=-1)
    student_log_probs = F.log_softmax(logits_student, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

def lambda_sd_schedule(iteration: int, n_total: int, max_lambda: float = 0.1) -> float:
    """λ_SD ramps from 0 to max_lambda over the second half of training."""
    if n_total <= 0 or iteration < n_total * 0.5:
        return 0.0
    progress = (iteration - 0.5 * n_total) / (0.5 * n_total)
    return float(max_lambda * min(1.0, progress))