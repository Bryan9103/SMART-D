"""Controller: shared MLP policy (actor + critic)."""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical

SUBGOAL_DIM = 18
NUM_ACTIONS = 6

class Controller(nn.Module):
    """Shared policy network for both agents (self-play).

    Input:  state (state_dim,) concatenated with subgoal (18,)
    Hidden: Linear(64) -> ReLU -> Linear(64) -> ReLU
    Heads:
      actor:        Linear -> 6  (action logits)
      critic:       Linear -> 1  (state value)
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int = NUM_ACTIONS,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        in_dim = state_dim + SUBGOAL_DIM
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        self.actor_head = nn.Linear(hidden, num_actions)
        self.critic_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> dict:
        """Forward pass for a batch of agent observations.

        Args:
            state:   (..., state_dim)
            subgoal: (..., 18)   zeros for student branch / eval

        Returns dict with keys:
            logits       (..., num_actions)
            value        (...,)
            hidden       (..., hidden)
        """
        x = torch.cat([state, subgoal], dim=-1)
        h = self.trunk(x)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return {"logits": logits, "value": value, "hidden": h}

    def act(
        self,
        state: torch.Tensor,
        subgoal: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) actions for a batch.

        state shape: (batch, state_dim / 18)
        Returns: actions (batch,), log_probs (batch,), values (batch,)
        """
        out = self.forward(state, subgoal)
        dist = Categorical(logits=out["logits"])
        if deterministic:
            actions = out["logits"].argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, out["value"]
