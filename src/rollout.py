"""Rollout buffer and collection loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch

from .env import OvercookedVecEnv
from .controller import Controller
from .reward_shaping import compute_final_reward
from .subgoal_encoder import encode_subgoal, zero_subgoal, SUBGOAL_DIM
from .state_summarizer import summarize_state

LLM_INTERVAL = 20  # env steps between LLM queries

@dataclass
class RolloutBuffer:
    """Stores one rollout of shape (T, n_envs, 2, ...)."""
    states:    np.ndarray = field(default=None)   # (T, n_envs, 2, state_dim)
    subgoals:  np.ndarray = field(default=None)   # (T, n_envs, 2, SUBGOAL_DIM)
    actions:   np.ndarray = field(default=None)   # (T, n_envs, 2)
    log_probs: np.ndarray = field(default=None)   # (T, n_envs, 2)
    values:    np.ndarray = field(default=None)   # (T, n_envs, 2)
    rewards:   np.ndarray = field(default=None)   # (T, n_envs)
    dones:     np.ndarray = field(default=None)   # (T, n_envs)


def collect_rollout(
    envs: OvercookedVecEnv,
    controller: Controller,
    rollout_len: int,
    shaped_weight: float,
    device: torch.device,
    method: str = "SMART-D",
    llm_planner: Optional[Any] = None,
) -> tuple[RolloutBuffer, np.ndarray]:
    """Collect rollout_len steps across all envs.

    Returns:
        buffer:      filled RolloutBuffer
        last_values: (n_envs, 2) bootstrap values for GAE
    """
    n_envs = envs.n_envs
    state_dim = envs.state_dim

    # Allocate buffers
    buf = RolloutBuffer(
        states=np.zeros((rollout_len, n_envs, 2, state_dim), dtype=np.float32),
        subgoals=np.zeros((rollout_len, n_envs, 2, SUBGOAL_DIM), dtype=np.float32),
        actions=np.zeros((rollout_len, n_envs, 2), dtype=np.int64),
        log_probs=np.zeros((rollout_len, n_envs, 2), dtype=np.float32),
        values=np.zeros((rollout_len, n_envs, 2), dtype=np.float32),
        rewards=np.zeros((rollout_len, n_envs), dtype=np.float32),
        dones=np.zeros((rollout_len, n_envs), dtype=bool),
    )

    # current subgoals per env: (n_envs, 2, SUBGOAL_DIM)
    current_subgoals = np.stack(
        [[zero_subgoal(), zero_subgoal()] for _ in range(n_envs)], axis=0
    ).astype(np.float32)

    states, _ = envs.reset()  # (n_envs, 2, state_dim)

    controller.eval()
    with torch.no_grad():
        for t in range(rollout_len):
            # -- SMART-D: query LLM every LLM_INTERVAL steps --
            if method == "SMART-D" and llm_planner is not None and t % LLM_INTERVAL == 0:
                raw_states = envs.get_raw_states()
                mdps = envs.get_mdps()
                summaries = [
                    summarize_state(raw_states[i], mdps[i], envs.layout_name)
                    for i in range(n_envs)
                ]
                jsons = llm_planner.query_batch(summaries)
                for i, js in enumerate(jsons):
                    enc = encode_subgoal(js, int(raw_states[i].timestep))
                    current_subgoals[i, 0] = enc
                    current_subgoals[i, 1] = enc

            # -- forward pass --
            states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
            subgoals_t = torch.as_tensor(current_subgoals, dtype=torch.float32, device=device)

            # reshape to (n_envs*2, dim) for batched forward
            s_flat = states_t.view(n_envs * 2, state_dim)
            g_flat = subgoals_t.view(n_envs * 2, SUBGOAL_DIM)

            actions_flat, log_probs_flat, values_flat = controller.act(s_flat, g_flat)

            actions = actions_flat.cpu().numpy().reshape(n_envs, 2)
            log_probs = log_probs_flat.cpu().numpy().reshape(n_envs, 2)
            values = values_flat.cpu().numpy().reshape(n_envs, 2)

            # -- env step --
            next_states, sparse_rewards, dones, infos = envs.step(actions)

            # -- shaped rewards (shared, broadcast to per-agent in GAE) --
            final_rewards = np.array([
                compute_final_reward(sparse_rewards[i], infos[i], shaped_weight)
                for i in range(n_envs)
            ], dtype=np.float32)

            # -- store --
            buf.states[t] = states
            buf.subgoals[t] = current_subgoals
            buf.actions[t] = actions
            buf.log_probs[t] = log_probs
            buf.values[t] = values
            buf.rewards[t] = final_rewards
            buf.dones[t] = dones

            states = next_states

            # reset subgoals for envs that terminated
            for i in range(n_envs):
                if dones[i]:
                    current_subgoals[i] = np.stack([zero_subgoal(), zero_subgoal()], axis=0)

        # -- bootstrap values --
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        subgoals_t = torch.as_tensor(current_subgoals, dtype=torch.float32, device=device)
        s_flat = states_t.view(n_envs * 2, -1)
        g_flat = subgoals_t.view(n_envs * 2, SUBGOAL_DIM)
        _, _, last_values_flat = controller.act(s_flat, g_flat)
        last_values = last_values_flat.cpu().numpy().reshape(n_envs, 2)

    controller.train()
    return buf, last_values
