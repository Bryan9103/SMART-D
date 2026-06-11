"""Evaluation: play_episodes, run_eval, distillation gap."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .env import OvercookedVecEnv
from .controller import Controller
from .subgoal_encoder import encode_subgoal, zero_subgoal, SUBGOAL_DIM
from .state_summarizer import summarize_state

LLM_INTERVAL = 50

def play_episodes(
    controller: Controller,
    layout_name: str,
    n_episodes: int,
    device: torch.device,
    use_llm: bool = False,
    llm_planner: Optional[Any] = None,
    horizon: int = 400,
) -> Tuple[list, list]:
    """Run n_episodes of self-play. Returns (returns, soup_counts)."""
    env = OvercookedVecEnv(layout_name=layout_name, n_envs=1, horizon=horizon)
    returns, soup_counts = [], []

    controller.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            states, _ = env.reset()
            current_subgoal = np.stack([zero_subgoal(), zero_subgoal()], axis=0)[np.newaxis]  # (1,2,SUBGOAL_DIM)
            total_reward = 0.0
            soups = 0

            for step in range(horizon):
                if use_llm and llm_planner is not None and step % LLM_INTERVAL == 0:
                    raw = env.get_raw_states()[0]
                    summary = summarize_state(raw, env.get_mdps()[0], layout_name)
                    js = llm_planner.query_one(summary)
                    enc = encode_subgoal(js, int(raw.timestep))
                    current_subgoal[0, 0] = enc
                    current_subgoal[0, 1] = enc

                s_t = torch.as_tensor(states, dtype=torch.float32, device=device).view(2, -1)
                g_t = torch.as_tensor(current_subgoal, dtype=torch.float32, device=device).view(2, SUBGOAL_DIM)
                actions, _, _ = controller.act(s_t, g_t, deterministic=True)
                actions_np = actions.cpu().numpy().reshape(1, 2)

                states, sparse_rewards, dones, infos = env.step(actions_np)
                total_reward += float(sparse_rewards[0])
                soups += int(round(float(sparse_rewards[0]) / 20.0))

                if dones[0]:
                    break

            returns.append(total_reward)
            soup_counts.append(soups)

    controller.train()
    return returns, soup_counts


def run_eval(
    controller: Controller,
    layout_name: str,
    device: torch.device,
    method: str = "SMART-D",
    llm_planner: Optional[Any] = None,
    n_episodes: int = 20,
    horizon: int = 400,
) -> Dict[str, float]:
    """Quick eval during training. Returns metrics dict."""
    # Eval without LLM (actual deployment setting)
    rets_no_llm, soups_no_llm = play_episodes(
        controller, layout_name, n_episodes, device,
        use_llm=False, horizon=horizon,
    )
    metrics: Dict[str, float] = {
        "eval/return_mean": float(np.mean(rets_no_llm)),
        "eval/return_std": float(np.std(rets_no_llm)),
        "eval/soups_mean": float(np.mean(soups_no_llm)),
    }

    # Distillation gap: only meaningful for M4/M5
    if method == "SMART-D" and llm_planner is not None:
        rets_llm, _ = play_episodes(
            controller, layout_name, n_episodes, device,
            use_llm=True, llm_planner=llm_planner, horizon=horizon,
        )
        gap = float(np.mean(rets_llm)) - float(np.mean(rets_no_llm))
        metrics["eval/return_with_llm"] = float(np.mean(rets_llm))
        metrics["eval/distillation_gap"] = gap

    return metrics
