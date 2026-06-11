"""
Self-play PPO baseline for Overcooked-AI.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

NUM_ACTIONS = len(Action.ALL_ACTIONS)  # 6

ExplorationFn = Callable[[np.ndarray, int], Optional[int]]

def random_epsilon_explore(
    obs: np.ndarray, agent_idx: int, epsilon: float
) -> Optional[int]:
    """Baseline: with probability epsilon return a uniform random action,
    otherwise return None (meaning: use the policy's sampled action)."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(NUM_ACTIONS))
    return None

def make_explore_fn(epsilon: float) -> Optional[ExplorationFn]:
    if epsilon <= 0.0:
        return None
    return lambda obs, idx: random_epsilon_explore(obs, idx, epsilon)

# ===========================================================================
# Network
# ===========================================================================
class ActorCritic(nn.Module):
    """Shared actor-critic MLP. Architecture: ReLU trunk + orthogonal init
    (actor head gain 0.01 so the policy starts near-uniform; critic/trunk gain 1.0)."""

    def __init__(self, obs_dim: int, n_actions: int = NUM_ACTIONS, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]

# ===========================================================================
# Config
# ===========================================================================
@dataclass
class Config:
    layout: str = "cramped_room"
    horizon: int = 400
    total_steps: int = 6_000_000
    n_rollouts_per_update: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    minibatch_size: int = 512
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    hidden: int = 64

    # Exploration mixture: with prob epsilon, override policy action.
    explore_epsilon: float = 0.0
    explore_epsilon_anneal_to: float = 0.0
    explore_epsilon_anneal_frac: float = 0.5

    # Reward shaping (paper-standard anneal-to-zero).
    shaped_reward_weight: float = 1.0
    shaped_reward_anneal_frac: float = 0.5

    # Logging / saving / misc.
    log_interval: int = 1
    save_interval: int = 50
    save_dir: str = "runs/Standard PPO"
    run_name: str = ""
    seed: int = 0
    device: str = "cuda"

# ===========================================================================
# Env helpers
# ===========================================================================
def make_env(cfg: Config) -> OvercookedEnv:
    mdp = OvercookedGridworld.from_layout_name(cfg.layout)
    return OvercookedEnv.from_mdp(mdp, horizon=cfg.horizon, info_level=0)


def get_obs_dim(env: OvercookedEnv) -> int:
    obs0, _ = env.featurize_state_mdp(env.state)
    return int(obs0.shape[0])

# ===========================================================================
# Rollout (self-play)
# ===========================================================================
@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

def collect_rollout(
    env: OvercookedEnv,
    policy: ActorCritic,
    cfg: Config,
    explore_fn: Optional[ExplorationFn],
    shaped_w: float,
    device: torch.device,
):
    """One full self-play episode. Both agents share `policy`.
    Each timestep yields 2 transitions (agent-0 view + agent-1 view).
    """
    env.reset()
    buf = RolloutBuffer()
    ep_sparse = 0.0
    ep_shaped = 0.0

    for _ in range(cfg.horizon):
        obs0, obs1 = env.featurize_state_mdp(env.state)
        obs_pair = np.stack([obs0, obs1], axis=0).astype(np.float32)
        obs_t = torch.from_numpy(obs_pair).to(device)

        with torch.no_grad():
            logits, values = policy(obs_t)
            dist = Categorical(logits=logits)
            policy_actions = dist.sample()

        chosen: List[int] = []
        for i in range(2):
            override = (
                explore_fn(obs_pair[i], i) if explore_fn is not None else None
            )
            chosen.append(
                int(override) if override is not None else int(policy_actions[i].item())
            )

        joint_action = tuple(Action.ALL_ACTIONS[a] for a in chosen)
        _, _, done, info = env.step(joint_action)

        sparse_by_agent = info["sparse_r_by_agent"]
        shaped_by_agent = info["shaped_r_by_agent"]
        rewards = [
            float(sparse_by_agent[i]) + shaped_w * float(shaped_by_agent[i])
            for i in range(2)
        ]
        ep_sparse += float(sum(sparse_by_agent))
        ep_shaped += float(sum(shaped_by_agent))

        # log_prob of the executed action under the *policy*.
        chosen_t = torch.tensor(chosen, dtype=torch.long, device=device)
        logp_chosen = dist.log_prob(chosen_t)

        for i in range(2):
            buf.obs.append(obs_pair[i])
            buf.actions.append(chosen[i])
            buf.logprobs.append(float(logp_chosen[i].item()))
            buf.values.append(float(values[i].item()))
            buf.rewards.append(rewards[i])
            buf.dones.append(bool(done))

        if done:
            break

    # Bootstrap value at the final state (0 if true terminal; here horizon-truncated).
    obs0, obs1 = env.featurize_state_mdp(env.state)
    last_obs = torch.from_numpy(
        np.stack([obs0, obs1], axis=0).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        last_values = policy.value(last_obs).cpu().numpy()

    return buf, ep_sparse, ep_shaped, last_values

def compute_gae(
    buf: RolloutBuffer,
    last_values: np.ndarray,
    gamma: float,
    lam: float,
):
    """Per-agent GAE. Buffer rows interleave (a0, a1, a0, a1, ...)."""
    n = len(buf.rewards)
    T = n // 2
    adv = np.zeros(n, dtype=np.float32)
    ret = np.zeros(n, dtype=np.float32)
    for agent_idx in range(2):
        gae = 0.0
        next_v = float(last_values[agent_idx])
        for t in reversed(range(T)):
            i = t * 2 + agent_idx
            r = buf.rewards[i]
            v = buf.values[i]
            nonterminal = 1.0 - float(buf.dones[i])
            delta = r + gamma * next_v * nonterminal - v
            gae = delta + gamma * lam * nonterminal * gae
            adv[i] = gae
            ret[i] = gae + v
            next_v = v
    return adv, ret

# ===========================================================================
# PPO update
# ===========================================================================
def ppo_update(policy, optimizer, batch, cfg: Config):
    obs = batch["obs"]
    actions = batch["actions"]
    old_lp = batch["logprobs"]
    returns = batch["returns"]
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = obs.shape[0]
    idx = np.arange(n)
    last_stats = {}

    for _ in range(cfg.update_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, cfg.minibatch_size):
            mb = idx[start : start + cfg.minibatch_size]
            mb_obs = obs[mb]
            mb_act = actions[mb]
            mb_old_lp = old_lp[mb]
            mb_ret = returns[mb]
            mb_adv = adv[mb]

            logits, values = policy(mb_obs)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(mb_act)
            entropy = dist.entropy().mean()

            ratio = (new_lp - mb_old_lp).exp()
            with torch.no_grad():
                logratio = new_lp - mb_old_lp
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = ((values - mb_ret) ** 2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            last_stats = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "approx_kl": float(approx_kl.item()),
                "clip_frac": float(clip_frac.item()),
            }
    return last_stats

# ===========================================================================
# Metrics logging
#
# Three files are written into the run directory (next to the checkpoints):
#   config.json   -- hyperparameters + hardware, written once at startup.
#   metrics.jsonl -- one JSON line per logged iteration (for plotting curves).
#   summary.json  -- final totals + wall-clock time, written when training ends.
# ===========================================================================
SOUP_REWARD = 20.0  # sparse reward for delivering one onion soup (Overcooked default)
ACTION_NAMES = ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"]

def hardware_info(device: torch.device) -> dict:
    """Hardware / library versions."""
    info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        info["gpu_count"] = n
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(n)]
    else:
        info["gpu_count"] = 0
        info["gpu_names"] = []
    return info

def write_run_config(save_path: str, cfg: Config, device: torch.device):
    """Dump information for reproducability."""
    config = {
        "run_name": os.path.basename(save_path),
        "explore_mode": "epsilon_random",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_actions": NUM_ACTIONS,
        "action_names": ACTION_NAMES,
        "hyperparameters": vars(cfg),
        "hardware": hardware_info(device),
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

def append_metrics(
    save_path, iteration, steps, wall_time, iter_time, sps,
    ep_sparses, ep_shapeds, stats, shaped_w, eps, extra=None,
):
    """Append one row to metrics.jsonl. Returns the record (for summary.json)."""
    sparse = np.asarray(ep_sparses, dtype=np.float64)
    shaped = np.asarray(ep_shapeds, dtype=np.float64)
    metrics = {
        # Reward / score -- mean and std across the episodes in this update.
        "reward_mean_sparse": float(sparse.mean()),
        "reward_std_sparse": float(sparse.std()),
        "reward_mean_shaped": float(shaped.mean()),
        "reward_std_shaped": float(shaped.std()),
        # Task success: soups delivered per episode, and the fraction of
        # episodes that delivered at least one soup (success rate).
        "soups_per_episode": float(sparse.mean() / SOUP_REWARD),
        "success_rate": float((sparse > 0.0).mean()),
        # PPO training diagnostics.
        "policy_loss": stats["policy_loss"],
        "value_loss": stats["value_loss"],
        "entropy": stats["entropy"],
        "approx_kl": stats.get("approx_kl"),
        "clip_frac": stats.get("clip_frac"),
        # Schedules / throughput.
        "shaped_weight": float(shaped_w),
        "epsilon": float(eps),
        "steps_per_sec": round(float(sps), 1),
        "n_episodes": int(sparse.size),
    }
    if extra:
        metrics.update(extra)
    record = {
        "iteration": int(iteration),
        "steps": int(steps),
        "wall_time_sec": round(float(wall_time), 2),  # cumulative since start
        "iter_time_sec": round(float(iter_time), 3),  # this update only
        "metrics": metrics,
    }
    with open(os.path.join(save_path, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    return record

def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"

def write_summary(save_path, cfg, steps_done, update_count, total_time, last_record):
    """Final one-shot summary: totals + wall-clock time + last metrics row."""
    summary = {
        "run_name": os.path.basename(save_path),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_steps": int(steps_done),
        "total_iterations": int(update_count),
        "total_wall_time_sec": round(float(total_time), 2),
        "total_wall_time_human": _fmt_hms(total_time),
        "mean_iter_time_sec": round(total_time / max(update_count, 1), 3),
        "final_metrics": last_record["metrics"] if last_record else None,
    }
    with open(os.path.join(save_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

# ===========================================================================
# Main
# ===========================================================================
def main(cfg: Config):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(
        cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"device={device}")

    env = make_env(cfg)
    obs_dim = get_obs_dim(env)
    print(f"layout={cfg.layout}  obs_dim={obs_dim}  n_actions={NUM_ACTIONS}")

    policy = ActorCritic(obs_dim, hidden=cfg.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    run_name = cfg.run_name or f"ppo_{cfg.layout}_{int(time.time())}"
    save_path = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"saving to {save_path}")
    write_run_config(save_path, cfg, device)

    steps_done = 0
    update_count = 0
    last_record = None
    start = time.time()

    while steps_done < cfg.total_steps:
        iter_start = time.time()
        frac = steps_done / cfg.total_steps
        # Anneal shaped reward weight.
        shaped_w = cfg.shaped_reward_weight * max(
            0.0, 1.0 - frac / max(cfg.shaped_reward_anneal_frac, 1e-9)
        )
        # Anneal exploration epsilon.
        eps_progress = min(1.0, frac / max(cfg.explore_epsilon_anneal_frac, 1e-9))
        current_eps = (
            cfg.explore_epsilon
            + (cfg.explore_epsilon_anneal_to - cfg.explore_epsilon) * eps_progress
        )
        explore_fn = make_explore_fn(current_eps)

        # Collect.
        all_obs, all_act, all_lp = [], [], []
        all_adv, all_ret = [], []
        ep_sparses, ep_shapeds = [], []
        for _ in range(cfg.n_rollouts_per_update):
            buf, ep_sparse, ep_shaped, last_v = collect_rollout(
                env, policy, cfg, explore_fn, shaped_w, device
            )
            adv, ret = compute_gae(buf, last_v, cfg.gamma, cfg.gae_lambda)
            all_obs.append(np.array(buf.obs))
            all_act.append(np.array(buf.actions))
            all_lp.append(np.array(buf.logprobs))
            all_adv.append(adv)
            all_ret.append(ret)
            steps_done += len(buf.rewards) // 2
            ep_sparses.append(ep_sparse)
            ep_shapeds.append(ep_shaped)

        batch = {
            "obs": torch.from_numpy(np.concatenate(all_obs)).float().to(device),
            "actions": torch.from_numpy(np.concatenate(all_act)).long().to(device),
            "logprobs": torch.from_numpy(np.concatenate(all_lp)).float().to(device),
            "returns": torch.from_numpy(np.concatenate(all_ret)).float().to(device),
            "advantages": torch.from_numpy(np.concatenate(all_adv)).float().to(device),
        }
        stats = ppo_update(policy, optimizer, batch, cfg)
        update_count += 1

        if update_count % cfg.log_interval == 0:
            elapsed = time.time() - start
            sps = steps_done / max(elapsed, 1e-9)
            print(
                f"upd={update_count:4d}  steps={steps_done:8d}  "
                f"sparse_mean={np.mean(ep_sparses):6.2f}  "
                f"shaped_mean={np.mean(ep_shapeds):7.2f}  "
                f"shaped_w={shaped_w:.2f}  eps={current_eps:.3f}  "
                f"pi_loss={stats['policy_loss']:+.4f}  "
                f"v_loss={stats['value_loss']:.3f}  "
                f"ent={stats['entropy']:.3f}  sps={sps:.0f}"
            )
            last_record = append_metrics(
                save_path, update_count, steps_done, elapsed,
                time.time() - iter_start, sps,
                ep_sparses, ep_shapeds, stats, shaped_w, current_eps,
            )

        if cfg.save_interval > 0 and update_count % cfg.save_interval == 0:
            ckpt = os.path.join(save_path, f"ckpt_{update_count}.pt")
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps_done": steps_done,
                    "update_count": update_count,
                    "config": vars(cfg),
                },
                ckpt,
            )

    torch.save(
        {
            "policy": policy.state_dict(),
            "config": vars(cfg),
            "steps_done": steps_done,
        },
        os.path.join(save_path, "final.pt"),
    )
    write_summary(save_path, cfg, steps_done, update_count, time.time() - start, last_record)
    print(f"done. final checkpoint -> {save_path}/final.pt")
    print(f"metrics -> {save_path}/metrics.jsonl  |  summary -> {save_path}/summary.json")


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    defaults = Config()
    for f in defaults.__dataclass_fields__.values():
        t = type(getattr(defaults, f.name))
        if t is bool:
            p.add_argument(f"--{f.name}", action="store_true")
        else:
            p.add_argument(f"--{f.name}", type=t, default=getattr(defaults, f.name))
    args = p.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    main(parse_args())
