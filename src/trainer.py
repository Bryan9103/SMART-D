"""Main training loop.

Usage:
    python -m src.trainer configs/SMART-D_cramped.yaml --seed 0
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from .controller import Controller
from .env import OvercookedVecEnv
from .eval import run_eval
from .gae import compute_gae
from .losses import lambda_sd_schedule
from .rollout import collect_rollout
from .update import update_step

from dotenv import load_dotenv
load_dotenv()

print(f"Loaded CUSTOM_LLM_KEY: {'FOUND (Starts with ' + os.environ['CUSTOM_LLM_KEY'][:4] + '...)' if 'CUSTOM_LLM_KEY' in os.environ else 'NOT FOUND'}")

def _load_config(path: str, seed_override: Optional[int] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    return cfg

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _make_run_dir(cfg: Dict[str, Any]) -> Path:
    method = cfg.get("method", "SMART-D")
    layout = cfg.get("layout", "cramped_room")
    seed = cfg.get("seed", 0)
    ts = int(time.time())
    name = cfg.get("run_name") or f"{method}_{layout}_seed{seed}_{ts}"
    run_dir = Path(cfg.get("save_dir", "runs/SMART-D")) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _init_wandb(cfg: Dict[str, Any], run_dir: Path) -> Any:
    try:
        import wandb
        run = wandb.init(
            project=cfg.get("wandb_project", "rltf-overcooked"),
            name=run_dir.name,
            config=cfg,
            dir=str(run_dir),
        )
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}), continuing without WandB")
        return None

def train(cfg: Dict[str, Any], resume_dir: Optional[str] = None) -> Path:
    _set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trainer] device={device}  method={cfg.get('method','SMART-D')}", flush=True)

    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        print(f"[trainer] Resuming from {run_dir}", flush=True)
    else:
        run_dir = _make_run_dir(cfg)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f)

    wandb_run = _init_wandb(cfg, run_dir)

    # -- env --
    method = str(cfg.get("method", "SMART-D"))
    layout = str(cfg.get("layout", "cramped_room"))
    n_envs = int(cfg.get("n_envs", 8))
    horizon = int(cfg.get("horizon", 400))
    rollout_len = int(cfg.get("rollout_len", 400))
    n_iter = int(cfg.get("n_iter", 1000))

    envs = OvercookedVecEnv(layout_name=layout, n_envs=n_envs, horizon=horizon)
    state_dim = envs.state_dim
    print(f"[trainer] layout={layout}  state_dim={state_dim}  n_envs={n_envs}", flush=True)

    # -- controller --
    hidden = int(cfg.get("hidden", 64))
    controller = Controller(
        state_dim=state_dim,
        hidden=hidden,
    ).to(device)

    lr = float(cfg.get("lr", 3e-4))
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr, eps=1e-5)

    # -- LLM planner --
    llm_planner = None
    if method == "SMART-D":
        from .llm_planner import LLMPlanner
        llm_cfg = cfg.get("llm", {})
        _planner = LLMPlanner(
            base_url=str(llm_cfg.get("base_url", "http://localhost:8001/v1")),
            model=str(llm_cfg.get("model", "Qwen/Qwen2.5-1.5B-Instruct")),
            temperature=float(llm_cfg.get("temperature", 0.3)),
            max_tokens=int(llm_cfg.get("max_tokens", 200)),
        )

        print("[trainer] Checking LLM server connectivity...", flush=True)
        try:
            import httpx as _httpx
            _base = str(llm_cfg.get("base_url", "http://localhost:11434/v1"))
            if _base.endswith("/v1"):
                _base = _base[:-3]
            _base = _base.rstrip("/")

            import os
            _api_key = os.environ.get("CUSTOM_LLM_KEY", "dummy")
            _headers = {"Authorization": f"Bearer {_api_key}"} if _api_key != "dummy" else {}
            _r = _httpx.get(_base + "/v1/models", headers=_headers, timeout=3.0)

            _server_ok = _r.status_code == 200
        except Exception:
            _server_ok = False

        if not _server_ok:
            raise RuntimeError(
                "[trainer] CRITICAL ERROR: LLM server is unreachable!"
            )
        else:
            llm_planner = _planner
            print("[trainer] LLM server OK", flush=True)

    # -- PPO hyperparams --
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_eps = float(cfg.get("clip_eps", 0.2))
    value_coef = float(cfg.get("value_coef", 0.5))
    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    k_epochs = int(cfg.get("k_epochs", 4))
    minibatch_size = int(cfg.get("minibatch_size", 256))
    eval_interval = int(cfg.get("eval_interval", 50))

    from .reward_shaping import shaped_weight_schedule

    best_eval_return = -float("inf")
    global_steps = 0
    start_iter = 1

    # -- resume: load checkpoint --
    if resume_dir is not None:
        ckpt_path = run_dir / "latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        controller.load_state_dict(ckpt["controller"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt["iteration"]) + 1
        global_steps = int(ckpt.get("global_steps", 0))
        best_eval_return = float(ckpt.get("eval_metrics", {}).get("eval/return_mean", -float("inf")))
        print(f"[trainer] Loaded checkpoint — resuming from iteration {start_iter}  "
              f"global_steps={global_steps:,}", flush=True)

    for iteration in range(start_iter, n_iter + 1):
        shaped_w = shaped_weight_schedule(iteration, n_iter)
        lambda_sd = lambda_sd_schedule(iteration, n_iter, max_lambda=float(cfg.get("lambda_sd_max", 0.1)))

        # -- rollout --
        buf, last_values = collect_rollout(
            envs=envs,
            controller=controller,
            rollout_len=rollout_len,
            shaped_weight=shaped_w,
            device=device,
            method=method,
            llm_planner=llm_planner,
        )
        global_steps += rollout_len * n_envs

        # -- GAE --
        advantages, returns = compute_gae(
            rewards=buf.rewards,
            values=buf.values,
            dones=buf.dones,
            last_values=last_values,
            gamma=gamma,
            lam=gae_lambda,
        )

        # normalise advantages
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # -- flatten buffers: (T, n_envs, 2, ...) -> (T*n_envs*2, ...) --
        T, E, A = buf.states.shape[:3]
        N = T * E * A

        def _flat(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x.reshape(N, *x.shape[3:]), dtype=torch.float32, device=device)

        states_t = _flat(buf.states)
        subgoals_t = _flat(buf.subgoals)
        actions_t = torch.as_tensor(buf.actions.reshape(N), dtype=torch.long, device=device)
        old_lp_t = torch.as_tensor(buf.log_probs.reshape(N), dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv_flat.reshape(N), dtype=torch.float32, device=device)
        ret_t = _flat(returns)

        # -- update --
        update_metrics = update_step(
            controller=controller,
            optimizer=optimizer,
            states=states_t,
            subgoals=subgoals_t,
            actions=actions_t,
            old_log_probs=old_lp_t,
            advantages=adv_t,
            returns=ret_t.view(N),
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            k_epochs=k_epochs,
            minibatch_size=minibatch_size,
            method=method,
            lambda_sd=lambda_sd,
            iteration=iteration,
            n_total=n_iter,
        )

        # -- logging --
        mean_reward = float(buf.rewards.mean())
        log_data = {
            "train/iteration": iteration,
            "train/global_steps": global_steps,
            "train/mean_reward": mean_reward,
            "train/shaped_weight": shaped_w,
            "train/lambda_sd": lambda_sd,
            "loss/total": update_metrics["loss"],
            "loss/L_ppo": update_metrics["L_ppo"],
            "loss/L_value": update_metrics["L_value"],
            "loss/entropy": update_metrics["entropy"],
            "loss/L_sd": update_metrics["L_sd"],
        }

        if llm_planner is not None:
            log_data["llm/malformed_rate"] = llm_planner.malformed_rate
            log_data["llm/avg_latency_ms"] = llm_planner.avg_latency_ms
            log_data["llm/cache_hit_rate"] = llm_planner.cache_hit_rate

        if wandb_run is not None:
            wandb_run.log(log_data)

        if iteration % 10 == 0:
            print(
                f"[{iteration:4d}/{n_iter}] steps={global_steps:,}  "
                f"reward={mean_reward:.2f}  λ_sd={lambda_sd:.4f}  "
                f"loss={update_metrics['loss']:.4f}  ent={update_metrics['entropy']:.3f}"
            )

        # -- eval & checkpoint --
        if iteration % eval_interval == 0 or iteration == n_iter:
            ckpt = {
                "controller": controller.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "global_steps": global_steps,
                "config": cfg,
            }
            torch.save(ckpt, run_dir / f"ckpt_{iteration}.pt")
            print(f"  [checkpoint] Saved ckpt_{iteration}.pt prior to evaluation.", flush=True)

            eval_metrics = run_eval(
                controller=controller,
                layout_name=layout,
                device=device,
                method=method,
                llm_planner=llm_planner,
                n_episodes=int(cfg.get("eval_episodes", 20)),
                horizon=horizon,
            )
            log_data.update(eval_metrics)
            if wandb_run is not None:
                wandb_run.log(eval_metrics)
            print(
                f"  [eval] return={eval_metrics['eval/return_mean']:.2f}  "
                f"soups={eval_metrics['eval/soups_mean']:.2f}"
                + (f"  gap={eval_metrics.get('eval/distillation_gap', 0):.2f}" if method == "SMART-D" else "")
            )

            ckpt["eval_metrics"] = eval_metrics
            torch.save(ckpt, run_dir / "latest.pt")
            if eval_metrics["eval/return_mean"] > best_eval_return:
                best_eval_return = eval_metrics["eval/return_mean"]
                torch.save(ckpt, run_dir / "best.pt")

    torch.save(
        {"controller": controller.state_dict(), "config": cfg, "global_steps": global_steps},
        run_dir / "final.pt",
    )

    del controller
    del optimizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[trainer] done — artifacts in {run_dir}")
    return run_dir

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a run directory to resume from (loads latest.pt)")
    args = parser.parse_args()

    cfg = _load_config(args.config, seed_override=args.seed)
    train(cfg, resume_dir=args.resume)

if __name__ == "__main__":
    main()
