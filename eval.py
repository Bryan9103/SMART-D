"""
Evaluate a trained Overcooked-AI self-play policy.

Loads a checkpoint (final.pt or ckpt_*.pt), runs N episodes with no
exploration (no epsilon, no LLM), and reports mean / std of score, soups
delivered, shaped reward, episode length and success rate. Optionally
renders one episode to PNG frames + a GIF (the game interface).

Usage:
    python eval.py runs/<run_name>/final.pt [--episodes 50 --render]
    python eval.py runs/ppo_cramped_room_xxx/ckpt_300.pt [--episodes 50 --render]

Outputs (next to the checkpoint, under <run>/eval/):
    eval/<ckpt>.json            -- stats + per-episode raw numbers
    eval/render_<ckpt>/frames/  -- one PNG per timestep
    eval/render_<ckpt>/episode.gif
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

# Render headlessly -- no display needed for pygame.image.save.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
from torch.distributions import Categorical

from overcooked_ai_py.mdp.actions import Action

from train_ppo import (
    NUM_ACTIONS,
    SOUP_REWARD,
    ActorCritic,
    Config,
    get_obs_dim,
    make_env,
)

SMART_D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SMART-D")

# ===========================================================================
# Checkpoint loading
# ===========================================================================
def _load_train_ppo(ckpt: dict, device: torch.device, ckpt_path: str):
    """Original train_ppo.py checkpoint (key 'policy', Config dataclass)."""
    cfg_fields = set(Config().__dataclass_fields__)
    cfg_dict = {k: v for k, v in ckpt.get("config", {}).items() if k in cfg_fields}
    cfg = Config(**cfg_dict)

    env = make_env(cfg)
    obs_dim = get_obs_dim(env)
    policy = ActorCritic(obs_dim, hidden=cfg.hidden).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    def act_fn(env, deterministic: bool):
        obs0, obs1 = env.featurize_state_mdp(env.state)
        obs = torch.from_numpy(np.stack([obs0, obs1]).astype(np.float32)).to(device)
        with torch.no_grad():
            logits, _ = policy(obs)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = Categorical(logits=logits).sample()
        return tuple(Action.ALL_ACTIONS[int(a)] for a in actions)

    steps_done = int(ckpt.get("steps_done", 0))
    if "SMART-D" in ckpt_path.lower():
        kind = "SMART-D"
    else:
        kind = "Standard_PPO"

    return act_fn, cfg, env, steps_done, kind

def _load_SMART_D(ckpt: dict, device: torch.device):
    """SMART-D checkpoint (key 'controller', plain-dict YAML config).

    Eval feeds a zero subgoal -- matches the no-LLM deployment setting
    """
    if SMART_D_DIR not in sys.path:
        sys.path.insert(0, SMART_D_DIR)
    from src.controller import SUBGOAL_DIM, Controller  # type: ignore

    raw_cfg = dict(ckpt.get("config", {}) or {})
    method = str(raw_cfg.get("method", "SMART-D"))
    cfg = SimpleNamespace(
        layout=str(raw_cfg.get("layout", "cramped_room")),
        horizon=int(raw_cfg.get("horizon", 400)),
        hidden=int(raw_cfg.get("hidden", 64)),
        method=method,
    )

    env = make_env(cfg)
    obs_dim = get_obs_dim(env)
    controller = Controller(
        state_dim=obs_dim,
        hidden=cfg.hidden,
    ).to(device)
    controller.load_state_dict(ckpt["controller"])
    controller.eval()

    # Zero subgoal for both agents -- shape (2, 16).
    zero_subgoal = torch.zeros(2, SUBGOAL_DIM, dtype=torch.float32, device=device)

    def act_fn(env, deterministic: bool):
        obs0, obs1 = env.featurize_state_mdp(env.state)
        obs = torch.from_numpy(np.stack([obs0, obs1]).astype(np.float32)).to(device)
        with torch.no_grad():
            actions, _, _ = controller.act(obs, zero_subgoal, deterministic=deterministic)
        return tuple(Action.ALL_ACTIONS[int(a)] for a in actions)

    steps_done = int(ckpt.get("global_steps", 0))
    return act_fn, cfg, env, steps_done, "SMART-D"


def load_policy(ckpt_path: str, device: torch.device):
    """Dispatch on checkpoint flavor. Returns (act_fn, cfg, env, steps_done, kind).

    act_fn(env, deterministic) -> joint Action tuple. The closure hides the
    network details (one-arg ActorCritic vs. (state, subgoal) Controller).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "controller" in ckpt:
        return _load_SMART_D(ckpt, device)
    if "policy" in ckpt:
        return _load_train_ppo(ckpt, device, ckpt_path)
    raise KeyError(
        f"Checkpoint at {ckpt_path} has neither 'policy' (train_ppo.py) nor "
        f"'controller' (SMART-D) at the top level. Keys: {list(ckpt.keys())}"
    )

# ===========================================================================
# Rollout (no exploration -- pure policy self-play)
# ===========================================================================
def run_episode(env, act_fn, deterministic: bool):
    """One self-play episode. Returns (sparse_reward, shaped_reward, length)."""
    env.reset()
    sparse = shaped = 0.0
    length = 0
    for _ in range(env.horizon):
        joint = act_fn(env, deterministic)
        _, _, done, info = env.step(joint)
        sparse += float(sum(info["sparse_r_by_agent"]))
        shaped += float(sum(info["shaped_r_by_agent"]))
        length += 1
        if done:
            break
    return sparse, shaped, length

# ===========================================================================
# Stats
# ===========================================================================
def summarize(values) -> dict:
    """mean / std / sem / min / max for a list of per-episode numbers."""
    a = np.asarray(values, dtype=np.float64)
    n = max(a.size, 1)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "sem": float(a.std() / np.sqrt(n)),  # std error of the mean -> error bars
        "min": float(a.min()),
        "max": float(a.max()),
    }

# ===========================================================================
# Rendering (the game interface)
# ===========================================================================
def _encode_video(pattern: str, out_dir: str, fps: int) -> dict:
    """Stitch PNG frames into a (palette-optimized) GIF and an MP4."""
    quiet = ["-loglevel", "error"]
    gif_path = os.path.join(out_dir, "episode.gif")
    mp4_path = os.path.join(out_dir, "episode.mp4")

    # GIF -- two-pass palette keeps it small and sharp (vs. ffmpeg's default).
    palette = os.path.join(out_dir, "_palette.png")
    try:
        subprocess.run(
            ["ffmpeg", "-y", *quiet, "-framerate", str(fps), "-i", pattern,
             "-vf", "palettegen=stats_mode=diff", palette],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", *quiet, "-framerate", str(fps), "-i", pattern,
             "-i", palette, "-lavfi", "paletteuse=dither=bayer", gif_path],
            check=True,
        )
    finally:
        if os.path.exists(palette):
            os.remove(palette)

    # MP4 -- h264, even dimensions, broadly playable (good for slides).
    subprocess.run(
        ["ffmpeg", "-y", *quiet, "-framerate", str(fps), "-i", pattern,
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", mp4_path],
        check=True,
    )
    return {"gif": gif_path, "mp4": mp4_path}


def render_episode(
    env, act_fn, deterministic: bool, out_dir: str, fps: int, max_steps: int
):
    """Render one episode to PNG frames; stitch GIF + MP4 with ffmpeg if available."""
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    viz = StateVisualizer()
    grid = env.mdp.terrain_mtx

    env.reset()
    score = 0.0
    n_frames = 0
    limit = env.horizon if max_steps <= 0 else min(max_steps, env.horizon)

    def snap(idx: int):
        hud = StateVisualizer.default_hud_data(env.state, score=int(score))
        viz.display_rendered_state(
            env.state,
            grid=grid,
            hud_data=hud,
            img_path=os.path.join(frames_dir, f"frame_{idx:04d}.png"),
        )

    for _ in range(limit):
        snap(n_frames)
        n_frames += 1
        joint = act_fn(env, deterministic)
        _, _, done, info = env.step(joint)
        score += float(sum(info["sparse_r_by_agent"]))
        if done:
            break
    snap(n_frames)  # final state
    n_frames += 1

    pattern = os.path.join(frames_dir, "frame_%04d.png")
    if shutil.which("ffmpeg"):
        videos = _encode_video(pattern, out_dir, fps)
    else:
        videos = {"gif": None, "mp4": None}
        print("  (ffmpeg not found -- kept PNG frames only)")

    return n_frames, score, frames_dir, videos

# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="path to a .pt checkpoint or a run directory")
    p.add_argument("--episodes", type=int, default=20, help="evaluation episodes")
    p.add_argument(
        "--deterministic", action="store_true",
        help="argmax actions (note: env is deterministic, so all episodes are "
        "identical -- use 1 episode). Default: sample from the policy.",
    )
    p.add_argument("--layout", default="", help="override layout (default: checkpoint's)")
    p.add_argument("--horizon", type=int, default=0, help="override horizon")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="render one episode -> PNG + GIF + MP4")
    p.add_argument("--render_fps", type=int, default=8, help="GIF/MP4 frames per second")
    p.add_argument(
        "--render_max_steps", type=int, default=0,
        help="cap the rendered clip length (0 = full episode)",
    )
    p.add_argument("--out", default="", help="output dir (default: <run>/eval/)")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        for name in ("best.pt", "latest.pt", "final.pt"):
            cand = os.path.join(ckpt_path, name)
            if os.path.isfile(cand):
                ckpt_path = cand
                break
        else:
            raise FileNotFoundError(
                f"No best.pt / latest.pt / final.pt found in {ckpt_path}"
            )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    act_fn, cfg, env, steps_done, kind = load_policy(ckpt_path, device)
    if args.layout:
        cfg.layout = args.layout
    if args.horizon:
        cfg.horizon = args.horizon
    if args.layout or args.horizon:
        env = make_env(cfg)

    mode = "deterministic" if args.deterministic else "stochastic"
    print(
        f"checkpoint={ckpt_path}\n"
        f"kind={kind}  layout={cfg.layout}  horizon={cfg.horizon}  "
        f"steps_done={steps_done}  device={device}  mode={mode}  "
        f"episodes={args.episodes}"
    )
    
    # --- evaluation episodes -------------------------------------------------
    t0 = time.time()
    scores, shapeds, lengths = [], [], []
    for ep in range(args.episodes):
        sparse, shaped, length = run_episode(env, act_fn, args.deterministic)
        scores.append(sparse)
        shapeds.append(shaped)
        lengths.append(length)
    eval_time = time.time() - t0

    scores_np = np.asarray(scores, dtype=np.float64)
    soups = scores_np / SOUP_REWARD
    success_rate = float((scores_np > 0.0).mean())

    result = {
        "checkpoint": os.path.abspath(ckpt_path),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "steps_done": steps_done,
        "layout": cfg.layout,
        "horizon": cfg.horizon,
        "episodes": args.episodes,
        "action_mode": mode,
        "eval_time_sec": round(eval_time, 2),
        "score": summarize(scores),               # sparse reward (= soups * 20)
        "soups_delivered": summarize(soups),
        "shaped_reward": summarize(shapeds),
        "episode_length": summarize(lengths),
        "success_rate": success_rate,             # fraction of episodes w/ >=1 soup
        "per_episode": {
            "score": [float(x) for x in scores],
            "soups": [float(x) for x in soups],
            "shaped_reward": [float(x) for x in shapeds],
            "length": [int(x) for x in lengths],
        },
    }

    # --- output dir ----------------------------------------------------------
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    eval_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # --- rendering -----------------------------------------------------------
    if args.render:
        render_dir = os.path.join(eval_dir, f"render_{stem}")
        print(f"rendering one episode -> {render_dir}/")
        n_frames, render_score, frames_dir, videos = render_episode(
            env, act_fn, args.deterministic, render_dir,
            args.render_fps, args.render_max_steps,
        )
        result["render"] = {
            "frames_dir": frames_dir,
            "gif": videos["gif"],
            "mp4": videos["mp4"],
            "n_frames": n_frames,
            "episode_score": render_score,
        }
        print(f"  {n_frames} frames; gif -> {videos['gif']}; mp4 -> {videos['mp4']}")

    stats_path = os.path.join(eval_dir, f"{stem}.json")
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)

    # --- report --------------------------------------------------------------
    sc, so = result["score"], result["soups_delivered"]
    print(
        "\n=== evaluation ===\n"
        f"  score (sparse reward) : {sc['mean']:.2f} +/- {sc['std']:.2f}  "
        f"(sem {sc['sem']:.2f}, min {sc['min']:.0f}, max {sc['max']:.0f})\n"
        f"  soups delivered       : {so['mean']:.2f} +/- {so['std']:.2f}\n"
        f"  shaped reward         : {result['shaped_reward']['mean']:.2f} "
        f"+/- {result['shaped_reward']['std']:.2f}\n"
        f"  episode length        : {result['episode_length']['mean']:.1f}\n"
        f"  success rate          : {success_rate * 100:.1f}%\n"
        f"  stats -> {stats_path}"
    )


if __name__ == "__main__":
    main()
