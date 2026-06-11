"""OvercookedVecEnv: vectorised self-play wrapper around OvercookedEnv."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

NUM_ACTIONS = len(Action.ALL_ACTIONS)  # 6

class OvercookedVecEnv:
    """n_envs independent Overcooked episodes running in lockstep.

    reset() -> (n_envs, 2, state_dim)
    step(actions: (n_envs, 2) int) -> states, rewards, dones, infos
    get_raw_states() -> List[OvercookedState]   (for LLM planner)
    """

    def __init__(self, layout_name: str = "cramped_room", n_envs: int = 8, horizon: int = 400) -> None:
        self.layout_name = layout_name
        self.n_envs = n_envs
        self.horizon = horizon
        self.action_dim = NUM_ACTIONS

        self._mdps: List[OvercookedGridworld] = []
        self._envs: List[OvercookedEnv] = []
        for _ in range(n_envs):
            mdp = OvercookedGridworld.from_layout_name(
                layout_name, rew_shaping_params=BASE_REW_SHAPING_PARAMS
            )
            env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
            self._mdps.append(mdp)
            self._envs.append(env)

        self._envs[0].reset()
        obs0, _ = self._envs[0].featurize_state_mdp(self._envs[0].state)
        self.state_dim: int = int(obs0.shape[0])

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all envs. Returns states (n_envs, 2, state_dim)."""
        states = []
        infos = []
        for env in self._envs:
            env.reset()
            obs0, obs1 = env.featurize_state_mdp(env.state)
            states.append(np.stack([obs0, obs1], axis=0))
            infos.append({})
        return np.stack(states, axis=0).astype(np.float32), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all envs.

        actions: (n_envs, 2) int
        returns:
          states  (n_envs, 2, state_dim)
          rewards (n_envs,)  shared reward (sum over agents)
          dones   (n_envs,)  bool
          infos   list of dicts with shaped_r_by_agent, sparse_r_by_agent
        """
        actions = np.asarray(actions, dtype=np.int64)
        next_states, rewards, dones, infos = [], [], [], []

        for i, env in enumerate(self._envs):
            a0, a1 = int(actions[i, 0]), int(actions[i, 1])
            joint = (Action.ALL_ACTIONS[a0], Action.ALL_ACTIONS[a1])
            _, sparse_r, done, info = env.step(joint)

            if done:
                env.reset()

            obs0, obs1 = env.featurize_state_mdp(env.state)
            next_states.append(np.stack([obs0, obs1], axis=0))
            rewards.append(float(sparse_r))
            dones.append(bool(done))
            infos.append(info)

        return (
            np.stack(next_states, axis=0).astype(np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def get_raw_states(self) -> list:
        """Return current OvercookedState for each env (for LLM planner)."""
        return [env.state for env in self._envs]

    def get_mdps(self) -> List[OvercookedGridworld]:
        return self._mdps
