"""Encode LLM JSON subgoal output into a fixed 18-D float vector.

Dimension layout:
  [0:8]   P1 subgoal one-hot  (8 classes)
  [8:16]  P2 subgoal one-hot  (8 classes)
  [16]    urgency scalar [0, 1]
  [17]    remaining time ratio  (400 - t) / 400
"""
from __future__ import annotations

import numpy as np

SUBGOAL_CLASSES = [
    "get_onion",
    "put_onion",
    "get_dish",
    "pick_soup",
    "deliver",
    "place_on_counter",
    "pickup_from_counter",
    "idle",
]
SUBGOAL_DIM = 18
_N_CLASSES = len(SUBGOAL_CLASSES)  # 8
_IDX = {s: i for i, s in enumerate(SUBGOAL_CLASSES)}

def _subgoal_onehot(name: str) -> np.ndarray:
    vec = np.zeros(_N_CLASSES, dtype=np.float32)
    idx = _IDX.get(str(name).lower().strip(), _IDX["idle"])
    vec[idx] = 1.0
    return vec


def encode_subgoal(llm_json: dict, current_t: int, horizon: int = 400) -> np.ndarray:
    """Convert sanitised LLM JSON to an 18-D vector.

    Expected keys: p1_subgoal, p2_subgoal, urgency.
    Missing keys fall back to safe defaults.
    """
    vec = np.zeros(SUBGOAL_DIM, dtype=np.float32)

    vec[0:8]  = _subgoal_onehot(llm_json.get("p1_subgoal", "idle"))
    vec[8:16] = _subgoal_onehot(llm_json.get("p2_subgoal", "idle"))

    urgency = float(llm_json.get("urgency", 0.5))
    vec[16] = float(np.clip(urgency, 0.0, 1.0))

    vec[17] = float(max(0.0, (horizon - current_t) / horizon))

    return vec


def zero_subgoal() -> np.ndarray:
    """All-zero subgoal used during student-branch forward and eval."""
    return np.zeros(SUBGOAL_DIM, dtype=np.float32)
