"""LLM Planner: wraps an OpenAI-compatible server (Ollama / vLLM) to produce subgoals.

Optimisations for practical training speed:
  1. In-memory LRU cache keyed on state summary hash — repeated states skip the LLM.
  2. ThreadPoolExecutor for parallel batch queries (all envs queried simultaneously).
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

VALID_SUBGOALS = ["get_onion", "put_onion", "get_dish", "pick_soup", "deliver",
                  "place_on_counter", "pickup_from_counter", "idle"]

_SYSTEM_PROMPT = """\
You are a cooperative cooking assistant planner for the Overcooked game.
Given the current game state, assign a high-level subgoal to each player.
Reply with a JSON object only — no extra text.

Valid subgoals:
- get_onion: go pick up an onion from a dispenser
- put_onion: place held onion into a pot
- get_dish: go pick up a dish from a dispenser
- pick_soup: use held dish to pick up ready soup from a pot
- deliver: bring held soup to a serving counter
- place_on_counter: place held item on a counter for partner to collect (use when direct delivery is blocked)
- pickup_from_counter: go collect an item your partner left on a counter
- idle: no useful action available right now

JSON schema:
{
  "p1_subgoal": "<subgoal>",
  "p2_subgoal": "<subgoal>",
  "urgency": <float 0-1>,
  "reason": "<one sentence>"
}

Rules:
- If a player holds soup, assign deliver.
- If soup is ready and a player holds a dish, assign pick_soup.
- If a pot is cooking and nobody has a dish, assign get_dish to someone.
- If a pot needs onions, assign get_onion / put_onion as appropriate.
- If agents need to exchange items but cannot reach each other directly, use place_on_counter and pickup_from_counter.
- Avoid both players idling unless truly nothing useful exists.
- Assign complementary subgoals: avoid both players doing the same thing.
"""

_FALLBACK_JSON: Dict[str, Any] = {
    "p1_subgoal": "idle",
    "p2_subgoal": "idle",
    "urgency": 0.5,
    "reason": "fallback",
}

def _sanitize(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["p1_subgoal"] = raw.get("p1_subgoal", "idle") if raw.get("p1_subgoal") in VALID_SUBGOALS else "idle"
    out["p2_subgoal"] = raw.get("p2_subgoal", "idle") if raw.get("p2_subgoal") in VALID_SUBGOALS else "idle"
    try:
        u = float(raw.get("urgency", 0.5))
    except (TypeError, ValueError):
        u = 0.5
    out["urgency"] = float(max(0.0, min(1.0, u)))
    out["reason"] = str(raw.get("reason", ""))
    return out

def _summary_hash(summary: str) -> str:
    return hashlib.md5(summary.encode("utf-8")).hexdigest()

class LLMPlanner:
    """Query an OpenAI-compatible server for subgoal assignments.

    Uses an in-process LRU cache and a thread pool for parallel batch queries.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.3,
        max_tokens: int = 200,
        timeout: float = 60.0,
        cache_size: int = 2048,
        max_workers: int = 8,
    ) -> None:
        import httpx
        from openai import OpenAI
        import os

        self._client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("CUSTOM_LLM_KEY", "dummy"),
            http_client=httpx.Client(timeout=httpx.Timeout(timeout, connect=2.0)),
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_workers = max_workers

        # in-memory LRU cache: hash → sanitised dict
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_size = cache_size

        self.llm_call_count: int = 0
        self.cache_hits: int = 0
        self.malformed_count: int = 0
        self._latencies: List[float] = []

        # persistent pool — avoids re-creating threads on every query_batch call
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies) * 1000.0

    @property
    def malformed_rate(self) -> float:
        if self.llm_call_count == 0:
            return 0.0
        return self.malformed_count / self.llm_call_count

    @property
    def cache_hit_rate(self) -> float:
        total = self.llm_call_count + self.cache_hits
        return self.cache_hits / total if total > 0 else 0.0

    def _call_llm(self, state_summary: str) -> Dict[str, Any]:
        """One LLM HTTP call (no cache). Records latency."""
        self.llm_call_count += 1
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": state_summary},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            raw_text = resp.choices[0].message.content or "{}"
            raw = json.loads(raw_text)
            result = _sanitize(raw)
        except Exception:
            self.malformed_count += 1
            result = dict(_FALLBACK_JSON)
        self._latencies.append(time.time() - t0)
        return result

    def query_one(self, state_summary: str) -> Dict[str, Any]:
        """Return subgoal for one state, using cache when available."""
        key = _summary_hash(state_summary)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        result = self._call_llm(state_summary)
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result

    def query_batch(self, summaries: List[str]) -> List[Dict[str, Any]]:
        """Query multiple summaries in parallel using a thread pool.

        Summaries that hit the cache are resolved immediately without
        occupying a thread.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(summaries)
        pending: List[int] = []

        # serve cache hits first
        for i, s in enumerate(summaries):
            key = _summary_hash(s)
            if key in self._cache:
                self.cache_hits += 1
                results[i] = self._cache[key]
            else:
                pending.append(i)

        if not pending:
            return results  # type: ignore[return-value]

        # parallel LLM calls for cache misses (reuse persistent pool)
        future_to_idx = {self._pool.submit(self._call_llm, summaries[i]): i for i in pending}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            res = future.result()
            results[idx] = res
            key = _summary_hash(summaries[idx])
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = res

        return results  # type: ignore[return-value]
