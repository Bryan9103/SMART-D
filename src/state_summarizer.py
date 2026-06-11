"""Convert OvercookedState to a structured text prompt for the LLM planner."""
from __future__ import annotations

from typing import Any


def summarize_state(state: Any, mdp: Any, layout_name: str) -> str:
    """Return a concise natural-language description of the current game state.

    Covers: layout, timestep, player positions/holdings, pot status,
    dispenser and serving locations.
    """
    horizon = getattr(mdp, "horizon", 400)
    t = int(state.timestep)
    remaining = horizon - t

    lines = [
        f"Layout: {layout_name}",
        f"Timestep: {t} / {horizon}  (remaining: {remaining})",
        "",
    ]

    # Players
    for idx, p in enumerate(state.players):
        held = p.held_object.name if p.held_object is not None else "nothing"
        lines.append(f"Player {idx + 1}: position={p.position}, facing={p.orientation}, holding={held}")
    lines.append("")

    # Pots
    pot_locs = mdp.get_pot_locations()
    for pot_pos in pot_locs:
        if state.has_object(pot_pos):
            obj = state.get_object(pot_pos)
            if obj.name == "soup":
                n_onions = sum(1 for ing in obj.ingredients if ing == "onion")
                if obj.is_ready:
                    status = "READY"
                elif obj.is_cooking:
                    status = f"cooking ({obj.cook_time_remaining} steps left)"
                elif obj.is_idle:
                    status = f"idle ({n_onions}/3 onions)"
                else:
                    status = f"filling ({n_onions}/3 onions)"
                lines.append(f"Pot at {pot_pos}: {status}")
        else:
            lines.append(f"Pot at {pot_pos}: empty")
    lines.append("")

    # Static locations
    onion_locs = mdp.get_onion_dispenser_locations()
    dish_locs = mdp.get_dish_dispenser_locations()
    serve_locs = mdp.get_serving_locations()
    lines.append(f"Onion dispensers: {onion_locs}")
    lines.append(f"Dish dispensers:  {dish_locs}")
    lines.append(f"Serving counters: {serve_locs}")

    return "\n".join(lines)
