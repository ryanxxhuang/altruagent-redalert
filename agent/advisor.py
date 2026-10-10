"""Three Claude layers that command the Red Alert bot from background threads.

The scripted bot (``agent/redalert_bot.py``) plays every tick by itself and
never waits for a model. Above it, three layers run at different speeds:

| Layer     | Default model       | Effort | Every | Decides                                               |
|-----------|---------------------|--------|-------|-------------------------------------------------------|
| strategy  | ``claude-fable-5-1`` | medium | 90 s  | posture, unit focus, defenses, refineries, a game plan |
| tactical  | ``claude-opus-5-5``  | low    | 20 s  | wave size, attack now / retreat, where to attack       |
| fast      | off (rules do it)   | low, thinking off | 4 s, only while fighting | focus-fire target, pull back damaged units |

The fast layer is off by default: the bot's own micro rules pick focus-fire
targets and pull back damaged units 5 times a second with no latency, which
beats any model call at this speed. ``FAST_MODEL=claude-sonnet-5-5`` (or a
Haiku) turns it back on, and its answers then override the rules.

Each layer sees the layer above's latest instructions, so the plan flows
down: Fable's plan → Opus's tactics → Sonnet's micro. A slow or failing
request only means that layer's instructions stay as they were.

Configuration (``.env``): ``ANTHROPIC_API_KEY`` turns the layers on. Per layer
``STRATEGY_MODEL`` / ``TACTICAL_MODEL`` / ``FAST_MODEL`` (``off`` disables
that layer) and ``STRATEGY_EFFORT`` / ``TACTICAL_EFFORT`` / ``FAST_EFFORT``.
The key is never printed or logged.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

# --- schemas -----------------------------------------------------------------------

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "posture": {"type": "string", "enum": ["economy", "balanced", "aggressive", "defensive"]},
        "unit_focus": {"type": "string", "enum": ["tanks", "infantry", "mixed", "anti_armor"]},
        "defenses": {"type": "integer"},
        "max_refineries": {"type": "integer"},
        "plan": {"type": "string"},
    },
    "required": ["posture", "unit_focus", "defenses", "max_refineries", "plan"],
    "additionalProperties": False,
}

TACTICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "attack_size": {"type": "integer"},
        "action": {"type": "string", "enum": ["continue", "attack_now", "retreat"]},
        "target_x": {"type": "integer"},
        "target_y": {"type": "integer"},
        "note": {"type": "string"},
    },
    "required": ["attack_size", "action", "target_x", "target_y", "note"],
    "additionalProperties": False,
}

FAST_SCHEMA = {
    "type": "object",
    "properties": {
        "focus_target_id": {"type": "integer"},
        "pull_back_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["focus_target_id", "pull_back_ids"],
    "additionalProperties": False,
}

GAME_RULES = """Command & Conquer: Red Alert (OpenRA), live 1v1 against another AI agent, real time (25 ticks/s). \
Destroying every enemy building and the MCV wins. Otherwise, at the time limit the higher \
(kills_cost - deaths_cost) wins, then higher assets_value: trade efficiently, keep your army alive when ahead \
late, push hard when behind with time running out. A fast scripted bot executes everything (build order, power, \
harvesters, production, base defense, attack waves, retreats); you steer it."""

STRATEGY_PROMPT = GAME_RULES + """

You are the STRATEGIC commander (called every ~90 s). Set the overall direction:
- posture: economy (more refineries, bigger later waves), balanced, aggressive (smaller earlier waves, no \
retreats), defensive (more static defenses, bigger waves).
- unit_focus: tanks, infantry, mixed, anti_armor (more rocket infantry).
- defenses: static defenses to keep (0-6). max_refineries: 1-5.
- plan: 2-3 sentences for your tactical officer: what to aim for in the next minutes and why (enemy faction, \
what you've seen, economy, the clock).
Reply with the JSON only."""

TACTICAL_PROMPT = GAME_RULES + """

You are the TACTICAL officer (called every ~20 s), following the strategic plan you are given.
- attack_size: combat units needed before the next wave launches (4-24).
- action: continue (let the bot decide), attack_now (launch a wave now with what we have), retreat (pull the \
army back home to regroup).
- target_x, target_y: the cell the next/current wave should attack (e.g. a weak enemy building or expansion you \
know of); -1, -1 to let the bot pick (nearest known enemy building, else the enemy base).
- note: one short sentence for the record.
Reply with the JSON only."""

FAST_PROMPT = """You micro-manage a fight in Command & Conquer: Red Alert (real time) for a scripted bot. You get \
our units near the fight and the visible enemies (id, type, cell, hp %). Reply instantly:
- focus_target_id: the enemy id our units should all shoot first (high value and low hp, or the most dangerous: \
tanks and defenses before infantry, artillery and harvesters are juicy), or -1 for no change.
- pull_back_ids: ids of OUR badly damaged valuable units (hp < 30, tanks/artillery) to pull back home; [] if none.
JSON only."""

# --- defaults: model, effort, interval ------------------------------------------------

LAYERS = {
    # name: (model env, default model, effort env, default effort, every seconds, timeout seconds, max calls)
    "strategy": ("STRATEGY_MODEL", "claude-fable-5-1", "STRATEGY_EFFORT", "medium", 90.0, 150.0, 20),
    "tactical": ("TACTICAL_MODEL", "claude-opus-5-5", "TACTICAL_EFFORT", "low", 20.0, 45.0, 80),
    "fast": ("FAST_MODEL", "off", "FAST_EFFORT", "low", 4.0, 15.0, 400),
}
FIGHT_RADIUS = 14  # the fast layer only runs while enemies are this close to our army
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class Layer:
    """One model on its own timer. ``maybe_run`` starts a background request
    when due; ``handle`` turns the parsed JSON into knobs/commands."""

    def __init__(self, name: str, client: Any, model: str, effort: str, every: float, timeout: float,
                 max_calls: int, system: str, schema: dict, log: Callable[[str], None]):
        self.name, self.model, self.effort = name, model, effort
        self.every, self.timeout, self.max_calls = every, timeout, max_calls
        self.system, self.schema = system, schema
        self._client = client
        self._log = log
        self._busy = False
        self._last = -1e9
        self.calls = 0
        self.failures = 0

    def due(self) -> bool:
        return not self._busy and time.monotonic() - self._last >= self.every and self.calls < self.max_calls

    def run(self, payload: dict, handle: Callable[[dict], None]) -> None:
        self._busy = True
        self._last = time.monotonic()
        self.calls += 1
        threading.Thread(target=self._ask, args=(payload, handle), daemon=True).start()

    def _request_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000 if self.name == "strategy" else 4000,
            "system": self.system,
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}},
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        # Sonnet 5.5: thinking off for the lowest latency (accepted at effort high or below).
        if self.model.startswith("claude-sonnet-5-5") and self.effort in ("low", "medium", "high"):
            kwargs["thinking"] = {"type": "between_tools"}
        return kwargs

    def _ask(self, payload: dict, handle: Callable[[dict], None]) -> None:
        import anthropic

        started = time.monotonic()
        try:
            response = self._client.with_options(timeout=self.timeout).beta.messages.create(
                messages=[{"role": "user", "content": json.dumps(payload, separators=(",", ":"))}],
                **self._request_kwargs(),
            )
            if response.stop_reason == "refusal":
                self._log(f"[{self.name}] request declined; keeping current instructions")
                return
            text = next((b.text for b in response.content if b.type == "text"), "")
            handle(json.loads(text))
            if self.name != "fast" or self.calls % 10 == 1:
                self._log(f"[{self.name}] answered in {time.monotonic() - started:.1f}s ({self.model}, {self.effort})")
        except (anthropic.APIError, ValueError, KeyError, TypeError, StopIteration) as exc:
            self.failures += 1
            self._log(f"[{self.name}] no answer this time ({type(exc).__name__}: {str(exc)[:120]})")
            # Back off after repeated failures (bad model id, no access...).
            if self.failures >= 3:
                self._last = time.monotonic() + self.every * min(self.failures, 10)
        finally:
            self._busy = False


class Commanders:
    """The three layers plus the shared state the bot reads from."""

    def __init__(self, layers: dict[str, Layer], log: Callable[[str], None] = print):
        self._layers = layers
        self._log = log
        self._lock = threading.Lock()
        self._knobs: dict[str, Any] = {}
        self._commands: list[dict] = []
        self._plan = ""
        self._plans: list[str] = []
        self._tactical_note = ""

    # -- read by the bot -------------------------------------------------------------

    def knobs(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._knobs)

    def pop_commands(self) -> list[dict]:
        with self._lock:
            out, self._commands = self._commands, []
        return out

    # -- called on every decision ------------------------------------------------------

    def observe(self, raw: dict) -> None:
        summary = summarize(raw)
        if summary is None:
            return
        strategy = self._layers.get("strategy")
        if strategy and strategy.due():
            strategy.run({"previous_plans": self._plans[-3:], "match": summary}, self._on_strategy)
        tactical = self._layers.get("tactical")
        if tactical and tactical.due():
            tactical.run(
                {"strategic_plan": self._plan, "your_last_note": self._tactical_note, "match": summary,
                 "army": army_view(raw), "known_enemy_buildings": summary["visible_enemy_buildings_cells"]},
                self._on_tactical,
            )
        fast = self._layers.get("fast")
        if fast and fast.due():
            fight = fight_view(raw)
            if fight is not None:
                fight["tactical_note"] = self._tactical_note
                fast.run(fight, self._on_fast)

    # -- answers -------------------------------------------------------------------------

    def _on_strategy(self, a: dict) -> None:
        knobs = {
            "posture": a["posture"] if a.get("posture") in ("economy", "balanced", "aggressive", "defensive") else "balanced",
            "unit_focus": a["unit_focus"] if a.get("unit_focus") in ("tanks", "infantry", "mixed", "anti_armor") else "mixed",
            "defenses": max(0, min(6, int(a.get("defenses", 2)))),
            "max_refineries": max(1, min(5, int(a.get("max_refineries", 3)))),
        }
        plan = str(a.get("plan", ""))[:600]
        with self._lock:
            self._knobs.update(knobs)
            self._plan = plan
        self._plans.append(plan)
        self._log(f"[strategy] {knobs['posture']}, {knobs['unit_focus']}, {knobs['defenses']} defenses, "
                  f"{knobs['max_refineries']} refineries: {plan[:200]}")

    def _on_tactical(self, a: dict) -> None:
        size = max(4, min(24, int(a.get("attack_size", 8))))
        tx, ty = int(a.get("target_x", -1)), int(a.get("target_y", -1))
        action = a.get("action")
        with self._lock:
            self._knobs["attack_size"] = size
            self._knobs["target"] = (tx, ty) if tx >= 0 and ty >= 0 else None
            if action in ("attack_now", "retreat"):
                self._commands.append({"type": action})
        self._tactical_note = str(a.get("note", ""))[:200]
        self._log(f"[tactical] {action}, waves of {size}, target {(tx, ty) if tx >= 0 else 'auto'}: {self._tactical_note}")

    def _on_fast(self, a: dict) -> None:
        cmds = []
        tid = a.get("focus_target_id")
        if isinstance(tid, int) and tid >= 0:
            cmds.append({"type": "focus", "target": tid, "until": time.monotonic() + 6.0})
        ids = [i for i in a.get("pull_back_ids") or [] if isinstance(i, int)][:20]
        if ids:
            cmds.append({"type": "pull_back", "units": ids})
        if cmds:
            with self._lock:
                self._commands.extend(cmds)


# --- views sent to the models --------------------------------------------------------------

def _alive(actors: Any) -> list[dict]:
    return [a for a in actors or [] if isinstance(a, dict) and not a.get("wreck") and not str(a.get("type", "")).endswith("husk")]


def _types(actors: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in _alive(actors):
        t = str(a.get("type"))
        out[t] = out.get(t, 0) + 1
    return out


def summarize(raw: dict) -> dict | None:
    obs = raw.get("observation")
    if not isinstance(obs, dict):
        return None
    events = [
        {k: v for k, v in e.items() if k in ("kind", "type", "tick")}
        for e in (obs.get("events") or [])[-10:]
        if isinstance(e, dict)
    ]
    return {
        "time": raw.get("time"),
        "map": {k: (raw.get("map") or {}).get(k) for k in ("id", "width", "height", "bounds")},
        "faction": obs.get("faction"),
        "enemy_faction": obs.get("enemy_faction"),
        "base_center": obs.get("base_center"),
        "enemy_start": (raw.get("legal_actions") or {}).get("enemy_base"),
        "economy": obs.get("economy"),
        "military": obs.get("military"),
        "my_units": _types(obs.get("units")),
        "my_buildings": _types(obs.get("buildings")),
        "production": [
            {k: p.get(k) for k in ("queue", "item", "progress")} for p in (obs.get("production") or [])[:10] if isinstance(p, dict)
        ],
        "visible_enemy_units": _types(obs.get("visible_enemies")),
        "visible_enemy_buildings": _types(obs.get("visible_enemy_buildings")),
        "visible_enemy_buildings_cells": [
            {"type": b.get("type"), "x": b.get("x"), "y": b.get("y"), "hp": b.get("hp")}
            for b in _alive(obs.get("visible_enemy_buildings"))[:20]
        ],
        "explored_percent": obs.get("explored_percent"),
        "recent_events": events,
    }


def army_view(raw: dict) -> dict:
    obs = raw.get("observation") or {}
    units = [u for u in _alive(obs.get("units")) if str(u.get("type")) not in ("harv", "mcv")]
    if not units:
        return {"count": 0}
    xs = [u.get("x", 0) for u in units]
    ys = [u.get("y", 0) for u in units]
    return {
        "count": len(units),
        "centre": [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))],
        "idle": sum(1 for u in units if u.get("idle")),
        "avg_hp": int(sum(u.get("hp", 100) for u in units) / len(units)),
    }


def fight_view(raw: dict) -> dict | None:
    """Our units and the enemies near them, or None when nothing is fighting."""
    obs = raw.get("observation") or {}
    enemies = _alive(obs.get("visible_enemies"))
    if not enemies:
        return None
    ours = [u for u in _alive(obs.get("units")) if str(u.get("type")) not in ("harv", "mcv")]

    def near(a: dict, others: list[dict]) -> bool:
        return any(abs(a.get("x", 0) - o.get("x", 0)) + abs(a.get("y", 0) - o.get("y", 0)) <= FIGHT_RADIUS for o in others)

    ours_near = [u for u in ours if near(u, enemies)]
    if not ours_near:
        return None
    enemies_near = [e for e in enemies if near(e, ours_near)]
    pick = lambda a: {k: a.get(k) for k in ("id", "type", "x", "y", "hp")}  # noqa: E731
    return {
        "our_units": [pick(u) for u in ours_near[:40]],
        "enemies": [pick(e) for e in enemies_near[:30]],
        "enemy_buildings_near": [pick(b) for b in _alive(obs.get("visible_enemy_buildings")) if near(b, ours_near)][:10],
    }


# --- construction -------------------------------------------------------------------------

def commanders_from_env(log: Callable[[str], None] = print) -> Commanders | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        log("[commanders] the anthropic package is not installed; playing on rules only")
        return None
    client = anthropic.Anthropic(api_key=key, max_retries=1)
    prompts = {
        "strategy": (STRATEGY_PROMPT, STRATEGY_SCHEMA),
        "tactical": (TACTICAL_PROMPT, TACTICAL_SCHEMA),
        "fast": (FAST_PROMPT, FAST_SCHEMA),
    }
    layers: dict[str, Layer] = {}
    for name, (model_env, model, effort_env, effort, every, timeout, max_calls) in LAYERS.items():
        model = os.environ.get(model_env) or model
        if not model or model.lower() == "off":
            continue
        effort = os.environ.get(effort_env) or effort
        if effort not in EFFORTS:
            effort = LAYERS[name][3]
        system, schema = prompts[name]
        layers[name] = Layer(name, client, model, effort, every, timeout, max_calls, system, schema, log)
    if not layers:
        return None
    log("[commanders] " + ", ".join(f"{n}={l.model}/{l.effort}" for n, l in layers.items()))
    return Commanders(layers, log)
