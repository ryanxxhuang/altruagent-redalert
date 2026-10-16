"""Three Claude layers that command the Red Alert bot from background threads.

The scripted bot (``agent/redalert_bot.py``) plays every tick by itself and
never waits for a model. Above it, three layers run at different speeds:

| Layer      | Default model        | Effort            | Every                     | Decides                                   |
|------------|----------------------|-------------------|---------------------------|-------------------------------------------|
| strategy   | ``claude-fable-5-1``  | high              | 20 s                      | the game plan, build targets, what to avoid |
| tactical   | ``claude-fable-5-1``  | low               | 6 s                       | production and every army move (field commander) |
| fast       | off (``FAST_MODEL``)  | low, thinking off | 3 s, only while fighting  | focus-fire target, pull back damaged units |

Measured on the commander's real prompt (Oct 15): Fable low 2.7 s, Opus low
2.3 s, Opus medium 4.9 s, Fable medium 11.6 s. Fable at low effort keeps the
6 s cadence; medium would not.

The bot keeps sub-second reflexes in code (kiting, focus fire, placing
buildings, harvesting, home defense, emergency retreat) and falls back to its
own rules when the field commander has been silent for 20 s.

Each layer sees the layer above's latest instructions, so the plan flows
down: the general's plan → the commander's tactics → the bot's reflexes. A slow or failing
request only means that layer's instructions stay as they were.

Configuration (``.env``): ``ANTHROPIC_API_KEY`` turns the layers on. Per layer
``STRATEGY_MODEL`` / ``TACTICAL_MODEL`` / ``FAST_MODEL`` (``off`` disables
that layer) and ``STRATEGY_EFFORT`` / ``TACTICAL_EFFORT`` / ``FAST_EFFORT``.
The key is never printed or logged.
"""

from __future__ import annotations

import json
from collections import deque
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

# --- schemas -----------------------------------------------------------------------

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "posture": {"type": "string", "enum": ["economy", "balanced", "aggressive", "defensive"]},
        "unit_focus": {"type": "string", "enum": ["tanks", "infantry", "mixed", "anti_armor"]},
        "defenses": {"type": "integer"},
        "max_refineries": {"type": "integer"},
        "build_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["item", "count"],
                "additionalProperties": False,
            },
        },
        "avoid": {"type": "array", "items": {"type": "string"}},
        "standing_raid": {"type": "integer"},
        "plan": {"type": "string"},
    },
    "required": ["posture", "unit_focus", "defenses", "max_refineries", "build_targets", "avoid", "standing_raid", "plan"],
    "additionalProperties": False,
}

TACTICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "build": {"type": "string"},
        "defense": {"type": "string"},
        "infantry": {"type": "string"},
        "infantry_count": {"type": "integer"},
        "vehicle": {"type": "string"},
        "vehicle_count": {"type": "integer"},
        "main_action": {"type": "string", "enum": ["attack", "pincer", "ambush", "hold", "retreat", "defend"]},
        "main_x": {"type": "integer"},
        "main_y": {"type": "integer"},
        "raid_size": {"type": "integer"},
        "raid_unit": {"type": "string"},
        "raid_x": {"type": "integer"},
        "raid_y": {"type": "integer"},
        "note": {"type": "string"},
    },
    "required": ["build", "defense", "infantry", "infantry_count", "vehicle", "vehicle_count", "main_action",
                 "main_x", "main_y", "raid_size", "raid_unit", "raid_x", "raid_y", "note"],
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

# The playbook is a plain Markdown file so it can be edited without touching code.
PLAYBOOK = (Path(__file__).with_name("playbook.md")).read_text(encoding="utf-8")

STRATEGY_PROMPT = PLAYBOOK + """

## Your role: the GENERAL
You are called every ~15-30 s. A field commander (another model, asked every few seconds) runs production and the \
army following your plan; the bot uses your knobs only if the commander goes silent. Decide:
- posture: economy (more refineries, bigger later waves), balanced, aggressive (early pressure, no retreats), \
defensive (more static defenses, bigger waves).
- unit_focus: tanks, infantry, mixed, anti_armor (more rocket infantry).
- defenses: static defenses to keep (0-6). max_refineries: 1-5.
- build_targets: up to 4 {item, count} the bot reaches FIRST, before anything else, e.g. [{"item": "proc", \
"count": 2}, {"item": "harv", "count": 4}]. Walk the tech ladder as soon as 4+ harvesters are mining (the OWNER'S STANDING ORDER: use as much high tech as the \
economy allows; never put dome/stek/atek on avoid while 4+ harvesters mine): dome, then stek/atek with an apwr, then 4tnk/v2rl/ttnk/shok/dtrk/dogs (Soviets) or \
arty/ctnk/stnk/e7 (Allies), plus a 3rd proc; never aircraft, ships or superweapons (the bot cannot use them). After 10:00 with the tech up, \
build_targets lead with the top units (4tnk/ttnk/v2rl/dtrk or arty/ctnk/stnk/e7), not more basic tanks. The \
situation line says "tech: ... next: ...". Counts are totals you want to own (buildings or units); names from \
available_production / my_buildings / my_units. [] to leave it alone. The bot keeps power positive by itself.
- avoid: items the bot must not build or train until your next call, e.g. ["dome"] to save money. [] for none. \
Never avoid a prerequisite of something you want (the bot ignores such an avoid anyway).
- standing_raid: the raid squad the commander must keep out at all times (0-4 units; the bot tops the commander's \
raid_size up to this unless our base is under attack). Raids on their harvesters run ALL GAME: set 3 as soon \
as we have 8+ combat units or any fast vehicles, 0 only while every unit is needed to survive.
- plan: 2-4 sentences for your commander: the opening (rush or macro and why, from the enemy faction and what the \
scout saw), when the first attack and the raids go out, what to target, how to play the clock. Plans must USE the \
army: the first attack or raid goes out by minute 2-3 and the army keeps fighting or raiding from then on.
Reply with the JSON only."""

TACTICAL_PROMPT = PLAYBOOK + """

## Your role: the FIELD COMMANDER
You are asked again the moment you answer (every few seconds) with the newest view, and you decide everything the \
base and army do next, following the general's plan. Your previous answer stays in force until the next one: the \
bot starts your build/defense choice once per answer (so repeating "proc" every answer queues ONE refinery at a \
time, not several) and keeps the infantry/vehicle queues producing what you last said.

Production (four queues run in parallel; the bot starts your choice the moment that queue is free):
- build: the next building ("proc", "weap", "powr", ...), "" for nothing, "auto" to let the bot choose.
- defense: the next static defense ("ftur", "tsla", "pbox", "gun", ...), "" or "auto".
- infantry / vehicle: the unit that queue keeps producing (e.g. "e1", "e3", "3tnk", "harv", "ftrk"), with how many \
per batch (1-5); "" to stop that queue, "auto" for the bot's choice.
Use only names in bot.catalog (what we can build right now, with [cost, queue]). bot.problems lists choices the \
bot could not carry out and why.

Army:
- main_action: attack (the main army advances on main_x, main_y as one group, fighting what it meets; the bot does \
NOT stop for what is ahead, you decide that from bot.strength_ahead), pincer (FRONT AND BACK: the bot splits the \
army, the fastest ~40% circle round to the far side of main_x/main_y while the rest close to ~12 cells in front and \
wait; when the flank is in place both hit at once; use it on their army in the field or on a base defended from \
one side), ambush (DECOY AND POCKET: the bot splits the army into a heavy jaw and a fast jaw either side of a pocket \
between us and their army, sends the raid squad to show itself in front of their army and run back into the \
pocket, and when chasers enter it both jaws close; needs a raid squad of 2+ and their army in the field; the \
situation says "AMBUSH POSSIBLE" when the pieces are there; main_x/main_y = the pocket, -1 for automatic), \
hold (gather at main_x, main_y), retreat (go home), defend (fight the enemies near our base). \
main_x/main_y: -1 = default (their largest known army group for pincer, else the enemy base for attack, our \
staging point for hold).
- raid_size: how many units form the raid squad (0 recalls it; the general's standing_raid is the minimum the \
bot keeps out, general_standing_raid in your input); raid_unit: their type ("" = the fastest \
available; raids need FAST units, jeep/1tnk/ftrk; "e1" only for a 1-unit scout; "apc" (Soviets) = an \
APC RAID: one APC carries up to 5 of our infantry, e3 best, to raid_x/raid_y, drops them on the harvesters, \
picks them up and leaves; the bot builds the APC if needed, you keep e3 coming). A squad of 1 is a scout: use "e1", never a tank (a lone \
700-credit tank chased by infantry is a gift to the enemy). raid_x/raid_y: the target (-1,-1 = the enemy ore \
field, where their harvesters are; bot.enemy_ore_field). A squad that runs home outgunned is not re-formed for \
20 s (bot.problems says so).
- note: one short sentence for spectators.

bot.main_state says what the main army is doing, bot.our_army_strength / enemy_army_known_strength compare \
armies (credits of units, hp-weighted), bot.strength_ahead is the enemy force within reach of the main army's \
next step, bot.main_idle_seconds how long the army has been held, bot.threat_near_base_strength what is at our \
base. Visible enemies come with cells; known_enemy_buildings with cells and types.
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
    "strategy": ("STRATEGY_MODEL", "claude-fable-5-1", "STRATEGY_EFFORT", "high", 20.0, 120.0, 120),
    "tactical": ("TACTICAL_MODEL", "claude-fable-5-1", "TACTICAL_EFFORT", "low", 6.0, 40.0, 500),
    "fast": ("FAST_MODEL", "off", "FAST_EFFORT", "low", 4.0, 15.0, 400),
}
FIGHT_RADIUS = 14  # the fast layer only runs while enemies are this close to our army
# USD per million tokens (input, output), for the live cost report.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
}
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
        self.answers = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0

    def due(self) -> bool:
        return not self._busy and time.monotonic() - self._last >= self.every and self.calls < self.max_calls

    def run(self, payload: dict, handle: Callable[[dict], None]) -> None:
        self._busy = True
        self._last = time.monotonic()
        self.calls += 1
        threading.Thread(target=self._ask, args=(payload, handle), daemon=True).start()

    def _dump(self, payload: dict, answer: str) -> None:
        """Keep the latest prompt and answer of this layer in logs/ so a human
        can read exactly what the model saw and decided."""
        try:
            folder = Path(os.environ.get("REDALERT_DUMP_DIR") or "logs")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{self.name}-last.json").write_text(json.dumps(
                {"model": self.model, "effort": self.effort, "system_prompt": self.system,
                 "input": payload, "answer": json.loads(answer) if answer.startswith("{") else answer},
                indent=1, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError):
            pass

    def _count(self, response: Any, seconds: float) -> None:
        usage = getattr(response, "usage", None)
        self.answers += 1
        self.seconds += seconds
        if usage is not None:
            self.tokens_in += int(getattr(usage, "input_tokens", 0) or 0) + int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0)
            # Cache reads cost about a tenth of fresh input tokens.
            self.tokens_in += int(getattr(usage, "cache_read_input_tokens", 0) or 0) // 10
            self.tokens_out += int(getattr(usage, "output_tokens", 0) or 0)

    def cost(self) -> float | None:
        price = next((p for m, p in PRICES.items() if self.model.startswith(m)), None)
        if price is None:
            return None
        return (self.tokens_in * price[0] + self.tokens_out * price[1]) / 1e6

    def report(self) -> str:
        cost = self.cost()
        avg = self.seconds / self.answers if self.answers else 0.0
        money = f"${cost:.2f}" if cost is not None else "$?"
        return f"{self.name}={self.model.removeprefix('claude-')} {money} ({self.answers} answers, avg {avg:.1f}s)"

    def _request_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000 if self.name == "strategy" else 4000,
            # The playbook is long and identical every call: cache it.
            "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}},
        }
        # Server-side refusal fallback: Opus/Sonnet/Fable only (Haiku has none).
        if not self.model.startswith("claude-haiku"):
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        # The micro layer must answer fast: thinking off (both accepted at effort high or below).
        if self.name == "fast" and self.effort in ("low", "medium", "high"):
            if self.model.startswith("claude-haiku-5-5"):
                kwargs["thinking"] = {"type": "disabled"}
            elif self.model.startswith("claude-sonnet-5-5"):
                kwargs["thinking"] = {"type": "between_tools"}
        return kwargs

    def _ask(self, payload: dict, handle: Callable[[dict], None]) -> None:
        import anthropic

        started = time.monotonic()
        kwargs = self._request_kwargs()
        try:
            api = self._client.with_options(timeout=self.timeout)
            create = api.beta.messages.create if "betas" in kwargs else api.messages.create
            response = create(
                messages=[{"role": "user", "content": json.dumps(payload, separators=(",", ":"))}],
                **kwargs,
            )
            if response.stop_reason == "refusal":
                self._log(f"[{self.name}] request declined; keeping current instructions")
                return
            text = next((b.text for b in response.content if b.type == "text"), "")
            self._count(response, time.monotonic() - started)
            self._dump(payload, text)
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
        self._decision: dict | None = None
        self._seq = 0
        self._recent_decisions: deque = deque(maxlen=5)

    # -- read by the bot -------------------------------------------------------------

    def knobs(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._knobs)

    def cost_report(self) -> str:
        total = sum(l.cost() or 0.0 for l in self._layers.values())
        return f"${total:.2f} so far: " + ", ".join(l.report() for l in self._layers.values())

    def pop_commands(self) -> list[dict]:
        with self._lock:
            out, self._commands = self._commands, []
        return out

    # -- called on every decision ------------------------------------------------------

    def observe(self, raw: dict, bot_status: dict | None = None) -> None:
        summary = summarize(raw)
        if summary is None:
            return
        summary["bot_status"] = bot_status or {}
        strategy = self._layers.get("strategy")
        if strategy and strategy.due():
            strategy.run({"previous_plans": self._plans[-3:], "situation": (bot_status or {}).get("situation"),
                          "opponent": (bot_status or {}).get("opponent"), "ledger": (bot_status or {}).get("ledger"),
                          "match": {k: v for k, v in summary.items() if k != "bot_status"}}, self._on_strategy)
        tactical = self._layers.get("tactical")
        if tactical and tactical.due():
            obs = raw.get("observation") or {}
            asked = time.monotonic()
            tactical.run(
                {"situation": (bot_status or {}).get("situation"),
                 "opponent": (bot_status or {}).get("opponent"),
                 "ledger": (bot_status or {}).get("ledger"),
                 "raid_candidates": (bot_status or {}).get("raid_candidates"),
                 "your_recent_decisions": [
                     {k: v for k, v in dd.items() if k in ("main_action", "main_x", "main_y", "raid_size", "build", "vehicle")}
                     for dd in list(self._recent_decisions)[-3:]],
                 "general_standing_raid": self._knobs.get("standing_raid", 0),
                 "strategic_plan": self._plan, "general_build_targets": self._knobs.get("build_targets"),
                 "general_avoid": self._knobs.get("avoid"), "your_last_note": self._tactical_note,
                 "match": {k: v for k, v in summary.items() if k not in ("bot_status", "visible_enemy_buildings_cells")},
                 "known_enemy_buildings": (bot_status or {}).get("known_enemy_buildings"),
                 "bot": {k: v for k, v in (bot_status or {}).items() if k not in ("known_enemy_buildings", "situation")},
                 "army": army_view(raw),
                 "visible_enemies": [
                     {k: e.get(k) for k in ("type", "x", "y", "hp")} for e in _alive(obs.get("visible_enemies"))[:30]
                 ]},
                lambda answer, asked=asked: self._on_tactical(answer, asked),
            )
        fast = self._layers.get("fast")
        if fast and fast.due():
            fight = fight_view(raw)
            if fight is not None:
                fight["tactical_note"] = self._tactical_note
                fast.run(fight, self._on_fast)

    # -- answers -------------------------------------------------------------------------

    def _on_strategy(self, a: dict) -> None:
        targets = []
        for t in a.get("build_targets") or []:
            if isinstance(t, dict) and isinstance(t.get("item"), str):
                targets.append((t["item"].strip().lower()[:12], max(0, min(12, int(t.get("count", 1))))))
        avoid = [str(x).strip().lower()[:12] for x in a.get("avoid") or [] if isinstance(x, str)][:6]
        try:
            standing = max(0, min(4, int(a.get("standing_raid", 0))))
        except (TypeError, ValueError):
            standing = 0
        knobs = {
            "build_targets": targets[:4],
            "avoid": avoid,
            "standing_raid": standing,
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
                  f"{knobs['max_refineries']} refineries, standing raid {standing}, build first {targets[:4]}, "
                  f"avoid {avoid}: {plan[:200]}")

    def _on_tactical(self, a: dict, asked_at: float | None = None) -> None:
        def name(key: str) -> str:
            return str(a.get(key, "auto")).strip().lower()[:12]

        def num(key: str, lo: int, hi: int, default: int) -> int:
            try:
                return max(lo, min(hi, int(a.get(key, default))))
            except (TypeError, ValueError):
                return default

        action = a.get("main_action") if a.get("main_action") in ("attack", "pincer", "ambush", "hold", "retreat", "defend") else "hold"
        decision = {
            "build": name("build"), "defense": name("defense"),
            "infantry": name("infantry"), "infantry_count": num("infantry_count", 1, 5, 2),
            "vehicle": name("vehicle"), "vehicle_count": num("vehicle_count", 1, 5, 1),
            "main_action": action, "main_x": num("main_x", -1, 500, -1), "main_y": num("main_y", -1, 500, -1),
            "raid_size": num("raid_size", 0, 8, 0), "raid_unit": "" if name("raid_unit") in ("auto", "") else name("raid_unit"),
            "raid_x": num("raid_x", -1, 500, -1),
            "raid_y": num("raid_y", -1, 500, -1),
        }
        self._tactical_note = str(a.get("note", ""))[:200]
        self._recent_decisions.append(decision)
        with self._lock:
            self._seq += 1
            self._decision = {**decision, "seq": self._seq, "t": time.monotonic(),
                              "asked_at": asked_at if asked_at is not None else time.monotonic()}
        where = f"({decision['main_x']},{decision['main_y']})" if decision["main_x"] >= 0 else "default"
        self._log(
            f"[commander] build={decision['build'] or '-'} def={decision['defense'] or '-'} "
            f"inf={decision['infantry'] or '-'}x{decision['infantry_count']} "
            f"veh={decision['vehicle'] or '-'}x{decision['vehicle_count']} main={action}@{where} "
            f"raid={decision['raid_size']}{(' ' + decision['raid_unit']) if decision['raid_unit'] else ''}: {self._tactical_note}"
        )

    def decision(self) -> dict | None:
        """The field commander's latest decision (with ``seq`` and ``t``), if any."""
        with self._lock:
            return dict(self._decision) if self._decision else None

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
        "available_production": sorted(str(a) for a in obs.get("available_production") or [])[:40],
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

def seat_env(name: str, seat: int | None) -> str | None:
    """An env value, or this seat's part of ``seat0:a,seat1:b`` (for A/B tests)."""
    value = os.environ.get(name)
    if not value or "seat" not in value or ":" not in value:
        return value
    for part in value.split(","):
        key, _, v = part.partition(":")
        if key.strip().lower() == f"seat{seat}":
            return v.strip()
    return None


def commanders_from_env(log: Callable[[str], None] = print, seat: int | None = None) -> Commanders | None:
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
        model = seat_env(model_env, seat) or model
        if not model or model.lower() == "off":
            continue
        effort = seat_env(effort_env, seat) or effort
        if effort not in EFFORTS:
            effort = LAYERS[name][3]
        try:
            every = float(seat_env(f"{name.upper()}_EVERY", seat) or every)
        except ValueError:
            pass
        system, schema = prompts[name]
        # An operator's instruction for this match (A/B tests: STRATEGY_HINT=seat0:...,seat1:...).
        hint = seat_env("STRATEGY_HINT", seat)
        if hint and name in ("strategy", "tactical"):
            system += f"\n\n## Operator instruction for this match (follow it)\n{hint.strip()}\n"
        layers[name] = Layer(name, client, model, effort, every, timeout, max_calls, system, schema, log)
    if not layers:
        return None
    log("[commanders] " + ", ".join(f"{n}={l.model}/{l.effort}/every {l.every:g}s" for n, l in layers.items()))
    return Commanders(layers, log)
