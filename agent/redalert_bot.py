"""A fast, rule-based Red Alert player.

Red Alert is real time: every second spent thinking is 25 ticks the enemy
plays without you. So every decision here is plain Python that runs in
microseconds, called about 5 times a second with the newest view:

- economy: deploy the MCV, follow a build order, keep power positive, place
  finished buildings at once, keep harvesters working, repair the base;
- production: keep the infantry and vehicle queues busy with the best units
  available (tanks first, anti-armour infantry alongside);
- army: gather at a staging point in front of the base, defend anything that
  comes near the base, attack in waves when the army is big enough, and pull
  back when a wave is losing (the time-limit tiebreak counts trades).

Optional ``advisor`` layers (``agent/advisor.py``: Claude strategy, tactics
and micro) adjust knobs and send one-shot commands (attack now, retreat,
focus fire, pull back) from background threads; the bot never waits for them.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from altruagent import WAIT, DecisionContext, GameState, WithReasoning

# --- unit and building knowledge (OpenRA Red Alert names) -----------------------------

NON_COMBAT = frozenset(
    {"mcv", "harv", "e6", "medi", "mech", "spy", "thf", "truk", "mnly", "lst", "tran", "badr", "u2", "hijacker"}
)
SHIPS = frozenset({"ss", "msub", "dd", "ca", "pt", "lst"})
AIRCRAFT = frozenset({"heli", "hind", "mh60", "yak", "mig", "tran", "badr", "u2"})
BARRACKS = ("tent", "barr")
POWER = ("apwr", "powr")
DEFENSES_ALLIES = ("gun", "pbox", "hbox")
DEFENSES_SOVIET = ("tsla", "ftur")
DEFENSE_TYPES = frozenset({"gun", "pbox", "hbox", "tsla", "ftur", "agun", "sam", "gap"})
# Best first. Mammoth tanks are handled separately (expensive and slow).
TANKS = ("3tnk", "2tnk", "ttnk", "1tnk", "ftrk", "apc", "jeep")
ARTILLERY = ("v2rl", "arty")
ANTI_ARMOUR_INFANTRY = "e3"
RIFLE_INFANTRY = "e1"

# Fallbacks when the server's rules are missing (credits, power).
DEFAULT_COST = {
    "powr": 300, "apwr": 500, "tent": 500, "barr": 500, "proc": 1400, "weap": 2000, "fix": 1200,
    "dome": 1000, "gun": 600, "pbox": 400, "hbox": 600, "ftur": 600, "tsla": 1500, "silo": 150,
    "e1": 100, "e3": 300, "harv": 1100, "1tnk": 700, "2tnk": 850, "3tnk": 1150, "4tnk": 1700,
    "jeep": 500, "apc": 850, "ftrk": 600, "v2rl": 700, "arty": 600, "ttnk": 1350,
}
DEFAULT_POWER = {
    "powr": 100, "apwr": 200, "tent": -20, "barr": -20, "proc": -30, "weap": -30, "fix": -30,
    "dome": -40, "gun": -40, "pbox": -15, "hbox": -15, "ftur": -20, "tsla": -100, "silo": -10,
    "atek": -200, "stek": -100, "hpad": -10, "afld": -30, "kenn": -10, "spen": -30, "syrd": -30,
}

# --- tuning -----------------------------------------------------------------------

BASE_RADIUS = 16            # enemies this close to the base centre are a threat
BUILDING_THREAT_RADIUS = 9  # ... or this close to any of our buildings
STAGING_DISTANCE = 7        # staging point, cells from the base towards the enemy
DEFAULT_KNOBS = {
    "posture": "balanced",   # economy | balanced | aggressive | defensive
    "attack_size": 8,        # combat units needed for the first wave
    "unit_focus": "mixed",   # tanks | infantry | mixed | anti_armor
    "defenses": 2,           # static defenses to keep
    "max_refineries": 3,
}
STATUS_EVERY_S = 30.0
# Micro (every decision, ~5 a second).
FOCUS_ENGAGE_RANGE = 8      # enemies this close to one of our units are in the fight
FOCUS_SHOOTER_RANGE = 10    # our units this close to the focus target join the focus fire
FOCUS_EVERY_S = 1.5
PULL_BACK_HP = 30           # valuable units under this hp % retreat home
PULL_BACK_TYPES = frozenset({"1tnk", "2tnk", "3tnk", "4tnk", "ttnk", "v2rl", "arty", "ftrk", "apc", "ctnk", "stnk"})
# How much killing an enemy unit is worth (scaled up as its hp drops).
TARGET_VALUE = {
    "mcv": 6, "v2rl": 5, "arty": 5, "4tnk": 4, "3tnk": 3.5, "ttnk": 3.5, "2tnk": 3, "1tnk": 2.5,
    "harv": 3, "ftrk": 2.5, "apc": 2, "jeep": 1.5, "e3": 1.5, "e4": 1.5, "shok": 2, "e7": 3,
    "e1": 1, "e2": 1, "dog": 0.8, "medi": 1.2, "e6": 1.5, "spy": 1, "thf": 1,
}


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _xy(actor: dict) -> tuple[int, int] | None:
    x, y = actor.get("x"), actor.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return int(x), int(y)
    return None


def _is_wreck(actor: dict) -> bool:
    return bool(actor.get("wreck")) or str(actor.get("type", "")).endswith("husk")


def _chunks(ids: list[int], size: int) -> list[list[int]]:
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def _item_rules(config: dict | None) -> dict[str, dict]:
    """``{item: {cost, power, ...}}`` from ``get_game_config``'s rules, however
    they are nested; empty when the config is missing."""
    found: dict[str, dict] = {}

    def walk(node: Any, depth: int) -> None:
        if depth > 4 or not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict) and ("cost" in value or "power" in value) and isinstance(key, str):
                found.setdefault(key.lower(), value)
            elif isinstance(value, dict):
                walk(value, depth + 1)

    if isinstance(config, dict):
        walk(config.get("rules"), 0)
    return found


class RedAlertBot:
    def __init__(self, *, advisor: Any = None, log: Callable[[str], None] = print, dump_dir: str | None = None):
        self.advisor = advisor
        self.log = log
        self.knobs = dict(DEFAULT_KNOBS)
        self.rules: dict[str, dict] = {}
        self._config_loaded = False
        self._dump_dir = Path(dump_dir) if dump_dir else None
        self._dumped_state = False
        # When we last sent something, by key (item name, order kind...).
        self._sent: dict[str, float] = {}
        # Items the server refused, by name: don't retry until this time.
        self._blocked: dict[str, float] = {}
        self._rally_set: dict[int, tuple[int, int]] = {}
        self._enemy_buildings: dict[int, dict] = {}  # last known enemy buildings
        self._attacking = False
        self._wave = 0
        self._wave_peak = 0
        self._last_status = 0.0
        self._last_reason = ""
        self._last_orders: list[dict] = []
        self.batches = 0
        self._commands: list[dict] = []   # one-shot commands from the advisor layers
        self._focus: dict | None = None   # focus-fire target from the fast layer

    # -- runtime hooks -------------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        raw = state.raw
        obs = raw.get("observation")
        if not isinstance(obs, dict):
            return WAIT
        if not self._config_loaded:
            self._load_config(context.game_config, raw)
        if self.advisor is not None:
            self.advisor.observe(raw)
            self.knobs.update(self.advisor.knobs())
            for cmd in self.advisor.pop_commands():
                if cmd.get("type") == "focus":
                    self._focus = cmd
                else:
                    self._commands.append(cmd)
        try:
            orders, reason = self._decide(raw, obs)
        except Exception as exc:  # noqa: BLE001 - one bad view must never stop the match
            self.log(f"[bot] decision error: {exc!r}")
            return WAIT
        self._status(raw, obs)
        if not orders:
            return WAIT
        orders = orders[: int((raw.get("limits") or {}).get("max_orders_per_batch") or 20)]
        orders = self._cap_commands(orders)
        self._last_orders = orders
        self.batches += 1
        self._last_reason = reason
        return WithReasoning({"type": "orders", "orders": orders}, reason[:480])

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        if not isinstance(result, dict):
            return
        rows = result.get("results") if isinstance(result.get("results"), list) else []
        if result.get("error") == "INVALID_ACTION":
            self.log(f"[bot] batch refused: {str(result.get('detail'))[:200]}")
            # Block every item order of that batch for a moment.
            for order in self._last_orders:
                if order.get("item"):
                    self._block(str(order["item"]), 4.0)
            return
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "rejected":
                continue
            idx = row.get("index")
            order = self._last_orders[idx] if isinstance(idx, int) and idx < len(self._last_orders) else {}
            item = order.get("item")
            if item:
                self._block(str(item), 8.0 if row.get("reason") == "not_available" else 3.0)

    # -- setup -----------------------------------------------------------------------

    def _load_config(self, config: dict | None, raw: dict) -> None:
        self._config_loaded = True
        self.rules = _item_rules(config)
        if self._dump_dir is not None:
            try:
                self._dump_dir.mkdir(parents=True, exist_ok=True)
                if config:
                    (self._dump_dir / "red_alert_config.json").write_text(json.dumps(config, indent=1))
                (self._dump_dir / "red_alert_first_state.json").write_text(json.dumps(raw, indent=1))
            except OSError:
                pass
        obs = raw.get("observation") or {}
        self.log(
            f"[bot] match start: faction={obs.get('faction')} vs {obs.get('enemy_faction')}, "
            f"map={(raw.get('map') or {}).get('id')}, rules for {len(self.rules)} items"
        )

    def _cost(self, item: str) -> int:
        value = (self.rules.get(item) or {}).get("cost")
        return int(value) if isinstance(value, (int, float)) else DEFAULT_COST.get(item, 800)

    def _power(self, item: str) -> int:
        value = (self.rules.get(item) or {}).get("power")
        return int(value) if isinstance(value, (int, float)) else DEFAULT_POWER.get(item, 0)

    # -- bookkeeping -------------------------------------------------------------------

    def _recent(self, key: str, seconds: float) -> bool:
        return time.monotonic() - self._sent.get(key, -1e9) < seconds

    def _mark(self, key: str) -> None:
        self._sent[key] = time.monotonic()

    def _block(self, item: str, seconds: float) -> None:
        self._blocked[item.lower()] = time.monotonic() + seconds

    def _blocked_now(self, item: str) -> bool:
        return time.monotonic() < self._blocked.get(item.lower(), 0.0)

    def _cap_commands(self, orders: list[dict]) -> list[dict]:
        """Keep the batch under the server's 200 engine-command cap."""
        kept, total = [], 0
        for order in orders:
            n = len(order.get("units") or []) or int(order.get("count") or 1)
            if total + n > 190:
                continue
            kept.append(order)
            total += n
        return kept

    # -- the decision ------------------------------------------------------------------

    def _decide(self, raw: dict, obs: dict) -> tuple[list[dict], str]:
        la = raw.get("legal_actions") if isinstance(raw.get("legal_actions"), dict) else {}
        units = [u for u in obs.get("units") or [] if isinstance(u, dict) and not _is_wreck(u)]
        buildings = [b for b in obs.get("buildings") or [] if isinstance(b, dict)]
        enemies = [e for e in obs.get("visible_enemies") or [] if isinstance(e, dict) and not _is_wreck(e)]
        enemy_buildings = [e for e in obs.get("visible_enemy_buildings") or [] if isinstance(e, dict) and not _is_wreck(e)]
        production = [p for p in obs.get("production") or [] if isinstance(p, dict)]
        available = {str(a).lower() for a in obs.get("available_production") or []}
        econ = obs.get("economy") or {}
        money = int(econ.get("cash") or 0) + int(econ.get("ore") or 0)
        power_margin = int(econ.get("power_provided") or 0) - int(econ.get("power_drained") or 0)

        base = self._base(obs, buildings, units)
        enemy_base = self._enemy_base(la, base)
        staging = self._staging(base, enemy_base, raw)
        self._remember_enemy_buildings(enemy_buildings, units, buildings)

        orders: list[dict] = []
        notes: list[str] = []
        btypes = Counter(str(b.get("type", "")).lower() for b in buildings)
        has_yard = btypes["fact"] > 0

        # 1. Deploy the MCV (also a rebuilt or second one, if we have no yard).
        mcvs = [u["id"] for u in units if str(u.get("type")).lower() == "mcv"]
        if mcvs and (not has_yard) and not self._recent("deploy", 3.0):
            orders.append({"cmd": "deploy", "units": mcvs[:1]})
            self._mark("deploy")
            notes.append("deploying MCV")

        # 2. Place finished buildings and defenses right away.
        for p in production:
            item = str(p.get("item", "")).lower()
            if p.get("ready") and p.get("queue") in ("Building", "Defense") and not self._recent(f"place:{item}", 1.5):
                order: dict[str, Any] = {"cmd": "place", "item": item}
                if item in DEFENSE_TYPES and base and enemy_base:
                    order["at"] = list(self._toward(base, enemy_base, 5 + 2 * (btypes[item] % 3), raw, jitter=btypes[item]))
                orders.append(order)
                self._mark(f"place:{item}")
                notes.append(f"placing {item}")

        queued = Counter(str(p.get("item", "")).lower() for p in production)
        busy_queues = {p.get("queue") for p in production}

        # 3. Build order (Building queue: one at a time).
        if has_yard and "Building" not in busy_queues and not self._recent("build", 2.5):
            item = self._next_building(btypes, queued, available, money, power_margin, econ, raw)
            if item:
                orders.append({"cmd": "build", "item": item})
                self._mark("build")
                notes.append(f"building {item}")

        # 4. Static defenses (Defense queue).
        want_def = int(self.knobs.get("defenses", 2))
        if self.knobs.get("posture") == "defensive":
            want_def += 2
        n_def = sum(btypes[t] for t in DEFENSE_TYPES) + sum(queued[t] for t in DEFENSE_TYPES)
        if (
            has_yard
            and "Defense" not in busy_queues
            and n_def < want_def
            and btypes["proc"] >= 1
            and money > 900
            and not self._recent("defense", 4.0)
        ):
            item = self._pick_defense(available, power_margin)
            if item:
                orders.append({"cmd": "build", "item": item})
                self._mark("defense")
                notes.append(f"building defense {item}")

        # 5. Units.
        harvesters = [u for u in units if str(u.get("type")).lower() == "harv"]
        army = [u for u in units if self._is_combat(u)]
        orders += self._train(available, busy_queues, queued, btypes, harvesters, army, money, notes)

        # 6. Economy upkeep: idle harvesters back to work, repairs, rally points.
        idle_harv = [u["id"] for u in harvesters if u.get("idle")]
        if idle_harv and not self._recent("harvest", 5.0):
            orders.append({"cmd": "harvest", "units": idle_harv[:50]})
            self._mark("harvest")
        if money > 400 and not self._recent("repair", 3.0):
            for entry in (la.get("repair") or [])[:3]:
                bid = entry.get("id") if isinstance(entry, dict) else entry
                if isinstance(bid, int):
                    orders.append({"cmd": "repair", "building": bid})
            if la.get("repair"):
                self._mark("repair")
        if staging:
            for b in la.get("production_buildings") or []:
                bid = b.get("id") if isinstance(b, dict) else None
                btype = str(b.get("type", "")).lower() if isinstance(b, dict) else ""
                if isinstance(bid, int) and btype not in ("proc", "fact") and self._rally_set.get(bid) != staging:
                    orders.append({"cmd": "rally", "building": bid, "to": list(staging)})
                    self._rally_set[bid] = staging

        # 7. The army.
        orders += self._command_army(raw, obs, army, enemies, buildings, base, enemy_base, staging, notes)

        return orders, "; ".join(notes) or "macro"

    # -- economy -------------------------------------------------------------------------

    def _next_building(self, have: Counter, queued: Counter, available: set[str], money: int,
                       power_margin: int, econ: dict, raw: dict) -> str | None:
        def count(*items: str) -> int:
            return sum(have[i] + queued[i] for i in items)

        def ok(item: str) -> bool:
            return item in available and not self._blocked_now(item)

        def power_item() -> str | None:
            for p in POWER:
                if ok(p):
                    return p
            return None

        barracks = next((b for b in BARRACKS if ok(b)), None)
        steps: list[tuple[bool, str | None]] = []
        steps.append((count(*POWER) == 0, power_item()))
        steps.append((count(*BARRACKS) == 0, barracks))
        steps.append((count("proc") == 0, "proc"))
        # Keep power ahead of what the next building drains.
        steps.append((power_margin < 30 or (power_margin < 60 and count(*POWER) < 4), power_item()))
        steps.append((count("weap") == 0, "weap"))
        steps.append((count("proc") < 2, "proc"))
        steps.append((count("fix") == 0, "fix"))
        steps.append((count("dome") == 0 and money > 1500, "dome"))
        steps.append((count("weap") < 2 and money > 2500 and self.knobs.get("posture") != "economy", "weap"))
        max_proc = int(self.knobs.get("max_refineries", 3))
        if self.knobs.get("posture") == "economy":
            max_proc += 1
        steps.append((count("proc") < max_proc and money > 1800, "proc"))
        steps.append((count(*BARRACKS) < 2 and money > 3000, barracks))
        steps.append((power_margin < 80 and money > 2000, power_item()))
        # Spare money with nothing else to build: one more war factory.
        steps.append((count("weap") < 3 and money > 5000, "weap"))
        for wanted, item in steps:
            if wanted and item and ok(item):
                # Don't start a power-hungry building without the power for it.
                drain = self._power(item)
                if drain < 0 and power_margin + drain < 0 and item not in POWER:
                    p = power_item()
                    return p if p and count(*POWER) < 12 else item
                return item
        return None

    def _pick_defense(self, available: set[str], power_margin: int) -> str | None:
        for item in DEFENSES_ALLIES + DEFENSES_SOVIET:
            if item not in available or self._blocked_now(item):
                continue
            if item == "tsla" and power_margin < 120:
                continue
            return item
        return None

    def _train(self, available: set[str], busy: set, queued: Counter, have: Counter,
               harvesters: list[dict], army: list[dict], money: int, notes: list[str]) -> list[dict]:
        orders: list[dict] = []

        def ok(item: str) -> bool:
            return item in available and not self._blocked_now(item)

        focus = self.knobs.get("unit_focus", "mixed")
        n_proc = have["proc"]
        # Vehicles: harvesters first, then the best tank.
        if "Vehicle" not in busy and not self._recent("vehicle", 2.0):
            item = None
            want_harv = min(n_proc * 2, 6) if n_proc else 0
            if len(harvesters) + queued["harv"] < max(want_harv, n_proc + (1 if n_proc else 0)) and ok("harv"):
                item = "harv"
            elif money > 600:
                tanks = [t for t in TANKS if ok(t)]
                arty = [t for t in ARTILLERY if ok(t)]
                n_arty = sum(1 for u in army if str(u.get("type")).lower() in ARTILLERY)
                if ok("4tnk") and money > 3500:
                    item = "4tnk"
                elif arty and len(army) >= 6 and n_arty * 5 < len(army):
                    item = arty[0]
                elif tanks and focus != "infantry":
                    item = tanks[0]
            if item:
                orders.append({"cmd": "train", "item": item, "count": 1})
                self._mark("vehicle")
                notes.append(f"training {item}")
        # Infantry: always some, more early or when the focus says so.
        if "Infantry" not in busy and not self._recent("infantry", 2.0):
            n_inf = sum(1 for u in army if str(u.get("type")).lower() in ("e1", "e2", "e3", "e4"))
            has_factory = have["weap"] > 0
            cap = 30 if focus == "infantry" else (8 if has_factory else 14)
            min_money = 150 if not has_factory else 900
            if n_inf < cap and money > min_money:
                n_e3 = sum(1 for u in army if str(u.get("type")).lower() == "e3")
                anti = focus == "anti_armor" or n_e3 * 2 < n_inf
                item = ANTI_ARMOUR_INFANTRY if anti and ok(ANTI_ARMOUR_INFANTRY) else RIFLE_INFANTRY
                if ok(item):
                    orders.append({"cmd": "train", "item": item, "count": 2})
                    self._mark("infantry")
                    notes.append(f"training {item}")
        return orders

    # -- army ------------------------------------------------------------------------------

    def _is_combat(self, unit: dict) -> bool:
        t = str(unit.get("type", "")).lower()
        return t not in NON_COMBAT and t not in SHIPS and t not in AIRCRAFT and not _is_wreck(unit)

    def _command_army(self, raw: dict, obs: dict, army: list[dict], enemies: list[dict], buildings: list[dict],
                      base: tuple[int, int] | None, enemy_base: tuple[int, int] | None,
                      staging: tuple[int, int] | None, notes: list[str]) -> list[dict]:
        orders = self._army_core(raw, obs, army, enemies, buildings, base, enemy_base, staging, notes)
        # Micro from the fast layer goes last, so it overrides the moves above.
        return orders + self._micro(raw, army, enemies, staging, notes)

    def _micro(self, raw: dict, army: list[dict], enemies: list[dict], staging, notes: list[str]) -> list[dict]:
        """Focus fire and pulling back damaged units. Rules by default; a fast
        LLM layer's commands (``focus`` / ``pull_back``) take over when present."""
        orders: list[dict] = []
        max_units = int((raw.get("limits") or {}).get("max_units_per_order") or 50)
        ours = {u["id"]: u for u in army if isinstance(u.get("id"), int) and _xy(u)}
        if not ours:
            return orders
        cells = {i: _xy(u) for i, u in ours.items()}
        fighting = [
            e for e in enemies
            if _xy(e) and any(_dist(_xy(e), c) <= FOCUS_ENGAGE_RANGE for c in cells.values())
        ]

        # Pull back: LLM-chosen ids, else damaged valuable units in a fight.
        pull = []
        for cmd in [c for c in self._commands if c.get("type") == "pull_back"]:
            self._commands.remove(cmd)
            pull += [i for i in cmd.get("units") or [] if i in ours]
        if fighting:
            pull += [
                i for i, u in ours.items()
                if str(u.get("type", "")).lower() in PULL_BACK_TYPES
                and isinstance(u.get("hp"), (int, float)) and u["hp"] < PULL_BACK_HP
                and not self._recent(f"pull:{i}", 12.0)
            ]
        pull = list(dict.fromkeys(pull))
        if pull and staging:
            orders.append({"cmd": "move", "units": pull[:max_units], "to": list(staging)})
            for i in pull:
                self._mark(f"pull:{i}")
            notes.append(f"pulling back {len(pull)} damaged units")
        pulled = set(pull)

        # Focus fire: LLM target while it lasts, else the best target in the fight.
        if self._recent("focus", FOCUS_EVERY_S):
            return orders
        target = None
        focus = self._focus
        if focus and time.monotonic() < focus.get("until", 0):
            target = next((e for e in enemies if e.get("id") == focus.get("target")), None)
        if target is None:
            self._focus = None
            if fighting:
                target = max(fighting, key=self._target_score)
        tcell = _xy(target) if target else None
        if not tcell or not isinstance(target.get("id"), int):
            return orders
        shooters = [i for i, c in cells.items() if i not in pulled and _dist(c, tcell) <= FOCUS_SHOOTER_RANGE]
        for chunk in _chunks(shooters, max_units):
            orders.append({"cmd": "attack", "units": chunk, "target": target["id"]})
        if shooters:
            self._mark("focus")
            notes.append(f"focus fire on {target.get('type')} {target['id']}")
        return orders

    @staticmethod
    def _target_score(enemy: dict) -> float:
        value = TARGET_VALUE.get(str(enemy.get("type", "")).lower(), 1.5)
        hp = enemy.get("hp") if isinstance(enemy.get("hp"), (int, float)) else 100
        return value * (1.0 + (100 - hp) / 50.0)

    def _army_core(self, raw: dict, obs: dict, army: list[dict], enemies: list[dict], buildings: list[dict],
                   base: tuple[int, int] | None, enemy_base: tuple[int, int] | None,
                   staging: tuple[int, int] | None, notes: list[str]) -> list[dict]:
        orders: list[dict] = []
        if not army:
            self._attacking = False
            return orders
        max_units = int((raw.get("limits") or {}).get("max_units_per_order") or 50)
        building_cells = [c for c in (_xy(b) for b in buildings) if c]

        # Threats: enemies near the base or any of our buildings.
        threats = []
        for e in enemies:
            c = _xy(e)
            if not c:
                continue
            near_base = base is not None and _dist(c, base) <= BASE_RADIUS
            near_building = any(_dist(c, bc) <= BUILDING_THREAT_RADIUS for bc in building_cells)
            if near_base or near_building:
                threats.append((e, c))

        army_ids = [u["id"] for u in army if isinstance(u.get("id"), int)]
        if threats and not self._recent("defend", 2.0):
            # Nearest threat to the base; everyone not deep in the enemy base responds.
            threats.sort(key=lambda t: _dist(t[1], base) if base else 0)
            target = threats[0][1]
            defenders = [
                u["id"] for u in army
                if isinstance(u.get("id"), int) and (not self._attacking or (_xy(u) and base and _dist(_xy(u), base) < 30))
            ]
            for chunk in _chunks(defenders, max_units):
                orders.append({"cmd": "attack_move", "units": chunk, "to": list(target)})
            self._mark("defend")
            notes.append(f"defending base against {len(threats)} enemies at {target}")
            return orders
        if threats:
            return orders

        size = len(army)
        wave_size = self._wave_size(raw)
        remaining = ((raw.get("time") or {}).get("remaining_seconds"))
        mil = obs.get("military") or {}
        trade = int(mil.get("kills_cost") or 0) - int(mil.get("deaths_cost") or 0)

        # One-shot orders from the tactical layer.
        for cmd in [c for c in self._commands if c.get("type") in ("retreat", "attack_now")]:
            self._commands.remove(cmd)
            if cmd["type"] == "retreat" and self._attacking:
                self._attacking = False
                if staging:
                    for chunk in _chunks(army_ids, max_units):
                        orders.append({"cmd": "move", "units": chunk, "to": list(staging)})
                notes.append("retreat ordered by tactical layer")
                self.log("[bot] retreat ordered by tactical layer")
                return orders
            if cmd["type"] == "attack_now" and not self._attacking and size >= 4:
                wave_size = 0  # launches below

        if self._attacking:
            self._wave_peak = max(self._wave_peak, size)
            # A wave that has lost most of its units pulls back to regroup.
            if size < max(3, int(self._wave_peak * 0.35)) and self.knobs.get("posture") != "aggressive":
                self._attacking = False
                if staging:
                    for chunk in _chunks(army_ids, max_units):
                        orders.append({"cmd": "move", "units": chunk, "to": list(staging)})
                notes.append("wave broken, regrouping")
                return orders
            # Late in a close match while ahead on trades: protect the lead.
            if isinstance(remaining, (int, float)) and remaining < 60 and trade > 1500:
                self._attacking = False
                if staging:
                    for chunk in _chunks(army_ids, max_units):
                        orders.append({"cmd": "move", "units": chunk, "to": list(staging)})
                notes.append("holding the lead to the time limit")
                return orders
        elif size >= wave_size:
            self._attacking = True
            self._wave += 1
            self._wave_peak = size
            self._sent.pop("attack_all", None)
            self._sent.pop("attack", None)
            notes.append(f"wave {self._wave}: attacking with {size} units")
            self.log(f"[bot] wave {self._wave}: attacking with {size} units")

        if self._attacking:
            if self._recent("attack", 3.0):
                return orders
            target = self._attack_target(enemies, army, enemy_base)
            if target:
                # Send idle units and stragglers; units already fighting keep going.
                # (every 15 s everyone is re-sent, so stragglers catch up).
                everyone = not self._recent("attack_all", 15.0)
                if everyone:
                    self._mark("attack_all")
                movers = [u["id"] for u in army if isinstance(u.get("id"), int) and (everyone or u.get("idle"))]
                for chunk in _chunks(movers, max_units):
                    orders.append({"cmd": "attack_move", "units": chunk, "to": list(target)})
                self._mark("attack")
            return orders

        # Gathering: idle units wait at the staging point.
        if staging and not self._recent("gather", 4.0):
            idle_away = [
                u["id"] for u in army
                if isinstance(u.get("id"), int) and u.get("idle") and _xy(u) and _dist(_xy(u), staging) > 4
            ]
            for chunk in _chunks(idle_away, max_units):
                orders.append({"cmd": "attack_move", "units": chunk, "to": list(staging)})
            if idle_away:
                self._mark("gather")
        return orders

    def _wave_size(self, raw: dict) -> int:
        base = int(self.knobs.get("attack_size", 8))
        posture = self.knobs.get("posture")
        if posture == "aggressive":
            base = max(4, base - 3)
        elif posture in ("defensive", "economy"):
            base += 4
        size = base + 3 * self._wave
        remaining = (raw.get("time") or {}).get("remaining_seconds")
        if isinstance(remaining, (int, float)) and remaining < 240:
            size = max(4, size // 2)  # last minutes: use the army
        return min(size, 24)

    def _attack_target(self, enemies: list[dict], army: list[dict], enemy_base: tuple[int, int] | None):
        chosen = self.knobs.get("target")
        if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
            return (int(chosen[0]), int(chosen[1]))
        centre = self._centroid([c for c in (_xy(u) for u in army) if c])
        visible_b = [c for c in (_xy(b) for b in self._enemy_buildings.values()) if c]
        if centre and visible_b:
            return min(visible_b, key=lambda c: _dist(c, centre))
        visible_u = [c for c in (_xy(e) for e in enemies) if c]
        if centre and visible_u:
            nearest = min(visible_u, key=lambda c: _dist(c, centre))
            if _dist(nearest, centre) < 20:
                return nearest
        return enemy_base

    # -- geometry ------------------------------------------------------------------------

    def _base(self, obs: dict, buildings: list[dict], units: list[dict]) -> tuple[int, int] | None:
        bc = obs.get("base_center")
        if isinstance(bc, (list, tuple)) and len(bc) == 2:
            return int(bc[0]), int(bc[1])
        if isinstance(bc, dict) and "x" in bc:
            return int(bc["x"]), int(bc["y"])
        cells = [c for c in (_xy(a) for a in buildings or units) if c]
        return self._centroid(cells)

    def _enemy_base(self, la: dict, base: tuple[int, int] | None) -> tuple[int, int] | None:
        eb = la.get("enemy_base")
        if isinstance(eb, (list, tuple)) and len(eb) == 2:
            self._enemy_base_cache = (int(eb[0]), int(eb[1]))
        return getattr(self, "_enemy_base_cache", None)

    def _staging(self, base, enemy_base, raw) -> tuple[int, int] | None:
        if not base:
            return None
        if not enemy_base:
            return base
        return self._toward(base, enemy_base, STAGING_DISTANCE, raw)

    def _toward(self, a, b, distance: float, raw: dict, jitter: int = 0) -> tuple[int, int]:
        d = _dist(a, b) or 1.0
        x = a[0] + (b[0] - a[0]) * distance / d
        y = a[1] + (b[1] - a[1]) * distance / d
        if jitter:
            # Spread defenses sideways (perpendicular to the enemy direction).
            px, py = -(b[1] - a[1]) / d, (b[0] - a[0]) / d
            side = (jitter % 2) * 2 - 1
            x += px * 3 * side * ((jitter + 1) // 2)
            y += py * 3 * side * ((jitter + 1) // 2)
        return self._clamp((int(round(x)), int(round(y))), raw)

    def _clamp(self, cell: tuple[int, int], raw: dict) -> tuple[int, int]:
        bounds = (raw.get("map") or {}).get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            bx, by, bw, bh = (int(v) for v in bounds)
            return (min(max(cell[0], bx + 1), bx + bw - 2), min(max(cell[1], by + 1), by + bh - 2))
        return cell

    @staticmethod
    def _centroid(cells: list[tuple[int, int]]) -> tuple[int, int] | None:
        if not cells:
            return None
        return (int(sum(c[0] for c in cells) / len(cells)), int(sum(c[1] for c in cells) / len(cells)))

    def _remember_enemy_buildings(self, seen: list[dict], units: list[dict], buildings: list[dict]) -> None:
        for b in seen:
            if isinstance(b.get("id"), int):
                self._enemy_buildings[b["id"]] = b
        # Forget remembered buildings whose cell one of our units can see but
        # which are no longer reported (destroyed).
        seen_ids = {b.get("id") for b in seen}
        ours = [c for c in (_xy(u) for u in units) if c]
        for bid, b in list(self._enemy_buildings.items()):
            c = _xy(b)
            if bid not in seen_ids and c and any(_dist(c, o) < 4 for o in ours):
                del self._enemy_buildings[bid]

    # -- logging ---------------------------------------------------------------------------

    def _status(self, raw: dict, obs: dict) -> None:
        now = time.monotonic()
        if now - self._last_status < STATUS_EVERY_S:
            return
        self._last_status = now
        econ = obs.get("economy") or {}
        mil = obs.get("military") or {}
        t = raw.get("time") or {}
        army = sum(1 for u in obs.get("units") or [] if isinstance(u, dict) and self._is_combat(u))
        self.log(
            f"[bot] t={t.get('game_seconds')}s left={t.get('remaining_seconds')}s cash={econ.get('cash')} "
            f"power={econ.get('power_provided')}/{econ.get('power_drained')} harv={econ.get('harvesters')} "
            f"army={army} buildings={len(obs.get('buildings') or [])} "
            f"kills={mil.get('kills_cost')} deaths={mil.get('deaths_cost')} wave={self._wave} "
            f"attacking={self._attacking} knobs={self.knobs.get('posture')}/{self.knobs.get('attack_size')}"
        )


def dump_dir_from_env() -> str | None:
    return os.environ.get("REDALERT_DUMP_DIR") or "logs"
