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
import re
import time
from collections import Counter, deque
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
    "dome": 1500, "gun": 800, "pbox": 600, "hbox": 600, "ftur": 600, "tsla": 1200, "silo": 150,
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
# Openings: "rush" (mass infantry from one barracks, hit the enemy at ~1:40,
# then macro) or "macro". REDALERT_OPENING=rush | macro | seat0:rush,seat1:macro.
DEFAULT_OPENING = "rush"
RUSH_LAUNCH_AT = 6          # infantry needed to launch the rush ...
RUSH_LAUNCH_LATE = (150, 5) # ... or, after 150 s, this many
RUSH_UNTIL_S = 420          # the rush opening ends by this game time
# Raids: while the main army waits, a small squad hunts enemy harvesters.
RAID_SIZE = 4
RAID_MIN_ARMY = 5           # don't raid with a smaller army at home
RAID_EVERY_S = 20.0
RAID_PULLBACK_COOLDOWN_S = 20.0  # no new raid squad this long after one ran home
RAID_DETOUR = 16            # a raid flanks this many cells to the quieter side of the direct line
# Hit and run: approach -> strike (briefly, harvesters first) -> withdraw the
# moment responders arrive -> reset on the other flank -> approach again.
RAID_STRIKE_S = 6.0         # hit for a few seconds, then go, whatever happens
RAID_RESPONSE_RANGE = 9     # enemy combat units this close to the squad count as responders
RAID_ALERT_RANGE = 13       # ... and these, if they are coming at us
RAID_WITHDRAW_RATIO = 0.35  # withdraw when responders reach this share of the squad's strength
RAID_HURT_SHARE = 0.12      # ... or when the squad has lost this share of its hp since the strike began
RAID_SAFE_RANGE = 14        # withdrawn once every enemy combat unit is this far away ...
RAID_NOT_CHASED_S = 2.0     # ... or none has come closer for this long (they stopped chasing)
RAID_WITHDRAW_MAX_S = 12.0  # or after this long
RAID_RESET_S = 3.0          # a short breath (and top-up), then straight back in
RAID_FAST = 2.0             # raid squads are units at least this fast (cells/s); infantry is 1.3
RAID_WITHDRAW_STEP = 16     # cells per withdraw order
RAID_TARGET_VALUE = {"harv": 10.0, "mcv": 8.0, "arty": 6.0, "v2rl": 6.0, "e1": 1.5, "e3": 1.2, "e6": 2.0,
                     "medi": 2.0, "jeep": 1.0, "ftrk": 0.8, "apc": 0.6}
RAID_DANGER_RANGE = 14      # enemies this close to a route point count as danger on it
HOME_IDLE_MAX_S = 40.0      # an army waiting at home longer than this attacks if not weaker
# The field commander (an LLM) decides production and army moves; its last
# decision counts for this long. Silent longer, the bot's own rules take over.
LLM_STALE_S = 20.0
CRUSHED = 0.4               # reflex: the main army retreats when this much weaker in its fight
POWER_SURPLUS_REFUSE = 100  # a commanded power plant is refused while the margin is this big
# Concentration of force ("many beat few"): fight only where we outnumber.
SUPERIOR = 1.5              # attack at any size when this much stronger than their known army
WAVE_EDGE = 1.2             # a full-size wave still needs this edge over their known army
ENEMY_MEMORY_S = 90.0       # how long a seen enemy unit counts towards their army
COHESION_RADIUS = 8         # units this close to the army's centre are "with the group"
COHESION_SHARE = 0.7        # the group advances only with this share of the army's strength
COHESION_REACH = 20         # ... counting only units this close to the group (not reinforcements at home)
LEG = 8                     # the group advances this many cells at a time
AHEAD_EDGE = 1.3            # ... and only when this much stronger than what is ahead
HOLD_GIVE_UP_S = 25.0       # outnumbered ahead this long: give up the attack
# Kiting (hit and run): a unit that is faster than, or outranges, the enemy
# chasing it runs while the enemy closes in and turns to shoot when the enemy
# stops or turns back.
KITE_WATCH = 3.0            # look at enemies within their range + this
KITE_MARGIN = 1.2           # run when the enemy is within its range + this and closing
KITE_STEP = 4               # cells to run per order
KITE_REISSUE_FLEE = 0.8     # seconds before repeating the same run order
KITE_REISSUE_FIGHT = 1.5
KITE_MAX_ORDERS = 10
# Weapon range in cells and speed, used when the server's rules don't say.
DEFAULT_RANGE = {
    "e1": 5.0, "e2": 4.0, "e3": 5.0, "e4": 3.5, "e7": 5.0, "dog": 1.5, "shok": 5.0, "medi": 0, "e6": 0,
    "spy": 0, "thf": 0, "harv": 0, "mcv": 0, "truk": 0, "mnly": 0,
    "1tnk": 4.75, "2tnk": 4.75, "3tnk": 4.75, "4tnk": 4.75, "ttnk": 6.0, "ctnk": 6.0, "stnk": 5.0,
    "jeep": 4.0, "apc": 4.0, "ftrk": 5.0, "v2rl": 10.0, "arty": 12.0, "qtnk": 3.0, "dtrk": 1.0,
}
# The server lists each unit's MAIN weapon only. For these that is an
# anti-air weapon, so their ground range comes from DEFAULT_RANGE instead
# (e3 fires Dragon rockets at ground targets, 5 cells; ftrk its ground flak).
ANTI_AIR_WEAPONS = re.compile(r"redeye|nike|-aa\b|\baa\b|zsu|stinger", re.I)
DEFAULT_SPEED = {
    "e1": 1.32, "e2": 1.32, "e3": 1.32, "e4": 1.32, "e6": 1.32, "e7": 1.6, "medi": 1.32, "shok": 1.32,
    "spy": 1.32, "thf": 1.32, "dog": 2.6, "harv": 1.76, "mcv": 1.3,
    "1tnk": 2.76, "2tnk": 1.76, "3tnk": 1.56, "4tnk": 1.05, "ttnk": 2.25, "jeep": 4.0, "apc": 3.12,
    "ftrk": 2.76, "v2rl": 1.76, "arty": 1.76, "ctnk": 2.0, "stnk": 2.5, "qtnk": 1.0, "dtrk": 1.76,
}
FIGHT_RADIUS = 12           # local strength is measured within this range
# Which enemy building the army shoots first once inside their base (lower
# first): defenses that shoot back, then power, production, economy, the yard.
STRUCTURE_PRIORITY = {
    "tsla": 0, "gun": 0, "ftur": 0, "pbox": 0, "hbox": 0, "agun": 1, "sam": 1,
    "powr": 2, "apwr": 2, "weap": 3, "barr": 3, "tent": 3, "proc": 4, "fix": 5, "dome": 5, "fact": 6,
}
STRUCTURE_FOCUS_RANGE = 9
UNIT_ENGAGE_RANGE = 12      # a unit with an enemy unit this close fights units, never buildings
MAX_MICRO_ORDERS = 6
CONTACT_RANGE = 10          # the main army is "in contact" with enemies this close to its centre
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
        self.opening = DEFAULT_OPENING
        self._rush_active = False
        self._rush_launched = False
        self._raiders: set[int] = set()
        self._raid_target: tuple[int, int] | None = None
        self._raid_side: int | None = None       # which flank the current squad sneaks along
        self._raid_phase = "approach"            # approach | strike | withdraw | reset
        self._raid_phase_since = 0.0
        self._raid_strike_until = 0.0
        self._raid_info: dict = {}               # what the squad is doing, for the commander
        self._raid_hp_start = 0.0                # squad hp when the strike began
        self._raid_last_chased = 0.0             # when an enemy last came closer to the squad
        self._no_queued = False                  # the server refused queued orders
        self._enemy_ore: tuple[int, int] | None = None
        self._enemy_harv_seen: dict[int, tuple[tuple[int, int], float]] = {}
        self._config: dict | None = None
        self._game_s = 0.0
        self._home_since: float | None = None
        self._enemy_hist: dict[int, deque] = {}
        self._kite_state: dict[int, tuple[str, Any, float]] = {}
        self._range_cache: dict[str, float] = {}
        self._idle_acc: Counter = Counter()
        self._last_tick: float | None = None
        self._session = ""
        self._retreat_until = 0.0
        self._enemy_army: dict[int, dict] = {}   # enemy combat units seen recently
        self._enemy_army_seen_at = -1e9          # when we last saw any enemy combat unit
        self._held_since: float | None = None
        self._decision: dict | None = None           # the field commander's latest decision
        self._consumed: set[tuple[int, str]] = set() # one-shot parts of decisions already done
        self._main_state = "waiting for orders"
        self._problems: deque = deque(maxlen=5)      # commander choices we couldn't carry out
        self._catalog: dict[str, list] = {}
        self._threat = 0.0
        self._started_at: dict[str, float] = {}      # when we last started each building
        self._prod_seen: dict[str, list[float]] = {} # item -> [first, last] time seen in a queue (this run)
        self._holding_since: float | None = None     # game time the main army started idling at home
        self._onward: tuple[int, int] | None = None  # where an attack carries on after clearing its target
        self._onward_for: tuple | None = None        # the commander's target that onward belongs to
        self._retreat_issued = False
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
            self._session = str(context.session_id or "")[:8]
            self._load_config(context.game_config, raw, context.seat_position)
        if self.advisor is not None:
            self.advisor.observe(raw, self.status())
            self.knobs.update(self.advisor.knobs())
            for cmd in self.advisor.pop_commands():
                if cmd.get("type") == "focus":
                    self._focus = cmd
                else:
                    self._commands.append(cmd)
            decide = getattr(self.advisor, "decision", None)
            self._decision = decide() if callable(decide) else None
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
            detail = str(result.get("detail"))
            if "target_not_visible" in detail:
                # The focus target walked out of sight: drop it, pick again next time.
                self._focus = None
                self._sent["focus"] = time.monotonic() + 1.0
                return
            self.log(f"[bot] batch refused: {detail[:200]}")
            # Block every item order of that batch for a moment.
            for order in self._last_orders:
                if order.get("item"):
                    self._block(str(order["item"]), 4.0)
            return
        if result.get("error") is None and not rows:
            return
        for row in rows:
            if isinstance(row, dict) and row.get("status") == "rejected" and "queued" in str(row.get("detail", "")):
                self._no_queued = True
            if isinstance(row, dict) and row.get("reason") == "target_not_visible":
                self._focus = None
                self._sent["focus"] = time.monotonic() + 1.0
            if not isinstance(row, dict) or row.get("status") != "rejected":
                continue
            idx = row.get("index")
            order = self._last_orders[idx] if isinstance(idx, int) and idx < len(self._last_orders) else {}
            item = order.get("item")
            if item:
                self._block(str(item), 8.0 if row.get("reason") == "not_available" else 3.0)

    def status(self) -> dict:
        """What the bot is doing, shown to the Claude layers."""
        return {
            "opening": self.opening, "rush_active": self._rush_active, "rush_launched": self._rush_launched,
            "attacking": self._attacking, "wave": self._wave, "raiders": len(self._raiders),
            "enemy_army_known_strength": round(self.enemy_army_estimate()[0]),
            "our_army_strength": round(getattr(self, "_our_strength", 0)),
            "main_state": self._main_state,
            "situation": self._situation(),
            "threat_near_base_strength": round(self._threat),
            "enemy_ore_field": list(self._enemy_ore) if self._enemy_ore else None,
            "raid_squad": len(self._raiders),
            "raid": self._raid_info,
            "main_idle_seconds": round(self._game_s - self._holding_since) if self._holding_since is not None else 0,
            "idle_combat_units": getattr(self, "_idle_combat", 0),
            "strength_ahead": round(getattr(self, "_ahead", 0)),
            "our_ore_field": list(self._our_ore) if getattr(self, "_our_ore", None) else None,
            "known_enemy_buildings": [
                {"type": b.get("type"), "x": b.get("x"), "y": b.get("y")} for b in list(self._enemy_buildings.values())[:20]
            ],
            "catalog": self._catalog,
            "problems": list(self._problems),
        }

    # -- setup -----------------------------------------------------------------------

    def _load_config(self, config: dict | None, raw: dict, seat: int | None = None) -> None:
        self._config_loaded = True
        self._config = config
        self.rules = _item_rules(config)
        self.opening = _opening_for_seat(os.environ.get("REDALERT_OPENING", ""), seat)
        self._rush_active = self.opening == "rush"
        # Keep the server's real reference (not a test stub), one file per match.
        real = len(((config or {}).get("rules") or {}).get("units") or {}) >= 10
        if self._dump_dir is not None and real:
            try:
                self._dump_dir.mkdir(parents=True, exist_ok=True)
                (self._dump_dir / "red_alert_config.json").write_text(json.dumps(config, indent=1))
                (self._dump_dir / f"{self._session}-first_state.json").write_text(json.dumps(raw, indent=1))
            except OSError:
                pass
        obs = raw.get("observation") or {}
        self.log(
            f"[bot] match start: faction={obs.get('faction')} vs {obs.get('enemy_faction')}, "
            f"map={(raw.get('map') or {}).get('id')}, rules for {len(self.rules)} items, opening={self.opening}"
        )

    def _cost(self, item: str) -> int:
        value = (self.rules.get(item) or {}).get("cost")
        return int(value) if isinstance(value, (int, float)) else DEFAULT_COST.get(item, 800)

    def _power(self, item: str) -> int:
        value = (self.rules.get(item) or {}).get("power")
        return int(value) if isinstance(value, (int, float)) else DEFAULT_POWER.get(item, 0)

    def _avoid(self) -> set[str]:
        """The general's avoid list, minus prerequisites of what it (or the
        commander) asked for: avoiding the service depot while asking for heavy
        tanks would block them forever."""
        avoid = {str(a).lower() for a in self.knobs.get("avoid") or ()}
        if not avoid:
            return avoid
        wanted = [i for i, _ in self.knobs.get("build_targets") or ()]
        d = self._llm()
        if d:
            wanted += [d.get(k) for k in ("build", "defense", "infantry", "vehicle") if d.get(k) not in ("", "auto", None)]
        needed: set[str] = set()
        for item in wanted:
            for pre in (self.rules.get(item) or {}).get("prerequisites") or []:
                needed |= {w for w in re.findall(r"[a-z0-9]+", str(pre).lower())}
        return avoid - needed

    def _situation(self) -> str:
        ours = getattr(self, "_our_strength", 0)
        theirs, known = self.enemy_army_estimate()
        ratio = f"{ours / theirs:.1f}x" if theirs else ("no enemy army seen" if not known else "their army is gone")
        held = f", main army idle {self._game_s - self._holding_since:.0f}s" if self._holding_since is not None else ""
        big, cell, age = self.enemy_largest_group()
        group = ""
        if cell:
            eb = getattr(self, "_enemy_base_cache", None)
            ob = getattr(self, "_base_cell", None)
            mc = getattr(self, "_main_centre", None)
            where = (f", {_dist(cell, eb):.0f} cells from their base" if eb else "") + (
                f", {_dist(cell, ob):.0f} from our base" if ob else "") + (
                f", {_dist(cell, mc):.0f} from our main army" if mc else "")
            group = f"; largest enemy group seen: {big:.0f} at {list(cell)} {age:.0f}s ago{where}"
        if self._raid_info:
            r = self._raid_info
            group += (f"; raid squad ({r.get('size')}): {r.get('phase')} at {r.get('at')}, responders {r.get('responders')}"
                      + (f" ({r.get('responders_from_their_base')} cells from their base)" if r.get("responders") else ""))
        return (f"our army {ours:.0f} vs their known army {theirs:.0f} ({ratio}); at our base {self._threat:.0f}; "
                f"main army: {self._main_state}{held}; ahead of it {getattr(self, '_ahead', 0):.0f}{group}")

    def _llm(self) -> dict | None:
        """The field commander's decision while fresh, else None (rules take over)."""
        d = self._decision
        if d and time.monotonic() - d.get("t", 0) < LLM_STALE_S:
            return d
        return None

    def _problem(self, text: str) -> None:
        if not self._problems or self._problems[-1] != text:
            self._problems.append(text)

    def _queue_of(self, item: str) -> str | None:
        """Which production queue makes ``item`` (from the rules, else by name)."""
        q = (self.rules.get(item) or {}).get("queue")
        if isinstance(q, str):
            return q
        if item in DEFENSE_TYPES:
            return "Defense"
        if item in DEFAULT_POWER or item in ("fact",):
            return "Building"
        if item in ("e1", "e2", "e3", "e4", "e6", "e7", "medi", "dog", "shok", "spy", "thf", "mech"):
            return "Infantry"
        if item in TANKS + ARTILLERY + ("harv", "mcv", "4tnk", "mnly", "truk"):
            return "Vehicle"
        return None

    def _range(self, actor: dict) -> float:
        t = str(actor.get("type", "")).lower()
        if t not in self._range_cache:
            r = _weapon_range(self.rules.get(t))
            self._range_cache[t] = r if r is not None else DEFAULT_RANGE.get(t, 4.0)
        return self._range_cache[t]

    def _speed(self, actor: dict) -> float:
        t = str(actor.get("type", "")).lower()
        v = (self.rules.get(t) or {}).get("speed")
        return float(v) if isinstance(v, (int, float)) else DEFAULT_SPEED.get(t, 1.5)

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

        game_s = (raw.get("time") or {}).get("game_seconds")
        self._game_s = float(game_s) if isinstance(game_s, (int, float)) else self._game_s
        if self._rush_active and self._game_s > RUSH_UNTIL_S:
            self._rush_active = False
        self._visible_enemy_buildings = enemy_buildings
        base = self._base(obs, buildings, units)
        self._base_cell = base
        enemy_base = self._enemy_base(la, base)
        if base and getattr(self, "_our_ore", None) is None and self._config:
            self._our_ore = _ore_near(self._config, base) or False
        staging = self._staging(base, enemy_base, raw)
        self._remember_enemy_buildings(enemy_buildings, units, buildings)
        self._remember_enemy_harvesters(enemies)
        self._track_enemies(enemies, units)

        orders: list[dict] = []
        notes: list[str] = []
        btypes = Counter(str(b.get("type", "")).lower() for b in buildings)
        self._btypes = btypes
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
        self._account_idle(btypes, busy_queues, money)
        now = time.monotonic()
        for item in queued:
            seen = self._prod_seen.get(item)
            if seen and now - seen[1] < 2.0:
                seen[1] = now
            else:
                self._prod_seen[item] = [now, now]
        self._catalog = {i: [self._cost(i), self._queue_of(i)] for i in sorted(available)}
        d = self._llm()

        # 3. Build order (Building queue: one at a time).
        if has_yard and "Building" not in busy_queues and not self._recent("build", 1.0):
            if d and d["build"] != "auto":
                item = self._commanded("build", d, available, power_margin, queued)
            else:
                item = self._next_building(btypes, queued, available, money, power_margin, econ, raw)
            if item:
                orders.append({"cmd": "build", "item": item})
                self._mark("build")
                self._mark(f"start:{item}")
                self._started_at[item] = time.monotonic()
                notes.append(f"building {item}")

        # 4. Static defenses (Defense queue).
        want_def = int(self.knobs.get("defenses", 2))
        if self.knobs.get("posture") == "defensive":
            want_def += 2
        n_def = sum(btypes[t] for t in DEFENSE_TYPES) + sum(queued[t] for t in DEFENSE_TYPES)
        if d and d["defense"] != "auto":
            if has_yard and "Defense" not in busy_queues and not self._recent("defense", 1.5):
                item = self._commanded("defense", d, available, power_margin, queued)
                if item:
                    orders.append({"cmd": "build", "item": item})
                    self._mark("defense")
                    notes.append(f"building defense {item} (commander)")
        elif (
            has_yard
            and "Defense" not in busy_queues
            and n_def < want_def
            and btypes["proc"] >= 1
            # Defenses wait for the war factory (unless told to turtle).
            and (btypes["weap"] >= 1 or self.knobs.get("posture") == "defensive")
            and money > 1200
            and not self._recent("defense", 1.5)
        ):
            item = self._pick_defense(available, power_margin)
            if item:
                orders.append({"cmd": "build", "item": item})
                self._mark("defense")
                notes.append(f"building defense {item}")

        # 5. Units.
        harvesters = [u for u in units if str(u.get("type")).lower() == "harv"]
        army = [u for u in units if self._is_combat(u)]
        # A key building waiting for money gets it before units do.
        critical = {"powr", "apwr", "tent", "barr", "proc", "weap", "fix"}
        starving = money < 400 and any(
            p.get("queue") == "Building" and str(p.get("item", "")).lower() in critical and not p.get("ready")
            for p in production
        )
        orders += self._train(available, busy_queues, queued, btypes, harvesters, army, money, notes, starving, d)

        # 6. Economy upkeep: idle harvesters back to work, repairs, rally points.
        idle_harv = [u["id"] for u in harvesters if u.get("idle")]
        if idle_harv and not self._recent("harvest", 2.0):
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

        # 7. The army: the commander's orders, else the bot's own rules.
        if d:
            orders += self._army_llm(d, raw, obs, army, enemies, buildings, base, enemy_base, staging, notes)
        else:
            orders += self._command_army(raw, obs, army, enemies, buildings, base, enemy_base, staging, notes)

        return orders, "; ".join(notes) or "macro"

    # -- economy -------------------------------------------------------------------------

    def _commanded(self, key: str, d: dict, available: set[str], power_margin: int, queued: Counter) -> str | None:
        """The commander's building/defense choice, once per decision. Reflex:
        power first when it has run out."""
        if power_margin < 0 and not any(queued[p] for p in POWER):
            power = next((p for p in POWER if p in available and not self._blocked_now(p)), None)
            if power:
                return power
        want = d.get(key) or ""
        if not want:
            return None
        if (d["seq"], key) in self._consumed:
            return self._general_next(available, queued) if key == "build" else None
        self._consumed.add((d["seq"], key))
        # A repeat of a building we just started, decided before the commander
        # could see it start, is the same order again, not a second copy. The
        # same goes for one named while it was still in production: "proc"
        # said while a refinery builds means that one, not another.
        asked = d.get("asked_at", 0)
        started = self._started_at.get(want)
        seen = self._prod_seen.get(want)
        if (started is not None and asked < started + 1.0) or (seen and seen[0] <= asked <= seen[1] + 1.5):
            return self._general_next(available, queued) if key == "build" else None
        if want not in available:
            self._problem(f"{key} {want}: not available now")
            return None
        if self._blocked_now(want):
            self._problem(f"{key} {want}: just refused by the server")
            return None
        drain = self._power(want)
        if want in POWER and power_margin >= POWER_SURPLUS_REFUSE:
            self._consumed.discard((d["seq"], key))  # still pending, should power drop
            self._problem(f"{want} not built: power margin is already +{power_margin}")
            return self._general_next(available, queued) if key == "build" else None
        if drain < 0 and power_margin + drain < 0 and want not in POWER and not any(queued[p] for p in POWER):
            power = next((p for p in POWER if p in available and not self._blocked_now(p)), None)
            if power:
                self._consumed.discard((d["seq"], key))  # the order stands; power goes first
                self._problem(f"{want} would drop power to {power_margin + drain}: building {power} first")
                return power
        return want

    def _general_next(self, available: set[str], queued: Counter) -> str | None:
        """Between commander orders, the general's build targets keep the
        Building queue busy (still an LLM decision, just the slower one)."""
        have = getattr(self, "_btypes", Counter())
        for item, wanted in self.knobs.get("build_targets") or ():
            if self._queue_of(item) == "Building" and have[item] + queued[item] < wanted \
                    and item in available and not self._blocked_now(item) and not self._recent(f"start:{item}", 5.0):
                return item
        return None

    def _next_building(self, have: Counter, queued: Counter, available: set[str], money: int,
                       power_margin: int, econ: dict, raw: dict) -> str | None:
        def count(*items: str) -> int:
            return sum(have[i] + queued[i] for i in items)

        avoid = self._avoid()

        def ok(item: str) -> bool:
            return item in available and not self._blocked_now(item) and item not in avoid

        def power_item() -> str | None:
            for p in POWER:
                if item_ok(p):
                    return p
            return None

        def item_ok(item: str) -> bool:  # power is never avoided
            return item in available and not self._blocked_now(item)

        barracks = next((b for b in BARRACKS if ok(b)), None)
        steps: list[tuple[bool, str | None]] = []
        steps.append((count(*POWER) == 0, power_item()))
        # The strategy layer's build targets come first (power stays positive).
        steps.append((power_margin < 30 and bool(self.knobs.get("build_targets")), power_item()))
        for item, wanted in self.knobs.get("build_targets") or ():
            if self._queue_of(item) == "Building":
                steps.append((count(item) < wanted, item))
        steps.append((count(*BARRACKS) == 0, barracks))
        steps.append((count("proc") == 0, "proc"))
        # Keep power ahead of what the next building drains.
        steps.append((power_margin < 30 or (power_margin < 60 and count(*POWER) < 4), power_item()))
        # Two refineries (each comes with a harvester) before the war factory:
        # one harvester can't pay for anything.
        steps.append((count("proc") < 2, "proc"))
        rushing = self._rush_active and not self._rush_launched and self._game_s < 120
        steps.append((count("weap") == 0 and not rushing, "weap"))
        steps.append((count("fix") == 0, "fix"))
        max_proc = int(self.knobs.get("max_refineries", 3))
        if self.knobs.get("posture") == "economy":
            max_proc += 1
        steps.append((power_margin < 80 and money > 1000, power_item()))
        steps.append((count("proc") < max_proc and money > 1400, "proc"))
        steps.append((count("weap") < 2 and money > 1500 and self.knobs.get("posture") != "economy", "weap"))
        steps.append((count("dome") == 0 and money > 1500, "dome"))
        steps.append((count(*BARRACKS) < 2 and money > 1200, barracks))
        # Money piling up: more factories, so it turns into units faster.
        steps.append((count("weap") < 4 and money > 3000, "weap"))
        steps.append((count(*BARRACKS) < 3 and money > 3000, barracks))
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
               harvesters: list[dict], army: list[dict], money: int, notes: list[str],
               starving: bool = False, d: dict | None = None) -> list[dict]:
        orders: list[dict] = []
        handled: set[str] = set()
        # The commander's choices: each queue keeps producing what it said.
        if d:
            for queue, key in (("Vehicle", "vehicle"), ("Infantry", "infantry")):
                want = d.get(key, "auto")
                if want == "auto":
                    continue
                handled.add(queue)
                if queue in busy or self._recent(key, 0.8):
                    continue
                item = want
                # Reflex: every refinery keeps at least one harvester.
                if queue == "Vehicle" and have["proc"] and len(harvesters) + queued["harv"] < have["proc"] \
                        and "harv" in available:
                    item = "harv"
                if not item:
                    continue
                if item not in available or self._blocked_now(item):
                    self._problem(f"{key} {item}: not available now")
                    continue
                count = 1 if item == "harv" else int(d.get(f"{key}_count", 1))
                orders.append({"cmd": "train", "item": item, "count": count})
                self._mark(key)
                notes.append(f"training {item} x{count} (commander)")

        avoid_now = self._avoid()

        def ok(item: str) -> bool:
            return item in available and not self._blocked_now(item) and item not in avoid_now

        focus = self.knobs.get("unit_focus", "mixed")
        n_proc = have["proc"]
        targets = [(i, n) for i, n in self.knobs.get("build_targets") or () if self._queue_of(i) in ("Vehicle", "Infantry")]
        owned = Counter(str(u.get("type", "")).lower() for u in army + harvesters)

        def target_unit(queue: str) -> str | None:
            """A unit the strategy layer asked for, not yet owned in that number."""
            for item, wanted in targets:
                if self._queue_of(item) == queue and owned[item] + queued[item] < wanted and ok(item):
                    return item
            return None
        # Vehicles: harvesters first, then the best tank.
        if "Vehicle" not in busy and "Vehicle" not in handled and not self._recent("vehicle", 0.8):
            item = None
            want_harv = min(n_proc * 2, 8) if n_proc else 0
            asked = target_unit("Vehicle")
            if asked == "harv" or (asked and not starving):
                item = asked
            elif len(harvesters) + queued["harv"] < max(want_harv, n_proc + (1 if n_proc else 0)) and ok("harv"):
                item = "harv"
            elif not starving:
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
                count = 2 if money > 2500 and item != "harv" else 1
                orders.append({"cmd": "train", "item": item, "count": count})
                self._mark("vehicle")
                notes.append(f"training {item}")
        # Infantry: always some, more early or when the focus says so.
        if "Infantry" not in busy and "Infantry" not in handled and not self._recent("infantry", 0.8):
            n_inf = sum(1 for u in army if str(u.get("type")).lower() in ("e1", "e2", "e3", "e4"))
            has_factory = have["weap"] > 0
            # Before the war factory, money goes to buildings: only a few guards.
            cap = 30 if focus == "infantry" else (14 if has_factory else 5)
            if money > 1500:
                cap += 10  # spare money: spend it
            hold = starving or (not has_factory and money < 700)
            if self._rush_active and not self._rush_launched:
                cap, hold = RUSH_LAUNCH_AT + 2, starving or (not (have["proc"] + queued["proc"]) and money < 600)
            if n_inf < cap and not hold:
                n_e3 = sum(1 for u in army if str(u.get("type")).lower() == "e3")
                anti = focus == "anti_armor" or n_e3 * 2 < n_inf
                item = target_unit("Infantry") or (
                    ANTI_ARMOUR_INFANTRY if anti and ok(ANTI_ARMOUR_INFANTRY) else RIFLE_INFANTRY)
                if ok(item):
                    orders.append({"cmd": "train", "item": item, "count": 3 if money > 1500 else 2})
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
        self._retreat_issued = False
        micro, controlled = self._micro(raw, army, enemies, staging, base, notes)
        core = self._army_core(raw, obs, army, enemies, buildings, base, enemy_base, staging, notes)
        if self._retreat_issued:
            return core  # a retreat beats micro: everyone goes home
        # Units under micro control (running, kiting, pulled back) are left to it.
        kept = []
        for order in core:
            if "units" in order:
                ids = [i for i in order["units"] if i not in controlled]
                if not ids:
                    continue
                order = {**order, "units": ids}
            kept.append(order)
        return kept + micro

    def _micro(self, raw: dict, army: list[dict], enemies: list[dict], staging, base,
               notes: list[str]) -> tuple[list[dict], set[int]]:
        """Pulling back damaged units, kiting, and focus fire. Rules by default;
        a fast LLM layer's commands (``focus`` / ``pull_back``) take over when
        present. Returns the orders and the unit ids they control."""
        orders: list[dict] = []
        max_units = int((raw.get("limits") or {}).get("max_units_per_order") or 50)
        ours = {u["id"]: u for u in army if isinstance(u.get("id"), int) and _xy(u)}
        if not ours:
            return orders, set()
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

        # Kiting: units that out-run or out-range their attacker run and shoot.
        if time.monotonic() < self._retreat_until:
            return orders, pulled  # retreating: nobody turns back to fight
        # A raid squad on its way out is running already: micro leaves it alone.
        raid_away = set(self._raiders) if self._raid_phase in ("withdraw", "reset") else set()
        kite_orders, kiting = self._kite(raw, ours, enemies, pulled | raid_away, base)
        if kite_orders:
            orders += kite_orders
            notes.append(f"kiting with {len(kiting)} units")
        controlled = pulled | kiting

        # Focus fire: LLM target while it lasts, else the best target in the fight.
        if self._recent("focus", FOCUS_EVERY_S):
            return orders, controlled
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
            # No enemy units in the fight: wreck their buildings in the right order.
            wreck, wreckers = self._focus_structure(ours, controlled | set(self._raiders), max_units, notes)
            return orders + wreck, controlled | wreckers
        shooters = [i for i, c in cells.items()
                    if i not in controlled and i not in self._raiders and _dist(c, tcell) <= FOCUS_SHOOTER_RANGE]
        for chunk in _chunks(shooters, max_units):
            orders.append({"cmd": "attack", "units": chunk, "target": target["id"]})
        if shooters:
            self._mark("focus")
            notes.append(f"focus fire on {target.get('type')} {target['id']}")
        engaged = set(shooters)
        # Everyone else with an enemy unit nearby fights units too (the best one
        # in its reach), never buildings while enemy units are around.
        by_target: dict[int, list[int]] = {}
        for i, c in cells.items():
            if i in controlled or i in engaged or i in raid_away:
                continue
            reach = [e for e in enemies if _xy(e) and isinstance(e.get("id"), int) and _dist(_xy(e), c) <= UNIT_ENGAGE_RANGE]
            if reach:
                score = self._raid_target_score if i in self._raiders else self._target_score
                by_target.setdefault(max(reach, key=score)["id"], []).append(i)
        for tid, ids in sorted(by_target.items(), key=lambda kv: -len(kv[1]))[:MAX_MICRO_ORDERS]:
            orders.append({"cmd": "attack", "units": ids[:max_units], "target": tid})
            engaged |= set(ids[:max_units])
        # Units with no enemy unit in reach wreck buildings in priority order.
        wreck, wreckers = self._focus_structure(ours, controlled | engaged | set(self._raiders), max_units, notes)
        return orders + wreck, controlled | engaged | wreckers

    # -- kiting ----------------------------------------------------------------------------

    def _can_kite(self, unit: dict, enemy: dict) -> bool:
        ru, re_ = self._range(unit), self._range(enemy)
        su, se = self._speed(unit), self._speed(enemy)
        if ru <= 0 or re_ <= 0 or su <= 0:
            return False
        return ru >= re_ + 1.0 or (su >= se * 1.3 and ru >= re_ - 0.75)

    def _kite(self, raw: dict, ours: dict[int, dict], enemies: list[dict], exclude: set[int],
              base) -> tuple[list[dict], set[int]]:
        now = time.monotonic()
        threats = [e for e in enemies if _xy(e) and isinstance(e.get("id"), int) and self._range(e) > 0]
        if not threats:
            self._kite_state.clear()
            return [], set()
        flee: dict[tuple[int, int], list[int]] = {}
        fight: dict[int, list[int]] = {}
        controlled: set[int] = set()
        for uid, u in ours.items():
            if uid in exclude:
                continue
            uc = _xy(u)
            near = [(e, _dist(_xy(e), uc)) for e in threats if _dist(_xy(e), uc) <= self._range(e) + KITE_WATCH]
            if not near:
                self._kite_state.pop(uid, None)
                continue
            # The enemy closest to having us in range.
            e, d = min(near, key=lambda p: p[1] - self._range(p[0]))
            if not self._can_kite(u, e):
                continue
            controlled.add(uid)
            re_ = self._range(e)
            approaching = self._approaching(e, uc)
            if self._range(u) >= re_ + 1.0:
                # We outrange it: keep out of its reach, shoot from beyond it.
                run = d <= re_ + 0.8 or (approaching and d <= re_ + KITE_MARGIN)
            else:
                # We're faster: run only while it chases; trade shots if it holds.
                run = approaching and d <= re_ + KITE_MARGIN
            if run:
                mode, key = "flee", None
            else:
                # It stopped or turned back: turn and shoot (best target in range, else it).
                ru = self._range(u)
                in_range = [x for x in enemies if _xy(x) and isinstance(x.get("id"), int) and _dist(_xy(x), uc) <= ru + 0.5]
                score = self._raid_target_score if uid in self._raiders else self._target_score
                key = (max(in_range, key=score) if in_range else e)["id"]
                mode = "fight"
            prev = self._kite_state.get(uid)
            wait = KITE_REISSUE_FLEE if mode == "flee" else KITE_REISSUE_FIGHT
            if prev and prev[0] == mode and prev[1] == key and now - prev[2] < wait:
                continue  # already doing exactly this
            self._kite_state[uid] = (mode, key, now)
            if mode == "flee":
                dest = self._flee_cell(uc, [_xy(x) for x, _ in near], base, raw)
                flee.setdefault(dest, []).append(uid)
            else:
                fight.setdefault(key, []).append(uid)
        orders = [{"cmd": "move", "units": ids, "to": list(dest)} for dest, ids in flee.items()]
        orders += [{"cmd": "attack", "units": ids, "target": tid} for tid, ids in fight.items()]
        return orders[:KITE_MAX_ORDERS], controlled

    def _flee_cell(self, uc, threat_cells, base, raw, step: int = KITE_STEP) -> tuple[int, int]:
        """A cell ``step`` away from the threats, leaning towards home."""
        ax = sum((uc[0] - t[0]) / (_dist(uc, t) or 1.0) for t in threat_cells)
        ay = sum((uc[1] - t[1]) / (_dist(uc, t) or 1.0) for t in threat_cells)
        n = math.hypot(ax, ay) or 1.0
        ax, ay = ax / n, ay / n
        if base:
            hx, hy = base[0] - uc[0], base[1] - uc[1]
            hn = math.hypot(hx, hy) or 1.0
            ax, ay = 0.75 * ax + 0.25 * hx / hn, 0.75 * ay + 0.25 * hy / hn
            n = math.hypot(ax, ay) or 1.0
            ax, ay = ax / n, ay / n
        x = uc[0] + ax * step
        y = uc[1] + ay * step
        # Round to a 2-cell grid so a group running together shares one order.
        return self._clamp((int(round(x / 2) * 2), int(round(y / 2) * 2)), raw)

    def _track_enemies(self, enemies: list[dict], units: list[dict] | None = None) -> None:
        now = time.monotonic()
        visible = set()
        for e in enemies:
            if isinstance(e.get("id"), int) and _xy(e):
                self._enemy_hist.setdefault(e["id"], deque(maxlen=10)).append((now, _xy(e)))
                visible.add(e["id"])
                if self._is_combat(e):
                    self._enemy_army[e["id"]] = {"type": e.get("type"), "hp": e.get("hp"), "x": e["x"], "y": e["y"], "t": now}
                    self._enemy_army_seen_at = now
        # An enemy that vanished right where our units can see is dead, not hiding.
        ours = [c for c in (_xy(u) for u in units or []) if c]
        for eid, e in list(self._enemy_army.items()):
            if eid in visible:
                continue
            if now - e["t"] > ENEMY_MEMORY_S or (now - e["t"] < 3.0 and any(_dist((e["x"], e["y"]), o) <= 4 for o in ours)):
                del self._enemy_army[eid]
        if len(self._enemy_hist) > 300:
            for eid in [i for i, h in self._enemy_hist.items() if now - h[-1][0] > 10]:
                del self._enemy_hist[eid]

    def enemy_largest_group(self) -> tuple[float, tuple[int, int] | None, float]:
        """(strength, cell, seconds ago) of the biggest cluster of enemy combat
        units seen in the last ENEMY_MEMORY_S seconds (within 12 cells)."""
        now = time.monotonic()
        units = [e for e in self._enemy_army.values() if now - e["t"] <= ENEMY_MEMORY_S]
        best, cell, age = 0.0, None, 0.0
        for e in units:
            group = [f for f in units if _dist((e["x"], e["y"]), (f["x"], f["y"])) <= 12]
            strength = sum(self._strength(f) for f in group)
            if strength > best:
                best, cell, age = strength, (e["x"], e["y"]), now - max(f["t"] for f in group)
        return best, cell, age

    def enemy_army_estimate(self) -> tuple[float, bool]:
        """(strength of the enemy army we know of, whether we've seen it lately)."""
        known = time.monotonic() - self._enemy_army_seen_at <= ENEMY_MEMORY_S
        return sum(self._strength(e) for e in self._enemy_army.values()), known

    def _approaching(self, enemy: dict, cell) -> bool:
        """Has this enemy moved towards ``cell`` over the last second?"""
        hist = self._enemy_hist.get(enemy.get("id"))
        if not hist or len(hist) < 2:
            return False
        now = time.monotonic()
        prev = next((c for t, c in hist if now - t <= 1.0), None)
        if prev is None or prev == hist[-1][1]:
            return False
        return _dist(prev, cell) - _dist(hist[-1][1], cell) >= 0.4

    # -- idle accounting -------------------------------------------------------------------

    def _account_idle(self, have: Counter, busy: set, money: int) -> None:
        now = time.monotonic()
        dt = 0.0 if self._last_tick is None else min(max(now - self._last_tick, 0.0), 2.0)
        self._last_tick = now
        acc = self._idle_acc
        acc["total"] += dt
        if have["fact"] and "Building" not in busy:
            acc["Building"] += dt
        if (have["tent"] or have["barr"]) and "Infantry" not in busy:
            acc["Infantry"] += dt
        if have["weap"] and "Vehicle" not in busy:
            acc["Vehicle"] += dt
        if money > 1500:
            acc["cash>1500"] += dt

    def _focus_structure(self, ours: dict[int, dict], controlled: set[int], max_units: int,
                         notes: list[str]) -> tuple[list[dict], set[int]]:
        """Units standing among enemy buildings shoot the most important one
        (defenses first, then power, production, refinery, yard), low hp first.
        Returns the orders and the ids they control."""
        buildings = [b for b in getattr(self, "_visible_enemy_buildings", []) if _xy(b) and isinstance(b.get("id"), int)]
        if not buildings:
            return [], set()
        cells = {i: _xy(u) for i, u in ours.items() if i not in controlled}
        near = [b for b in buildings if any(_dist(_xy(b), c) <= STRUCTURE_FOCUS_RANGE for c in cells.values())]
        if not near:
            return [], set()
        def key(b: dict):
            hp = b.get("hp") if isinstance(b.get("hp"), (int, float)) else 100
            return (STRUCTURE_PRIORITY.get(str(b.get("type", "")).lower(), 7), hp)
        target = min(near, key=key)
        shooters = [i for i, c in cells.items() if _dist(c, _xy(target)) <= STRUCTURE_FOCUS_RANGE + 2]
        if not shooters:
            return [], set()
        self._mark("focus")
        notes.append(f"wrecking {target.get('type')} {target['id']}")
        return [{"cmd": "attack", "units": chunk, "target": target["id"]} for chunk in _chunks(shooters, max_units)], set(shooters)

    @staticmethod
    def _raid_target_score(enemy: dict) -> float:
        """Raiders hit what hurts the economy and dies fast, never tanks."""
        value = RAID_TARGET_VALUE.get(str(enemy.get("type", "")).lower(), 0.3)
        hp = enemy.get("hp") if isinstance(enemy.get("hp"), (int, float)) else 100
        return value * (1.0 + (100 - hp) / 50.0)

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
        self._our_strength = sum(self._strength(u) for u in army)
        wave_size = self._wave_size(raw, army)
        remaining = ((raw.get("time") or {}).get("remaining_seconds"))
        mil = obs.get("military") or {}
        trade = int(mil.get("kills_cost") or 0) - int(mil.get("deaths_cost") or 0)
        ours_s, theirs_s = self._local_strength(army, enemies)
        winning = ours_s >= 1.5 * theirs_s

        # One-shot orders from the tactical layer.
        for cmd in [c for c in self._commands if c.get("type") in ("retreat", "attack_now")]:
            self._commands.remove(cmd)
            if cmd["type"] == "retreat" and self._attacking:
                if winning or (self._rush_active and size >= 3):
                    self.log(f"[bot] ignoring retreat: winning locally ({ours_s:.0f} vs {theirs_s:.0f})")
                    continue
                self._attacking = False
                orders += self._move_home(army_ids, staging, max_units)
                notes.append("retreat ordered by tactical layer")
                self.log("[bot] retreat ordered by tactical layer")
                return orders
            if cmd["type"] == "attack_now" and not self._attacking and size >= 4:
                wave_size = 0  # launches below

        if self._attacking:
            self._wave_peak = max(self._wave_peak, size)
            # Pull back only when the wave is spent AND losing the local fight.
            spent = size < max(3, int(self._wave_peak * 0.35))
            if spent and theirs_s > ours_s * 0.8 and self.knobs.get("posture") != "aggressive":
                self._attacking = False
                self._rush_active = False
                orders += self._move_home(army_ids, staging, max_units)
                notes.append(f"wave lost the fight ({ours_s:.0f} vs {theirs_s:.0f}), regrouping")
                self.log(f"[bot] wave {self._wave} lost the fight ({ours_s:.0f} vs {theirs_s:.0f}), regrouping")
                return orders
            if size == 0 or (spent and self._rush_active):
                self._rush_active = False
            # Late in a close match while ahead on trades: protect the lead.
            if isinstance(remaining, (int, float)) and remaining < 60 and trade > 1500:
                self._attacking = False
                orders += self._move_home(army_ids, staging, max_units)
                notes.append("holding the lead to the time limit")
                return orders
        launch, why = (False, "")
        if not self._attacking:
            launch, why = self._should_launch(army, size, wave_size, remaining)
        if launch:
            self._home_since = None
            self._attacking = True
            self._wave += 1
            self._wave_peak = size
            self._raiders.clear()  # raiders join the wave
            if self._rush_active:
                self._rush_launched = True
            self._sent.pop("attack_all", None)
            self._sent.pop("attack", None)
            kind = "rush" if self._rush_active else f"wave {self._wave}"
            notes.append(f"{kind}: attacking with {size} units ({why})")
            self.log(f"[bot] {kind}: attacking with {size} units at t={self._game_s:.0f}s ({why})")

        if self._attacking:
            if self._recent("attack", 2.0):
                return orders
            target = self._attack_target(enemies, army, enemy_base)
            if target:
                orders += self._advance(raw, army, enemies, target, staging, notes)
                self._mark("attack")
            return orders

        # Not attacking: a raid squad hunts harvesters, the rest gather.
        orders += self._raid(raw, army, enemies, enemy_base, staging, notes)
        if staging and not self._recent("gather", 4.0):
            idle_away = [
                u["id"] for u in army
                if isinstance(u.get("id"), int) and u["id"] not in self._raiders
                and u.get("idle") and _xy(u) and _dist(_xy(u), staging) > 4
            ]
            for chunk in _chunks(idle_away, max_units):
                orders.append({"cmd": "attack_move", "units": chunk, "to": list(staging)})
            if idle_away:
                self._mark("gather")
        return orders

    def _army_llm(self, d: dict, raw: dict, obs: dict, army: list[dict], enemies: list[dict],
                  buildings: list[dict], base, enemy_base, staging, notes: list[str]) -> list[dict]:
        """Carry out the commander's army orders, with the bot's reflexes:
        micro, home defense against raids, emergency retreat."""
        self._retreat_issued = False
        alive = {u["id"]: u for u in army if isinstance(u.get("id"), int) and _xy(u)}
        self._our_strength = sum(self._strength(u) for u in alive.values())

        def cell(x: int, y: int, default):
            return self._clamp((x, y), raw) if x >= 0 and y >= 0 else default

        # The raid squad's phase comes first: micro leaves a withdrawing squad alone.
        self._raiders &= set(alive)
        rtarget = cell(d["raid_x"], d["raid_y"], None) or self._raid_cell(enemy_base) or enemy_base
        if self._raiders:
            self._raid_phase_update(rtarget, alive, enemies)
        micro, controlled = self._micro(raw, army, enemies, staging, base, notes)
        if self._retreat_issued:
            return micro
        orders: list[dict] = []
        max_units = 50
        # Enemies at our base (home units fight them by reflex; raids wait).
        building_cells = [c for c in (_xy(b) for b in buildings) if c]
        threats = [e for e in enemies if _xy(e) and (
            (base and _dist(_xy(e), base) <= BASE_RADIUS)
            or any(_dist(_xy(e), bc) <= BUILDING_THREAT_RADIUS for bc in building_cells))]
        self._threat = sum(self._strength(e) for e in threats)

        # Raid squad: the size the commander chose, at least the general's
        # standing raid (unless enemies are at our base and everyone is needed).
        before = set(self._raiders)
        size = d["raid_size"]
        standing = int(self.knobs.get("standing_raid", 0) or 0)
        if standing > size and self._threat <= 0 and len(alive) >= standing + 4:
            size = standing
        if size <= 0 and self._raiders:
            if staging:
                orders.append({"cmd": "move", "units": sorted(self._raiders)[:max_units], "to": list(staging)})
            self._raiders.clear()
        elif len(self._raiders) > size:
            for uid in sorted(self._raiders)[size:]:
                self._raiders.discard(uid)
        elif len(self._raiders) < size and self._raid_phase in ("approach", "reset") \
                and not self._recent("raid_pullback", RAID_PULLBACK_COOLDOWN_S):
            if not self._raiders:
                self._raid_phase, self._raid_phase_since = "approach", time.monotonic()
            spare = [u for i, u in alive.items() if i not in self._raiders and u.get("idle")]
            want = str(d.get("raid_unit") or "").lower()
            if want and size >= 2 and DEFAULT_SPEED.get(want, 2.0) < RAID_FAST \
                    and any(self._speed(u) >= RAID_FAST for u in spare):
                self._problem(f"raid_unit {want} is too slow to hit and run; using fast units instead")
                want = ""
            if want:
                typed = [u for u in spare if str(u.get("type", "")).lower() == want]
                if not typed and spare:
                    self._problem(f"raid_unit {want}: none idle, using the fastest units instead")
                spare = typed or spare
            self._raid_side = None
            if size == 1 and not want:
                # A squad of one is a scout: the cheapest unit, never a tank.
                spare.sort(key=lambda u: self._cost(str(u.get("type", "")).lower()))
            else:
                spare.sort(key=self._speed, reverse=True)
            self._raiders |= {u["id"] for u in spare[: size - len(self._raiders)]}
        if self._raiders:
            if self._raiders != before:
                self._raid_phase_update(rtarget, alive, enemies)  # a new or changed squad: its phase now
            orders += self._run_raid(rtarget, alive, enemies, enemy_base, staging, raw, controlled)
        else:
            self._raid_info = {}

        main = [u for i, u in alive.items() if i not in self._raiders]
        self._idle_combat = sum(1 for u in main if u.get("idle"))
        self._main_centre = self._centroid([_xy(u) for u in main]) if main else None
        # Reflex: enemies at our base are fought by whoever is home.
        home = [u for u in main if base and _dist(_xy(u), base) <= 20]
        if threats and home and not self._recent("home_defense", 2.0):
            spot = min((_xy(e) for e in threats), key=lambda c: _dist(c, base))
            ids = [u["id"] for u in home if u["id"] not in controlled]
            for chunk in _chunks(ids, max_units):
                orders.append({"cmd": "attack_move", "units": chunk, "to": list(spot)})
            self._mark("home_defense")
            notes.append(f"home units fight {len(threats)} raiders at the base")
        busy_home = {u["id"] for u in home} if threats else set()

        action = d["main_action"]
        force = [u for u in main if u["id"] not in busy_home]
        # How long the army has been held at home: reported to the commander
        # (the playbook says not to idle); the decision stays the LLM's.
        if action == "hold" and not threats and len(force) >= 6:
            if self._holding_since is None:
                self._holding_since = self._game_s
        else:
            self._holding_since = None
        if action == "attack":
            ordered = cell(d["main_x"], d["main_y"], None) or self._attack_target(enemies, force, enemy_base)
            if self._onward_for != (d["main_x"], d["main_y"]):
                self._onward, self._onward_for = None, (d["main_x"], d["main_y"])
            target = self._onward or ordered
            if not self._attacking:
                self._attacking = True
                self._wave += 1
                self.log(f"[bot] commander attack #{self._wave} with {len(force)} units on {target}")
            ours_s, theirs_s = self._local_strength(force, enemies)
            if force and theirs_s > 0 and ours_s < CRUSHED * theirs_s:
                # Reflex: crushed in a fight, get out now; the commander sees it next.
                self._attacking = False
                self._main_state = f"emergency retreat: crushed ({ours_s:.0f} vs {theirs_s:.0f})"
                self.log(f"[bot] {self._main_state}")
                return micro + orders + self._move_home([u["id"] for u in force], staging, max_units)
            if target and force and not self._recent("attack", 2.0):
                # Reached the target and (almost) nothing is there: an attack
                # that has cleared its point carries on to the nearest known
                # enemy building, and keeps doing so until the commander gives
                # a new target, rather than standing in an empty field.
                centre = self._centroid([_xy(u) for u in force])
                at_target = centre is not None and _dist(centre, target) <= 5
                quiet = centre is not None and sum(
                    self._strength(e) for e in enemies if _xy(e) and _dist(_xy(e), centre) <= 10) < 300 and not any(
                    _dist(_xy(b), centre) <= STRUCTURE_FOCUS_RANGE
                    for b in getattr(self, "_visible_enemy_buildings", []) if _xy(b))
                if at_target and quiet:
                    onward = self._attack_target(enemies, force, enemy_base)
                    if onward and _dist(onward, target) > 5:
                        self.log(f"[bot] reached {list(target)}, nothing there; carrying on to {list(onward)}")
                        self._onward = onward
                        target = onward
                step = self._advance(raw, force, enemies, target, staging, notes, give_up=False, obey=True)
                orders += step
                self._mark("attack")
                if self._onward:
                    self._main_state = (f"ordered target {list(ordered)} was empty; carrying on to {list(self._onward)}: "
                                        + (notes[-1] if notes else ""))
                else:
                    self._main_state = f"attacking {list(target)}: " + (notes[-1] if notes else "")
        elif action == "hold":
            self._attacking = False
            self._onward = None
            spot = cell(d["main_x"], d["main_y"], staging)
            if spot and not self._recent("hold", 3.0):
                ids = [u["id"] for u in force if _dist(_xy(u), spot) > 4 and (u.get("idle") or not self._recent("hold_all", 10.0))]
                for chunk in _chunks(ids, max_units):
                    orders.append({"cmd": "attack_move", "units": chunk, "to": list(spot)})
                self._mark("hold")
                self._main_state = f"holding at {list(spot)}"
        elif action == "retreat":
            self._attacking = False
            away = [u["id"] for u in force if staging and _dist(_xy(u), staging) > 5]
            if away and not self._recent("retreat_cmd", 3.0):
                orders += self._move_home(away, staging, max_units)
                self._mark("retreat_cmd")
            self._main_state = "retreating home"
        elif action == "defend":
            self._attacking = False
            spot = min((_xy(e) for e in threats), key=lambda c: _dist(c, base)) if threats else (base or staging)
            if spot and not self._recent("defend_cmd", 2.0):
                for chunk in _chunks([u["id"] for u in force], max_units):
                    orders.append({"cmd": "attack_move", "units": chunk, "to": list(spot)})
                self._mark("defend_cmd")
            self._main_state = f"defending at {list(spot) if spot else '?'}"

        # Micro last wins; units it controls are left out of the orders above.
        kept = []
        for order in orders:
            if "units" in order and order["cmd"] != "move":
                ids = [i for i in order["units"] if i not in controlled]
                if not ids:
                    continue
                order = {**order, "units": ids}
            kept.append(order)
        return kept + micro

    def _should_launch(self, army: list[dict], size: int, wave_size: int, remaining) -> tuple[bool, str]:
        """Many beat few: go when clearly stronger than the enemy army we know of,
        whatever our size; never send a full wave into a stronger enemy."""
        ours = sum(self._strength(u) for u in army)
        theirs, known = self.enemy_army_estimate()
        if wave_size == 0:
            return True, "ordered by tactical layer"
        if self._rush_active and not self._rush_launched:
            return size >= wave_size, "rush"
        if isinstance(remaining, (int, float)) and remaining < 240 and size >= 4:
            return True, "last minutes"
        if known and size >= 3 and ours >= SUPERIOR * theirs:
            return True, f"their army is weak ({ours:.0f} vs {theirs:.0f})"
        if size >= wave_size and (not known or ours >= WAVE_EDGE * theirs):
            return True, f"wave ready ({ours:.0f} vs {theirs:.0f})"
        # Waiting at home too long: go, unless they are stronger.
        if size < 5 or (self.opening == "macro" and self._game_s < 300 and wave_size >= 999):
            self._home_since = None
            return False, ""
        if self._home_since is None:
            self._home_since = self._game_s
        elif self._game_s - self._home_since > HOME_IDLE_MAX_S and (not known or ours >= theirs):
            self._home_since = None
            return True, f"no idling at home ({ours:.0f} vs {theirs:.0f})"
        return False, ""

    def _advance(self, raw: dict, army: list[dict], enemies: list[dict], target, staging,
                 notes: list[str], give_up: bool = True, obey: bool = False) -> list[dict]:
        """Move as one group: the front is the half of the army nearest the
        target; everyone else joins it (never alone across the map). The front
        steps forward when it has left home, is in contact, or has most of the
        nearby strength with it; strung out en route, it waits a moment for the
        stragglers close behind. Outnumbered ahead for too long (rules mode),
        it calls the attack off."""
        max_units = 50
        mine = [u for u in army if _xy(u) and isinstance(u.get("id"), int)]
        if not mine:
            return []
        target = tuple(target)
        by_closeness = sorted(mine, key=lambda u: _dist(_xy(u), target))
        front = by_closeness[: max(1, (len(by_closeness) + 1) // 2)]
        xs = sorted(_xy(u)[0] for u in front)
        ys = sorted(_xy(u)[1] for u in front)
        core = (xs[len(xs) // 2], ys[len(ys) // 2])
        near = [u for u in mine if _dist(_xy(u), core) <= COHESION_RADIUS]
        far = [u for u in mine if _dist(_xy(u), core) > COHESION_RADIUS]
        s_reach = sum(self._strength(u) for u in mine if _dist(_xy(u), core) <= COHESION_REACH) or 1.0
        s_core = sum(self._strength(u) for u in near)
        orders: list[dict] = []
        for chunk in _chunks([u["id"] for u in far], max_units):
            orders.append({"cmd": "attack_move", "units": chunk, "to": list(core)})

        foes = [e for e in enemies if _xy(e)] + [
            b for b in getattr(self, "_visible_enemy_buildings", [])
            if str(b.get("type", "")).lower() in DEFENSE_TYPES and _xy(b)
        ]
        nxt = target if _dist(core, target) <= LEG else self._toward(core, target, LEG, raw)
        ahead = sum(self._strength(f) for f in foes if _dist(_xy(f), nxt) <= 10 or _dist(_xy(f), core) <= 8)
        self._ahead = ahead
        core_ids = [u["id"] for u in near]
        contact = any(_dist(_xy(e), core) <= CONTACT_RANGE for e in enemies if _xy(e)) or any(
            _dist(_xy(b), core) <= STRUCTURE_FOCUS_RANGE for b in getattr(self, "_visible_enemy_buildings", []) if _xy(b))
        at_home = staging is not None and _dist(core, staging) <= 12
        grouped = s_core >= COHESION_SHARE * s_reach
        if contact:
            dest, why = nxt, f"in contact, pressing on ({s_core:.0f} vs {ahead:.0f} ahead)"
            self._held_since = None
        elif at_home or grouped:
            if obey or s_core >= AHEAD_EDGE * ahead:
                dest, why = nxt, f"advancing ({s_core:.0f} vs {ahead:.0f} ahead)"
                self._held_since = None
            else:
                dest, why = core, f"outnumbered ahead ({s_core:.0f} vs {ahead:.0f}), waiting for reinforcements"
                if self._held_since is None:
                    self._held_since = self._game_s
                elif give_up and self._game_s - self._held_since > HOLD_GIVE_UP_S:
                    self._held_since = None
                    self._attacking = False
                    self.log(f"[bot] attack called off: still outnumbered ahead ({s_core:.0f} vs {ahead:.0f})")
                    notes.append("attack called off, too strong ahead")
                    return self._move_home([u["id"] for u in mine], staging, max_units)
        else:
            dest, why = core, f"regrouping en route ({len(near)} of {len(mine)} with the front)"
            self._held_since = None
        for chunk in _chunks(core_ids, max_units):
            orders.append({"cmd": "attack_move", "units": chunk, "to": list(dest)})
        notes.append(why)
        head = why.split(" (")[0]
        if head != getattr(self, "_last_advance_why", None):
            self._last_advance_why = head
            self.log(f"[bot] army: {why}; front at {list(core)}, {len(far)} joining, next step {list(dest)}")
        return orders

    def _move_home(self, ids: list[int], staging, max_units: int) -> list[dict]:
        if not staging or not ids:
            return []
        self._retreat_issued = True
        self._retreat_until = time.monotonic() + 6.0
        return [{"cmd": "move", "units": chunk, "to": list(staging)} for chunk in _chunks(ids, max_units)]

    def _strength(self, actor: dict) -> float:
        hp = actor.get("hp") if isinstance(actor.get("hp"), (int, float)) else 100
        return self._cost(str(actor.get("type", "")).lower()) * max(hp, 5) / 100.0

    def _local_strength(self, army: list[dict], enemies: list[dict]) -> tuple[float, float]:
        """The army's main body vs the enemy units and defenses it is fighting
        (within FIGHT_RADIUS). The main body is what stands within
        COHESION_REACH of the army's median position; a few stragglers next
        to the enemy base don't make the whole army "crushed"."""
        foes = list(enemies) + [
            b for b in getattr(self, "_visible_enemy_buildings", [])
            if str(b.get("type", "")).lower() in DEFENSE_TYPES
        ]
        foes = [f for f in foes if _xy(f)]
        mine = [u for u in army if _xy(u)]
        if not mine:
            return 0.0, 0.0
        xs = sorted(_xy(u)[0] for u in mine)
        ys = sorted(_xy(u)[1] for u in mine)
        core = (xs[len(xs) // 2], ys[len(ys) // 2])
        body = [u for u in mine if _dist(_xy(u), core) <= COHESION_REACH] or mine
        near_foes = [f for f in foes if any(_dist(_xy(f), _xy(u)) <= FIGHT_RADIUS for u in body)]
        near_mine = [u for u in body if any(_dist(_xy(u), _xy(f)) <= FIGHT_RADIUS for f in near_foes)] or body
        return sum(self._strength(u) for u in near_mine), sum(self._strength(f) for f in near_foes)

    def _raid(self, raw, army, enemies, enemy_base, staging, notes) -> list[dict]:
        orders: list[dict] = []
        alive = {u["id"]: u for u in army if isinstance(u.get("id"), int)}
        self._raiders &= set(alive)
        max_units = int((raw.get("limits") or {}).get("max_units_per_order") or 50)
        if self._raiders:
            squad = [alive[i] for i in self._raiders]
            ours_s, theirs_s = self._local_strength(squad, enemies)
            if theirs_s > ours_s * 1.2:
                orders += self._move_home(sorted(self._raiders), staging, max_units)
                notes.append("raid outgunned, pulling back")
                self.log(f"[bot] raid outgunned ({ours_s:.0f} vs {theirs_s:.0f}), pulling back")
                self._raiders.clear()
                return orders
            if not self._recent("raid", 4.0):
                target = self._raid_cell(enemy_base)
                idle = [i for i in self._raiders if alive[i].get("idle")]
                if target and idle:
                    # Arrived and nothing to hit: try the enemy base edge next.
                    if all(_xy(alive[i]) and _dist(_xy(alive[i]), target) < 5 for i in idle):
                        self._raid_target = enemy_base
                        target = enemy_base or target
                    orders += self._raid_orders(idle[:max_units], squad, target, raw)
                    self._mark("raid")
            return orders
        # Start a raid: fastest units first, from those waiting at home.
        if len(army) < RAID_MIN_ARMY or self._recent("raid_start", RAID_EVERY_S) or self._rush_active:
            return orders
        target = self._raid_cell(enemy_base)
        if not target:
            return orders
        speed = lambda u: float((self.rules.get(str(u.get("type", "")).lower()) or {}).get("speed") or 1.0)  # noqa: E731
        candidates = sorted((u for u in army if u.get("idle")), key=speed, reverse=True)[:RAID_SIZE]
        if len(candidates) < 3:
            return orders
        self._raiders = {u["id"] for u in candidates}
        self._raid_target = None
        self._raid_side = None
        orders += self._raid_orders(sorted(self._raiders), candidates, target, raw)
        self._mark("raid_start")
        self._mark("raid")
        notes.append(f"raiding enemy harvesters at {target}")
        self.log(f"[bot] raid: {len(self._raiders)} units to {target}")
        return orders

    def _raid_view(self, alive: dict[int, dict], enemies: list[dict]):
        squad = [alive[i] for i in self._raiders if i in alive and _xy(alive[i])]
        if not squad:
            return None
        centre = self._centroid([_xy(u) for u in squad])
        combat_foes = [e for e in enemies if _xy(e) and self._is_combat(e)]
        # Responders: combat units on top of us, or coming at us from a bit further.
        near_foes = [e for e in combat_foes if _dist(_xy(e), centre) <= RAID_RESPONSE_RANGE
                     or (_dist(_xy(e), centre) <= RAID_ALERT_RANGE and self._approaching(e, centre))]
        responders = sum(self._strength(e) for e in near_foes)
        squad_s = sum(self._strength(u) for u in squad) or 1.0
        return squad, centre, combat_foes, near_foes, responders, squad_s

    def _raid_phase_update(self, target, alive: dict[int, dict], enemies: list[dict]) -> None:
        """Hit and run, to annoy. The squad approaches along a flank, hits
        (harvesters first) for a few seconds, and runs the moment anything
        comes at it or it gets hurt. Once they stop chasing it goes straight
        back in, on the other flank. It never fights to the end."""
        view = self._raid_view(alive, enemies)
        if view is None or not target:
            return
        squad, centre, combat_foes, near_foes, responders, squad_s = view
        now = time.monotonic()
        hp = sum(float(u.get("hp", 100) or 0) for u in squad)
        if any(self._approaching(e, centre) and _dist(_xy(e), centre) <= RAID_SAFE_RANGE for e in combat_foes):
            self._raid_last_chased = now
        phase = self._raid_phase
        threatened = responders >= RAID_WITHDRAW_RATIO * squad_s
        if phase == "approach":
            if _dist(centre, target) <= 8:
                phase, self._raid_strike_until, self._raid_hp_start = "strike", now + RAID_STRIKE_S, hp
            elif threatened:
                phase = "withdraw"  # intercepted on the way in
        if phase == "strike":
            hurt = self._raid_hp_start > 0 and hp < self._raid_hp_start * (1 - RAID_HURT_SHARE)
            if threatened or hurt or now >= self._raid_strike_until:
                phase = "withdraw"
        if phase == "withdraw":
            # Run for at least RAID_NOT_CHASED_S, then stop once nobody chases.
            started = self._raid_phase_since if self._raid_phase == "withdraw" else now
            calm_since = max(self._raid_last_chased, started)
            clear = all(_dist(_xy(e), centre) > RAID_SAFE_RANGE for e in combat_foes)
            not_chased = now - calm_since >= RAID_NOT_CHASED_S and not threatened
            if (clear and now - started >= RAID_NOT_CHASED_S) or not_chased or now - started > RAID_WITHDRAW_MAX_S:
                phase = "reset"
        reset_since = self._raid_phase_since if self._raid_phase == "reset" else now
        if phase == "reset" and now - reset_since >= RAID_RESET_S and not threatened:
            # They stopped chasing: back in at once, from the other flank.
            phase = "approach"
            self._raid_side = -(self._raid_side or 1)
        if phase != self._raid_phase:
            self._raid_phase, self._raid_phase_since = phase, now
            self.log(f"[bot] raid: {phase} ({len(squad)} units at {list(centre)}, responders {responders:.0f} vs {squad_s:.0f})")
        eb = getattr(self, "_enemy_base_cache", None)
        self._raid_info = {
            "size": len(squad), "phase": phase, "at": list(centre), "target": list(target),
            "responders": round(responders),
            "responders_from_their_base": round(min(_dist(_xy(e), eb) for e in near_foes)) if near_foes and eb else None,
            "slow_units": sum(1 for u in squad if self._speed(u) < RAID_FAST),
        }

    def _run_raid(self, target, alive: dict[int, dict], enemies: list[dict], enemy_base, staging, raw: dict,
                  controlled: set[int]) -> list[dict]:
        """The squad's orders for its current phase (see _raid_phase_update)."""
        view = self._raid_view(alive, enemies)
        if view is None or not target:
            return []
        squad, centre, combat_foes, near_foes, responders, squad_s = view
        phase = self._raid_phase
        ids = [i for i in sorted(self._raiders) if i not in controlled]
        if not ids:
            return []
        orders: list[dict] = []
        if phase == "approach":
            if not self._recent("raid", 4.0):
                orders += self._raid_orders(ids, squad, target, raw)
                self._mark("raid")
        elif phase == "strike":
            if not self._recent("raid", 2.0):
                orders.append({"cmd": "attack_move", "units": ids, "to": list(target)})
                self._mark("raid")
        else:  # withdraw / reset: get away from the responders, towards home
            if not self._recent("raid_withdraw", 3.0):
                threats = [_xy(e) for e in (near_foes or combat_foes)]
                if threats:
                    dest = self._flee_cell(centre, threats, staging or getattr(self, "_base_cell", None), raw, RAID_WITHDRAW_STEP)
                elif staging and _dist(centre, staging) > 10:
                    dest = self._toward(centre, staging, 12, raw)
                else:
                    dest = None
                if dest:
                    orders.append({"cmd": "move", "units": ids, "to": list(dest)})
                    self._mark("raid_withdraw")
        return orders

    def _raid_route(self, start, target, raw: dict) -> list[tuple[int, int]]:
        """Waypoints for a raid: straight when the way is clear, else a flank
        point RAID_DETOUR cells to the quieter side of the direct line (kept for
        the life of the squad), then the target."""
        d = _dist(start, target)
        if d < 12:
            return [target]
        now = time.monotonic()
        foes = [((e["x"], e["y"]), self._strength(e)) for e in self._enemy_army.values()
                if now - e["t"] < ENEMY_MEMORY_S and str(e.get("type", "")).lower() != "harv"]
        foes += [(_xy(b), self._strength(b)) for b in list(self._enemy_buildings.values())
                 if _xy(b) and str(b.get("type", "")).lower() in DEFENSE_TYPES]

        def danger(cell) -> float:
            return sum(st for c, st in foes if _dist(c, cell) <= RAID_DANGER_RANGE)

        mid = ((start[0] + target[0]) / 2, (start[1] + target[1]) / 2)
        if self._raid_side is None and danger(mid) == 0 and danger(start) == 0:
            return [target]
        px, py = -(target[1] - start[1]) / d, (target[0] - start[0]) / d
        options = {}
        for side in (1, -1):
            wp = self._clamp((int(round(mid[0] + px * RAID_DETOUR * side)), int(round(mid[1] + py * RAID_DETOUR * side))), raw)
            leg = ((wp[0] + target[0]) / 2, (wp[1] + target[1]) / 2)
            options[side] = (danger(wp) + 0.5 * danger(leg), wp)
        best = min(options, key=lambda k: options[k][0])
        if self._raid_side in options and options[self._raid_side][0] <= options[best][0] + 300:
            best = self._raid_side  # keep the flank we chose, unless it got much worse
        self._raid_side = best
        return [options[best][1], target]

    def _raid_orders(self, ids: list[int], squad: list[dict], target, raw: dict) -> list[dict]:
        """Sneak (move, not attack-move) along the route, fight only at the target."""
        centre = self._centroid([_xy(u) for u in squad if _xy(u)]) or target
        route = self._raid_route(centre, target, raw)
        if len(route) == 2 and _dist(centre, route[0]) <= 5:
            route = route[1:]  # the flank point is reached: on to the target
        if len(route) == 1:
            if _dist(centre, target) <= 8:
                return [{"cmd": "attack_move", "units": ids, "to": list(target)}]
            return [{"cmd": "move", "units": ids, "to": list(target)}]
        orders = [{"cmd": "move", "units": ids, "to": list(route[0])}]
        if not self._no_queued:
            orders.append({"cmd": "attack_move", "units": ids, "to": list(target), "queued": True})
        return orders

    def _raid_cell(self, enemy_base) -> tuple[int, int] | None:
        """Where enemy harvesters are: one seen recently, else the ore field by their base."""
        now = time.monotonic()
        recent = [c for c, t in self._enemy_harv_seen.values() if now - t < 45]
        if recent:
            return recent[-1]
        if self._raid_target:
            return self._raid_target
        if self._enemy_ore is None and enemy_base:
            self._enemy_ore = _ore_near(self._config, enemy_base) or enemy_base
        return self._enemy_ore

    def _wave_size(self, raw: dict, army: list[dict] | None = None) -> int:
        n_inf = sum(1 for u in army or [] if str(u.get("type", "")).lower() in ("e1", "e2", "e3", "e4"))
        if self._rush_active and not self._rush_launched:
            late_s, late_n = RUSH_LAUNCH_LATE
            return late_n if self._game_s >= late_s else RUSH_LAUNCH_AT
        has_vehicles = any(str(u.get("type", "")).lower() in TANKS + ARTILLERY + ("4tnk",) for u in army or [])
        if not has_vehicles and self._game_s < 360 and self.knobs.get("posture") != "aggressive" and n_inf < 14:
            return 999  # no small infantry-only waves in the opening (macro)
        base = int(self.knobs.get("attack_size", 8))
        posture = self.knobs.get("posture")
        if posture == "aggressive":
            base = max(4, base - 3)
        elif posture in ("defensive", "economy"):
            base += 2
        size = base + 2 * self._wave
        remaining = (raw.get("time") or {}).get("remaining_seconds")
        if isinstance(remaining, (int, float)) and remaining < 240:
            size = max(4, size // 2)  # last minutes: use the army
        return min(size, 18)

    def _attack_target(self, enemies: list[dict], army: list[dict], enemy_base: tuple[int, int] | None):
        chosen = self.knobs.get("target")
        if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
            return (int(chosen[0]), int(chosen[1]))
        centre = self._centroid([c for c in (_xy(u) for u in army) if c])
        # A small force does more damage killing harvesters than hitting buildings.
        if sum(self._strength(u) for u in army) < 1500:
            cell = self._raid_cell(enemy_base)
            if cell:
                return cell
        known = [b for b in self._enemy_buildings.values() if _xy(b)]
        if centre and known:
            nearest = min(known, key=lambda b: _dist(_xy(b), centre))
            # Among buildings near the nearest one, hit the cheap critical ones
            # first: power (cripples production and defenses), then production.
            cluster = [b for b in known if _dist(_xy(b), _xy(nearest)) <= 10]
            return _xy(min(cluster, key=lambda b: (BUILDING_PRIORITY.get(str(b.get("type", "")).lower(), 9),
                                                    _dist(_xy(b), centre))))
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

    def _remember_enemy_harvesters(self, enemies: list[dict]) -> None:
        now = time.monotonic()
        for e in enemies:
            if str(e.get("type", "")).lower() == "harv" and _xy(e) and isinstance(e.get("id"), int):
                self._enemy_harv_seen[e["id"]] = (_xy(e), now)

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
        acc, total = self._idle_acc, self._idle_acc["total"] or 1.0
        idle = " ".join(f"{k}={100 * acc[k] / total:.0f}%" for k in ("Building", "Infantry", "Vehicle", "cash>1500"))
        self._idle_acc = Counter()
        self.log(f"[bot] idle since last report: {idle}")
        report = getattr(self.advisor, "cost_report", None)
        if callable(report):
            self.log(f"[cost] {report()}")
        self.log(
            f"[bot] t={t.get('game_seconds')}s left={t.get('remaining_seconds')}s cash={econ.get('cash')} "
            f"power={econ.get('power_provided')}/{econ.get('power_drained')} harv={econ.get('harvesters')} "
            f"army={army} buildings={len(obs.get('buildings') or [])} "
            f"kills={mil.get('kills_cost')} deaths={mil.get('deaths_cost')} wave={self._wave} "
            f"attacking={self._attacking} knobs={self.knobs.get('posture')}/{self.knobs.get('attack_size')}"
        )


def _weapon_range(entry: dict | None) -> float | None:
    """Longest weapon range in cells from a rules entry, whatever its shape
    (number of cells, OpenRA "5c512" strings, or world units)."""
    if not isinstance(entry, dict):
        return None
    weapon = entry.get("weapon")
    name = weapon.get("weapon") if isinstance(weapon, dict) else weapon
    if isinstance(name, str) and ANTI_AIR_WEAPONS.search(name):
        return None  # an anti-air main weapon says nothing about ground range
    found: list[float] = []

    def value(v: Any) -> float | None:
        if isinstance(v, (int, float)):
            return float(v) / 1024.0 if v > 100 else float(v)
        if isinstance(v, str):
            m = re.fullmatch(r"\s*(\d+)c(\d+)\s*", v)
            if m:
                return int(m.group(1)) + int(m.group(2)) / 1024.0
            try:
                return float(v)
            except ValueError:
                return None
        return None

    def walk(node: Any, depth: int) -> None:
        if depth > 3:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if "range" in str(k).lower() and "min" not in str(k).lower():
                    r = value(v)
                    if r is not None and 0 < r < 30:
                        found.append(r)
                elif isinstance(v, (dict, list)):
                    walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, depth + 1)

    walk(entry.get("weapon"), 0)
    walk({k: v for k, v in entry.items() if "range" in str(k).lower()}, 0)
    return max(found) if found else None


BUILDING_PRIORITY = {"powr": 0, "apwr": 0, "barr": 1, "tent": 1, "weap": 2, "proc": 3, "fact": 4}


def _opening_for_seat(spec: str, seat: int | None) -> str:
    """``rush`` / ``macro``, or per seat: ``seat0:rush,seat1:macro``."""
    spec = (spec or "").strip().lower()
    if not spec:
        return DEFAULT_OPENING
    if ":" not in spec:
        return spec if spec in ("rush", "macro") else DEFAULT_OPENING
    for part in spec.split(","):
        key, _, value = part.partition(":")
        if key.strip() == f"seat{seat}" and value.strip() in ("rush", "macro"):
            return value.strip()
    return DEFAULT_OPENING


def _ore_near(config: dict | None, cell: tuple[int, int], radius: int = 22) -> tuple[int, int] | None:
    """The centre of the ore/gem cells nearest ``cell`` on the map picture."""
    try:
        maps = config["rules"]["maps"]
    except (TypeError, KeyError):
        return None
    for m in maps.values():
        spawns = [tuple(s) for s in m.get("spawns") or []]
        if not any(_dist(s, cell) < 3 for s in spawns):
            continue
        layout = m.get("layout") or {}
        ox, oy = layout.get("origin") or (0, 0)
        ore = [
            (ox + j, oy + i)
            for i, row in enumerate(layout.get("rows") or [])
            for j, ch in enumerate(row)
            if ch in "$*" and _dist((ox + j, oy + i), cell) <= radius
        ]
        if not ore:
            return None
        # The densest patch: the ore cell with most ore around it.
        best = max(ore, key=lambda c: sum(1 for o in ore if _dist(o, c) <= 4))
        return best
    return None


def dump_dir_from_env() -> str | None:
    return os.environ.get("REDALERT_DUMP_DIR") or "logs"
