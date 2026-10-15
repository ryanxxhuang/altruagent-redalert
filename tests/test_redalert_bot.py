from altruagent import WAIT, DecisionContext, WithReasoning
from altruagent.models import GameState

from agent.agent import Agent
from agent.redalert_bot import RedAlertBot


def _state(obs, legal=None, remaining=1000):
    raw = {
        "phase": "playing",
        "state_version": 100,
        "map": {"id": "sidestep", "width": 100, "height": 100, "bounds": [2, 2, 96, 96]},
        "time": {"max_minutes": 20, "game_seconds": 60, "remaining_seconds": remaining},
        "limits": {"max_orders_per_batch": 20, "max_units_per_order": 50, "max_train_count": 10},
        "observation": obs,
        "legal_actions": legal or {"enemy_base": [80, 80], "repair": [], "production_buildings": []},
    }
    return GameState(
        session_id="s", game_name="red_alert", status="in_progress", observation="", current_player=None,
        legal_actions=[], is_terminal=False, is_current_actor=True, state_version=100, returns=None,
        move_count=0, termination_reason=None, messaging_enabled=False, phase="playing", next_actions=[],
        new_messages=[], terminated_messaging=[], messaging_mode="none", raw=raw,
    )


CTX = DecisionContext(session_id="s", tournament_id=None, game_type="red_alert", agent_id="a")


def _obs(**kw):
    obs = {
        "faction": "russia", "enemy_faction": "england",
        "economy": {"cash": 5000, "ore": 0, "power_provided": 0, "power_drained": 0, "harvesters": 0},
        "military": {"kills_cost": 0, "deaths_cost": 0},
        "units": [], "buildings": [], "production": [], "available_production": [],
        "visible_enemies": [], "visible_enemy_buildings": [], "base_center": [20, 20], "events": [],
    }
    obs.update(kw)
    return obs


def _orders(decision):
    assert isinstance(decision, WithReasoning), decision
    return decision.action["orders"]


def bot():
    return RedAlertBot(log=lambda m: None, dump_dir=None)


def test_deploys_mcv_first():
    orders = _orders(bot().choose_action(_state(_obs(units=[{"id": 7, "type": "mcv", "x": 20, "y": 20, "idle": True}])), CTX))
    assert {"cmd": "deploy", "units": [7]} in orders


def test_builds_power_then_barracks():
    b = bot()
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], available_production=["powr", "barr", "proc"])
    assert {"cmd": "build", "item": "powr"} in _orders(b.choose_action(_state(obs), CTX))
    obs["buildings"].append({"id": 2, "type": "powr", "x": 22, "y": 20})
    obs["economy"].update(power_provided=100, power_drained=0)
    b._sent.clear()
    assert {"cmd": "build", "item": "barr"} in _orders(b.choose_action(_state(obs), CTX))


def test_places_ready_building_and_trains():
    obs = _obs(
        buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}, {"id": 2, "type": "barr", "x": 24, "y": 20}],
        production=[{"queue": "Building", "item": "proc", "progress": 1.0, "ready": True}],
        available_production=["powr", "proc", "e1", "e3"],
    )
    orders = _orders(bot().choose_action(_state(obs), CTX))
    assert {"cmd": "place", "item": "proc"} in orders
    assert any(o["cmd"] == "train" for o in orders)


def test_defends_base():
    army = [{"id": 100 + i, "type": "3tnk", "x": 25, "y": 25, "idle": True} for i in range(3)]
    obs = _obs(
        buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=army,
        visible_enemies=[{"id": 900, "type": "1tnk", "x": 22, "y": 23, "hp": 100}],
    )
    orders = _orders(bot().choose_action(_state(obs), CTX))
    # All three engage the intruder: attack-move to it, or focus fire on it.
    engaged = set()
    for o in orders:
        if (o["cmd"] == "attack_move" and o["to"] == [22, 23]) or (o["cmd"] == "attack" and o["target"] == 900):
            engaged |= set(o["units"])
    assert engaged == {100, 101, 102}


def test_attacks_with_big_army_and_respects_command_cap():
    army = [{"id": 100 + i, "type": "3tnk", "x": 25, "y": 25, "idle": True} for i in range(120)]
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=army)
    orders = _orders(bot().choose_action(_state(obs), CTX))
    attack = [o for o in orders if o["cmd"] == "attack_move"]
    # The group steps towards the enemy base (80, 80) together.
    assert attack and attack[0]["to"][0] > 25 and attack[0]["to"][1] > 25
    assert sum(len(o.get("units") or []) or o.get("count", 1) for o in orders) <= 200
    assert all(len(o["units"]) <= 50 for o in attack)


def test_waits_without_observation_and_survives_garbage():
    assert bot().choose_action(_state(None), CTX) is WAIT
    assert bot().choose_action(_state(_obs(units="garbage", economy=None)), CTX) is WAIT


def test_agent_routes_other_games():
    agent = Agent()
    st = _state(None)
    st.game_name = "werewolf"
    ctx = DecisionContext(session_id="s", tournament_id=None, game_type="werewolf", agent_id="a")
    assert agent.choose_action(st, ctx) is WAIT


class FakeCommanders:
    def __init__(self, knobs=None, commands=None):
        self._knobs, self._commands = knobs or {}, commands or []

    def observe(self, raw, status=None):
        pass

    def knobs(self):
        return dict(self._knobs)

    def pop_commands(self):
        out, self._commands = self._commands, []
        return out


def _army_obs(n, enemies=()):
    army = [{"id": 100 + i, "type": "3tnk", "x": 50, "y": 50, "idle": True, "hp": 100} for i in range(n)]
    return _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=army, visible_enemies=list(enemies))


def test_tactical_attack_now_and_target():
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommanders({"target": (70, 30)}, [{"type": "attack_now"}]))
    orders = _orders(b.choose_action(_state(_army_obs(5)), CTX))
    # Heads for the chosen target (70, 30) from (50, 50): east and north.
    assert any(o["cmd"] == "attack_move" and o["to"][0] > 50 and o["to"][1] < 50 for o in orders)


def test_tactical_retreat_only_when_losing():
    b = RedAlertBot(log=lambda m: None)
    b.choose_action(_state(_army_obs(12)), CTX)
    assert b._attacking
    # Winning (no enemies near): the retreat is ignored.
    b.advisor = FakeCommanders(commands=[{"type": "retreat"}])
    b.choose_action(_state(_army_obs(12)), CTX)
    assert b._attacking
    # Losing to a pile of enemy heavy tanks: the retreat goes through.
    foes = [{"id": 900 + i, "type": "4tnk", "x": 54, "y": 50, "hp": 100} for i in range(15)]
    b.advisor = FakeCommanders(commands=[{"type": "retreat"}])
    b._rush_active = False
    orders = _orders(b.choose_action(_state(_army_obs(12, foes)), CTX))
    assert not b._attacking and any(o["cmd"] == "move" for o in orders)


def test_fast_focus_fire_and_pull_back():
    import time
    enemy = {"id": 900, "type": "2tnk", "x": 55, "y": 50, "hp": 20}
    cmds = [{"type": "focus", "target": 900, "until": time.monotonic() + 5}, {"type": "pull_back", "units": [101]}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommanders(commands=cmds))
    orders = _orders(b.choose_action(_state(_army_obs(3, [enemy])), CTX))
    focus = [o for o in orders if o["cmd"] == "attack"]
    assert focus and focus[-1]["target"] == 900 and 101 not in focus[-1]["units"]
    assert {"cmd": "move", "units": [101], "to": list(b._staging((20, 20), (80, 80), _state(None).raw))} in orders


def test_layer_settings_and_handlers(monkeypatch):
    from agent import advisor
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    for var in ("STRATEGY_MODEL", "TACTICAL_MODEL", "FAST_MODEL", "TACTICAL_EVERY"):
        monkeypatch.delenv(var, raising=False)
    c = advisor.commanders_from_env(log=lambda m: None)
    layers = c._layers
    assert (layers["strategy"].model, layers["strategy"].effort) == ("claude-opus-5-5", "medium")
    assert (layers["tactical"].model, layers["tactical"].effort, layers["tactical"].every) == ("claude-sonnet-5-5", "low", 3.0)
    assert (layers["fast"].model, layers["fast"].effort) == ("claude-haiku-5-5", "low")
    fast = layers["fast"]._request_kwargs()
    assert fast["thinking"] == {"type": "disabled"} and "fallbacks" not in fast  # Haiku: no server fallback
    assert "thinking" not in layers["tactical"]._request_kwargs()
    assert layers["strategy"]._request_kwargs()["fallbacks"] == "default"
    c._on_strategy({"posture": "aggressive", "unit_focus": "tanks", "defenses": 9, "max_refineries": 2,
                    "build_targets": [{"item": "PROC", "count": 2}], "avoid": ["fix"], "plan": "rush"})
    k = c.knobs()
    assert k["posture"] == "aggressive" and k["defenses"] == 6
    assert k["build_targets"] == [("proc", 2)] and k["avoid"] == ["fix"]
    c._on_tactical({"build": "WEAP", "defense": "", "infantry": "e3", "infantry_count": 9, "vehicle": "auto",
                    "vehicle_count": 1, "main_action": "attack", "main_x": 70, "main_y": 40, "raid_size": 3,
                    "raid_x": -1, "raid_y": -1, "note": "go"})
    d = c.decision()
    assert d["build"] == "weap" and d["infantry_count"] == 5 and d["main_action"] == "attack" and d["seq"] == 1
    monkeypatch.setenv("TACTICAL_EVERY", "6")
    assert advisor.commanders_from_env(log=lambda m: None)._layers["tactical"].every == 6.0

def test_rule_micro_focuses_weak_valuable_target_and_pulls_back():
    army = [{"id": 100 + i, "type": "3tnk", "x": 50, "y": 50, "idle": False, "hp": 100} for i in range(3)]
    army.append({"id": 200, "type": "3tnk", "x": 50, "y": 51, "idle": False, "hp": 15})
    enemies = [
        {"id": 900, "type": "e1", "x": 54, "y": 50, "hp": 100},
        {"id": 901, "type": "v2rl", "x": 55, "y": 50, "hp": 40},
    ]
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=army, visible_enemies=enemies)
    orders = _orders(bot().choose_action(_state(obs), CTX))
    focus = [o for o in orders if o["cmd"] == "attack"]
    assert focus and focus[-1]["target"] == 901 and 200 not in focus[-1]["units"]
    assert any(o["cmd"] == "move" and o["units"] == [200] for o in orders)


def test_no_infantry_only_wave_in_opening_for_macro(monkeypatch):
    monkeypatch.setenv("REDALERT_OPENING", "macro")
    army = [{"id": 100 + i, "type": "e1", "x": 25, "y": 25, "idle": True} for i in range(10)]
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=army)
    b = bot()
    b.choose_action(_state(obs), CTX)
    assert not b._attacking


def test_rush_launches_with_eight_infantry_and_delays_war_factory():
    b = bot()
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}, {"id": 2, "type": "powr", "x": 22, "y": 20},
                          {"id": 3, "type": "barr", "x": 24, "y": 20}, {"id": 4, "type": "proc", "x": 26, "y": 20}],
               available_production=["weap", "powr", "e1", "e3"])
    obs["economy"].update(power_provided=100, power_drained=50)
    orders = _orders(b.choose_action(_state(obs), CTX))
    assert {"cmd": "build", "item": "weap"} not in orders  # money goes to the rush first
    obs["units"] = [{"id": 100 + i, "type": "e1", "x": 25, "y": 25, "idle": True} for i in range(8)]
    b.choose_action(_state(obs), CTX)
    assert b._attacking and b._rush_launched


def test_raid_hunts_enemy_ore_field(monkeypatch):
    monkeypatch.setenv("REDALERT_OPENING", "macro")
    b = bot()
    army = [{"id": 100 + i, "type": "3tnk", "x": 27, "y": 27, "idle": True} for i in range(7)]
    st = _state(_obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=army))
    ctx = DecisionContext(session_id="s", tournament_id=None, game_type="red_alert", agent_id="a", game_config={
        "rules": {"maps": {"sidestep": {"spawns": [[20, 20], [80, 80]], "layout": {
            "origin": [0, 0], "rows": ["." * 100] * 70 + ["." * 70 + "$$$$" + "." * 26] * 4 + ["." * 100] * 26}}}}})
    b.knobs["attack_size"] = 20
    orders = _orders(b.choose_action(st, ctx))
    raid = [o for o in orders if o["cmd"] in ("move", "attack_move") and len(o["units"]) == 4]
    assert raid and 70 <= raid[0]["to"][0] <= 73 and 70 <= raid[0]["to"][1] <= 73


def _kite_state(enemy_x_seq):
    """A light tank (fast) chased by a heavy tank (slow); returns orders per step."""
    b = bot()
    b._rush_active = False
    out = []
    for ex in enemy_x_seq:
        army = [{"id": 100, "type": "1tnk", "x": 50, "y": 50, "idle": False, "hp": 100}]
        foe = [{"id": 900, "type": "3tnk", "x": ex, "y": 50, "hp": 100}]
        obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 50}], units=army, visible_enemies=foe,
                   base_center=[20, 50])
        d = b.choose_action(_state(obs), CTX)
        out.append(d.action["orders"] if isinstance(d, WithReasoning) else [])
    return out


def test_kiting_runs_from_a_chasing_enemy():
    steps = _kite_state([57, 55])  # heavy tank closing in from the east
    run = [o for o in steps[-1] if o["cmd"] == "move" and 100 in o["units"]]
    assert run and run[0]["to"][0] < 50  # runs west, away from it


def test_kiting_turns_and_shoots_when_the_enemy_turns_back():
    b_steps = _kite_state([55, 58])  # it was close, now it is moving away
    shoot = [o for o in b_steps[-1] if o["cmd"] == "attack" and 100 in o["units"]]
    assert shoot and shoot[0]["target"] == 900


def test_no_kiting_when_the_enemy_is_faster_and_outranges():
    from agent.redalert_bot import RedAlertBot as B
    b = B(log=lambda m: None)
    assert not b._can_kite({"type": "e1"}, {"type": "1tnk"})
    assert b._can_kite({"type": "1tnk"}, {"type": "3tnk"})
    assert b._can_kite({"type": "v2rl"}, {"type": "3tnk"})


def test_weapon_range_parsing():
    from agent.redalert_bot import _weapon_range
    assert _weapon_range({"weapon": {"name": "105mm", "range": "4c768"}}) == 4.75
    assert _weapon_range({"weapon": [{"range": 5}, {"range": 7.5}]}) == 7.5
    assert _weapon_range({"weapon": "105mm"}) is None


def test_kiting_trades_shots_with_a_stationary_enemy():
    steps = _kite_state([54, 54])  # in range, not moving: shoot it (once; it keeps shooting)
    assert any(o["cmd"] == "attack" and o["target"] == 900 for o in steps[0])
    assert not any(o["cmd"] == "move" and 100 in o["units"] for step in steps for o in step)


def test_anti_air_main_weapon_falls_back_to_ground_range():
    from agent.redalert_bot import _weapon_range
    assert _weapon_range({"weapon": {"weapon": "RedEye", "range_cells": 7.5}}) is None
    assert _weapon_range({"weapon": {"weapon": "FLAK-23-AA", "range_cells": 8.0}}) is None
    assert _weapon_range({"weapon": {"weapon": "105mm", "range_cells": 4.75}}) == 4.75
    b = bot()
    b.rules = {"e3": {"weapon": {"weapon": "RedEye", "range_cells": 7.5}, "speed": 1.32},
               "3tnk": {"weapon": {"weapon": "105mm", "range_cells": 4.75}, "speed": 1.56}}
    assert b._range({"type": "e3"}) == 5.0
    assert not b._can_kite({"type": "e3"}, {"type": "3tnk"})


def test_many_beat_few_attacks_after_wiping_their_army(monkeypatch):
    monkeypatch.setenv("REDALERT_OPENING", "macro")
    b = bot()
    mine = [{"id": 100 + i, "type": "e1", "x": 50, "y": 50, "idle": True, "hp": 100} for i in range(3)]
    foes = [{"id": 900 + i, "type": "e1", "x": 52, "y": 50, "hp": 100} for i in range(4)]  # 400 vs our 300
    b.choose_action(_state(_obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=mine,
                                visible_enemies=foes)), CTX)
    assert not b._attacking
    # They died where we could see: their known army is gone, our 3 riflemen go.
    b.choose_action(_state(_obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=mine)), CTX)
    assert b._attacking


def test_no_full_wave_into_a_stronger_known_army(monkeypatch):
    monkeypatch.setenv("REDALERT_OPENING", "macro")
    b = bot()
    b.knobs["attack_size"] = 6
    mine = [{"id": 100 + i, "type": "3tnk", "x": 30, "y": 30, "idle": True, "hp": 100} for i in range(8)]
    seen = [{"id": 900 + i, "type": "4tnk", "x": 70, "y": 70, "hp": 100} for i in range(10)]
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=mine, visible_enemies=seen)
    b.choose_action(_state(obs), CTX)
    assert not b._attacking


def test_group_waits_for_stragglers_and_holds_when_outnumbered_ahead(monkeypatch):
    monkeypatch.setenv("REDALERT_OPENING", "macro")
    b = bot()
    b._attacking = True
    b._wave_peak = 10
    front = [{"id": 100 + i, "type": "3tnk", "x": 50, "y": 50, "idle": True, "hp": 100} for i in range(6)]
    back = [{"id": 200 + i, "type": "3tnk", "x": 39, "y": 39, "idle": False, "hp": 100} for i in range(4)]
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=front + back)
    orders = _orders(b.choose_action(_state(obs), CTX))
    moves = [(tuple(o["to"]), set(o["units"])) for o in orders if o["cmd"] == "attack_move"]
    # Stragglers head to the group; nobody pushes past it towards (80, 80).
    assert any({200, 201, 202, 203} <= ids for _, ids in moves)
    assert all(to[0] <= 51 and to[1] <= 51 for to, _ in moves)
    # Grouped up, a stronger enemy force sits ahead but out of contact: hold.
    b2 = bot()
    b2._attacking, b2._wave_peak = True, 6
    foes = [{"id": 900 + i, "type": "4tnk", "x": 62, "y": 62, "hp": 100} for i in range(8)]
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=front, visible_enemies=foes)
    orders = _orders(b2.choose_action(_state(obs), CTX))
    assert all(o["to"] == [50, 50] for o in orders if o["cmd"] == "attack_move")
    # Already in contact with it: never stand still at our own centre.
    foes = [{"id": 900 + i, "type": "4tnk", "x": 56, "y": 56, "hp": 100} for i in range(8)]
    b2._sent.clear()
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=front, visible_enemies=foes)
    orders = _orders(b2.choose_action(_state(obs), CTX))
    assert not any(o["cmd"] == "attack_move" and o["to"] == [50, 50] for o in orders)


def test_strategy_build_targets_and_avoid_are_obeyed():
    b = bot()
    b.knobs.update(build_targets=[("proc", 2), ("harv", 3)], avoid=["fix"])
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}, {"id": 2, "type": "powr", "x": 22, "y": 20},
                          {"id": 3, "type": "barr", "x": 24, "y": 20}, {"id": 4, "type": "proc", "x": 26, "y": 20},
                          {"id": 5, "type": "weap", "x": 28, "y": 20}],
               units=[{"id": 50, "type": "harv", "x": 30, "y": 30}],
               available_production=["powr", "proc", "fix", "harv", "3tnk", "e1"])
    obs["economy"].update(power_provided=200, power_drained=100)
    orders = _orders(b.choose_action(_state(obs), CTX))
    assert {"cmd": "build", "item": "proc"} in orders
    assert any(o["cmd"] == "train" and o["item"] == "harv" for o in orders)
    assert {"cmd": "build", "item": "fix"} not in orders


class FakeCommander(FakeCommanders):
    def __init__(self, decision):
        super().__init__()
        import time as _t
        self._d = {"seq": 1, "t": _t.monotonic(), "build": "auto", "defense": "auto", "infantry": "auto",
                   "infantry_count": 2, "vehicle": "auto", "vehicle_count": 1, "main_action": "hold",
                   "main_x": -1, "main_y": -1, "raid_size": 0, "raid_unit": "", "raid_x": -1, "raid_y": -1, **decision}

    def decision(self):
        return dict(self._d)


def _base_obs(units=(), **kw):
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}, {"id": 2, "type": "powr", "x": 22, "y": 20},
                          {"id": 3, "type": "barr", "x": 24, "y": 20}, {"id": 4, "type": "proc", "x": 26, "y": 20},
                          {"id": 5, "type": "weap", "x": 28, "y": 20}],
               units=[{"id": 50, "type": "harv", "x": 30, "y": 30}, *units],
               available_production=["powr", "proc", "fix", "dome", "ftur", "harv", "3tnk", "e1", "e3"], **kw)
    obs["economy"].update(power_provided=200, power_drained=100)
    return obs


def test_commander_production_is_obeyed_once_per_decision():
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander(
        {"build": "dome", "defense": "ftur", "infantry": "e3", "infantry_count": 3, "vehicle": "3tnk", "vehicle_count": 2}))
    orders = _orders(b.choose_action(_state(_base_obs()), CTX))
    assert {"cmd": "build", "item": "dome"} in orders
    assert {"cmd": "build", "item": "ftur"} in orders
    assert {"cmd": "train", "item": "e3", "count": 3} in orders
    assert {"cmd": "train", "item": "3tnk", "count": 2} in orders
    b._sent.clear()
    again = b.choose_action(_state(_base_obs()), CTX)
    again = again.action["orders"] if isinstance(again, WithReasoning) else []
    assert {"cmd": "build", "item": "dome"} not in again  # one building per decision, not one per tick


def test_commander_unknown_item_is_reported_back():
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"build": "stek"}))
    b.choose_action(_state(_base_obs()), CTX)
    assert any("stek" in p for p in b.status()["problems"])


def test_commander_attack_retreat_and_raid():
    army = [{"id": 100 + i, "type": "3tnk", "x": 30, "y": 30, "idle": True, "hp": 100} for i in range(6)]
    army += [{"id": 150 + i, "type": "jeep", "x": 30, "y": 30, "idle": True, "hp": 100} for i in range(2)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 70, "main_y": 70,
                                                                "raid_size": 2, "raid_x": 60, "raid_y": 20}))
    orders = _orders(b.choose_action(_state(_base_obs(units=army)), CTX))
    assert b._attacking and len(b._raiders) == 2
    assert any(o["cmd"] in ("move", "attack_move") and o["to"] == [60, 20] and set(o["units"]) == b._raiders for o in orders)
    assert any(o["cmd"] == "attack_move" and o["to"][0] > 30 and not set(o["units"]) & b._raiders for o in orders)
    b.advisor = FakeCommander({"main_action": "retreat", "raid_size": 0})
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=army)), CTX))
    assert not b._attacking and any(o["cmd"] == "move" for o in orders)


def test_stale_commander_falls_back_to_rules():
    import time as _t
    fake = FakeCommander({"build": "dome"})
    fake._d["t"] = _t.monotonic() - 60
    b = RedAlertBot(log=lambda m: None, advisor=fake)
    orders = _orders(b.choose_action(_state(_base_obs()), CTX))
    assert {"cmd": "build", "item": "dome"} not in orders  # the rules' own build order instead
    assert any(o["cmd"] == "build" for o in orders)


def test_avoid_never_blocks_a_prerequisite_of_a_wanted_unit():
    b = bot()
    b.rules = {"3tnk": {"prerequisites": ["fix"], "queue": "Vehicle"}, "fix": {"queue": "Building"}}
    b.knobs.update(build_targets=[("3tnk", 8)], avoid=["fix", "dome"])
    assert b._avoid() == {"dome"}


def test_per_seat_models_and_cost_report(monkeypatch):
    from types import SimpleNamespace
    from agent import advisor
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("TACTICAL_MODEL", "seat0:claude-sonnet-5-5,seat1:claude-haiku-5-5")
    monkeypatch.setenv("STRATEGY_EVERY", "seat0:15,seat1:30")
    a = advisor.commanders_from_env(log=lambda m: None, seat=0)
    b = advisor.commanders_from_env(log=lambda m: None, seat=1)
    assert a._layers["tactical"].model == "claude-sonnet-5-5" and a._layers["strategy"].every == 15.0
    assert b._layers["tactical"].model == "claude-haiku-5-5" and b._layers["strategy"].every == 30.0
    layer = b._layers["tactical"]
    layer._count(SimpleNamespace(usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=200_000)), 2.0)
    assert abs(layer.cost() - 0.20) < 1e-9  # Haiku: $0.10/M in + $0.50/M out
    assert "$0.20" in b.cost_report()


def test_commander_repeat_before_seeing_the_start_is_not_a_second_copy():
    import time as _t
    fake = FakeCommander({"build": "dome", "seq": 1})
    b = RedAlertBot(log=lambda m: None, advisor=fake)
    assert {"cmd": "build", "item": "dome"} in _orders(b.choose_action(_state(_base_obs()), CTX))
    # Next decision, asked before the dome started, says "dome" again: ignored.
    fake._d.update(seq=2, asked_at=_t.monotonic() - 2, t=_t.monotonic())
    b._sent.clear()
    d = b.choose_action(_state(_base_obs()), CTX)
    assert {"cmd": "build", "item": "dome"} not in (d.action["orders"] if isinstance(d, WithReasoning) else [])
    # A later decision that saw it start and still wants one: a second dome.
    fake._d.update(seq=3, asked_at=_t.monotonic() + 5, t=_t.monotonic())
    b._sent.clear()
    assert {"cmd": "build", "item": "dome"} in _orders(b.choose_action(_state(_base_obs()), CTX))


def test_general_targets_fill_the_gap_between_commander_orders():
    fake = FakeCommander({"build": "dome", "seq": 1})
    b = RedAlertBot(log=lambda m: None, advisor=fake)
    b.knobs["build_targets"] = [("proc", 2)]
    b.choose_action(_state(_base_obs()), CTX)  # dome done for decision 1
    b._sent.clear()
    assert {"cmd": "build", "item": "proc"} in _orders(b.choose_action(_state(_base_obs()), CTX))


def test_holding_is_reported_not_overridden():
    army = [{"id": 100 + i, "type": "3tnk", "x": 26, "y": 26, "idle": True, "hp": 100} for i in range(8)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "hold"}))
    b.choose_action(_state(_base_obs(units=army)), CTX)
    b._holding_since -= 41
    b.choose_action(_state(_base_obs(units=army)), CTX)
    assert not b._attacking and b.status()["main_idle_seconds"] >= 41  # the LLM decides, with the facts


def test_commander_attack_advances_into_a_stronger_force_and_reports_it():
    front = [{"id": 100 + i, "type": "3tnk", "x": 50, "y": 50, "idle": True, "hp": 100} for i in range(6)]
    foes = [{"id": 900 + i, "type": "4tnk", "x": 57, "y": 57, "hp": 100} for i in range(5)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 80}))
    orders = _orders(b.choose_action(_state(_base_obs(units=front, visible_enemies=foes)), CTX))
    assert any(o["cmd"] == "attack_move" and o["to"][0] > 50 for o in orders)  # obeys
    assert b.status()["strength_ahead"] > 0  # and tells the commander what is there


def test_build_named_while_in_production_is_that_one_not_another():
    import time as _t
    fake = FakeCommander({"build": "proc", "seq": 1})
    b = RedAlertBot(log=lambda m: None, advisor=fake)
    obs = _base_obs(production=[{"queue": "Building", "item": "proc", "progress": 0.5, "ready": False}])
    b.choose_action(_state(obs), CTX)  # proc seen in production
    # The refinery finishes; the commander's latest answer (asked while it was
    # still building) still says "proc": not a new one.
    fake._d.update(seq=2, asked_at=_t.monotonic(), t=_t.monotonic())  # asked just after that tick
    b._sent.clear()
    d = b.choose_action(_state(_base_obs()), CTX)
    assert {"cmd": "build", "item": "proc"} not in (d.action["orders"] if isinstance(d, WithReasoning) else [])
    # An answer asked after it left the queue and still asking for proc: a second one.
    fake._d.update(seq=3, asked_at=_t.monotonic() + 3, t=_t.monotonic())
    b._sent.clear()
    assert {"cmd": "build", "item": "proc"} in _orders(b.choose_action(_state(_base_obs()), CTX))


def test_army_inside_enemy_base_wrecks_buildings_in_priority_order():
    army = [{"id": 100 + i, "type": "3tnk", "x": 70, "y": 70, "idle": True, "hp": 100} for i in range(5)]
    theirs = [{"id": 900, "type": "fact", "x": 72, "y": 70, "hp": 100}, {"id": 901, "type": "proc", "x": 74, "y": 72, "hp": 60},
              {"id": 902, "type": "powr", "x": 68, "y": 73, "hp": 100}, {"id": 903, "type": "weap", "x": 75, "y": 68, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 72, "main_y": 70}))
    obs = _base_obs(units=army, visible_enemy_buildings=theirs)
    orders = _orders(b.choose_action(_state(obs), CTX))
    focus = [o for o in orders if o["cmd"] == "attack"]
    assert focus and focus[-1]["target"] == 902  # the power plant before anything else
    theirs.append({"id": 904, "type": "tsla", "x": 71, "y": 69, "hp": 100})
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=army, visible_enemy_buildings=theirs)), CTX))
    assert [o for o in orders if o["cmd"] == "attack"][-1]["target"] == 904  # a defense that shoots back first


def test_commanded_drainer_waits_for_power_when_it_would_go_negative():
    fake = FakeCommander({"build": "weap"})
    b = RedAlertBot(log=lambda m: None, advisor=fake)
    obs = _base_obs()
    obs["available_production"] = ["powr", "proc", "weap", "e1"]
    obs["economy"].update(power_provided=100, power_drained=80)  # weap (-30) would make it -10
    orders = _orders(b.choose_action(_state(obs), CTX))
    assert {"cmd": "build", "item": "powr"} in orders and {"cmd": "build", "item": "weap"} not in orders
    assert any("power" in p for p in b.status()["problems"])
    # With the power plant up, the same order goes through.
    obs["economy"].update(power_provided=200)
    b._sent.clear()
    assert {"cmd": "build", "item": "weap"} in _orders(b.choose_action(_state(obs), CTX))


def test_attack_that_cleared_its_target_carries_on_to_known_buildings():
    army = [{"id": 100 + i, "type": "3tnk", "x": 60, "y": 60, "idle": True, "hp": 100} for i in range(6)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 60, "main_y": 60}))
    b._enemy_buildings = {950: {"id": 950, "type": "powr", "x": 78, "y": 78}}
    orders = _orders(b.choose_action(_state(_base_obs(units=army)), CTX))
    moves = [o for o in orders if o["cmd"] == "attack_move"]
    assert moves and moves[-1]["to"][0] > 62 and "carrying on" in b.status()["main_state"]


def test_strategy_hint_per_seat_reaches_general_and_commander(monkeypatch):
    from agent import advisor
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("STRATEGY_HINT", "seat0:rush with infantry,seat1:macro into tanks")
    a = advisor.commanders_from_env(log=lambda m: None, seat=0)
    b = advisor.commanders_from_env(log=lambda m: None, seat=1)
    assert "rush with infantry" in a._layers["strategy"].system and "rush with infantry" in a._layers["tactical"].system
    assert "macro into tanks" in b._layers["strategy"].system and "macro" not in a._layers["strategy"].system.split("Operator")[1]
    assert "Operator instruction" not in b._layers["fast"].system


def test_enemy_units_are_fought_before_buildings_by_everyone_in_reach():
    army = [{"id": 100 + i, "type": "3tnk", "x": 60 + 3 * i, "y": 60, "idle": True, "hp": 100} for i in range(5)]
    army.append({"id": 200, "type": "3tnk", "x": 90, "y": 60, "idle": True, "hp": 100})  # far from the fight
    foes = [{"id": 900, "type": "e1", "x": 58, "y": 60, "hp": 100}, {"id": 901, "type": "e3", "x": 73, "y": 61, "hp": 50}]
    theirs = [{"id": 950, "type": "powr", "x": 66, "y": 62, "hp": 100}, {"id": 951, "type": "weap", "x": 92, "y": 61, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 66, "main_y": 62}))
    obs = _base_obs(units=army, visible_enemies=foes, visible_enemy_buildings=theirs)
    orders = _orders(b.choose_action(_state(obs), CTX))
    attacks = {}
    for o in orders:
        if o["cmd"] == "attack":
            for i in o["units"]:
                attacks[i] = o["target"]
    assert all(attacks.get(100 + i) in (900, 901) for i in range(5))  # units first, never the power plant
    assert attacks.get(200) == 951  # nobody near it: it wrecks the war factory
    # In contact, the group does not stand still to regroup.
    assert not any(o["cmd"] == "attack_move" and o["to"] == [66, 60] for o in orders)


def test_scout_is_the_cheapest_unit_and_raid_unit_is_honoured():
    units = [{"id": 100, "type": "1tnk", "x": 26, "y": 26, "idle": True, "hp": 100},
             {"id": 101, "type": "e1", "x": 26, "y": 27, "idle": True, "hp": 100},
             {"id": 102, "type": "ftrk", "x": 27, "y": 26, "idle": True, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 1}))
    b.rules = {"1tnk": {"cost": 700, "speed": 2.76}, "e1": {"cost": 100, "speed": 1.32}, "ftrk": {"cost": 600, "speed": 2.76}}
    b.choose_action(_state(_base_obs(units=units)), CTX)
    assert b._raiders == {101}  # the rifleman scouts, not the tank
    b2 = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 1, "raid_unit": "ftrk"}))
    b2.rules = b.rules
    b2.choose_action(_state(_base_obs(units=units)), CTX)
    assert b2._raiders == {102}


def test_outgunned_raid_withdraws_instead_of_fighting():
    units = [{"id": 100 + i, "type": "jeep", "x": 60, "y": 60, "idle": True, "hp": 100} for i in range(3)]
    foes = [{"id": 900 + i, "type": "3tnk", "x": 62, "y": 60, "hp": 100} for i in range(4)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 2, "raid_x": 80, "raid_y": 60}))
    b.choose_action(_state(_base_obs(units=units)), CTX)   # squad formed
    assert len(b._raiders) == 2
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=units, visible_enemies=foes)), CTX))
    assert b._raid_phase == "withdraw" and len(b._raiders) == 2  # runs, stays a squad
    assert any(o["cmd"] == "move" and set(o["units"]) == b._raiders and o["to"][0] < 60 for o in orders)
def test_reinforcements_at_home_do_not_hold_the_front_group_back():
    front = [{"id": 100 + i, "type": "3tnk", "x": 60, "y": 60, "idle": True, "hp": 100} for i in range(4)]
    home = [{"id": 200 + i, "type": "3tnk", "x": 27, "y": 27, "idle": True, "hp": 100} for i in range(6)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 80}))
    orders = _orders(b.choose_action(_state(_base_obs(units=front + home)), CTX))
    front_moves = [o for o in orders if o["cmd"] == "attack_move" and {100, 101, 102, 103} <= set(o["units"])]
    assert front_moves and front_moves[0]["to"][0] > 60  # the front advances; it does not wait at (60, 60)


def test_raid_sneaks_around_the_enemy_army():
    import time as _t
    units = [{"id": 100 + i, "type": "ftrk", "x": 30, "y": 50, "idle": True, "hp": 100} for i in range(3)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_x": 80, "raid_y": 50}))
    b.rules = {"ftrk": {"cost": 600, "speed": 2.76}, "3tnk": {"cost": 1150}}
    # Their army sits on the direct line, a little to the north (y < 50).
    b._enemy_army = {900 + i: {"type": "3tnk", "hp": 100, "x": 55, "y": 45, "t": _t.monotonic()} for i in range(5)}
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 50}], units=units, base_center=[20, 50])
    orders = _orders(b.choose_action(_state(obs), CTX))
    moves = [o for o in orders if o["cmd"] == "move" and set(o["units"]) == {100, 101, 102}]
    assert moves and moves[0]["to"][1] > 60  # flanks to the south, away from their tanks
    assert any(o["cmd"] == "attack_move" and o["to"] == [80, 50] and o.get("queued") for o in orders)
    # With nothing in the way the raid goes straight.
    b2 = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_x": 80, "raid_y": 50}))
    b2.rules = b.rules
    orders = _orders(b2.choose_action(_state(obs), CTX))
    assert any(o["cmd"] == "move" and o["to"] == [80, 50] for o in orders)


def test_a_blob_at_home_marches_out_together_when_told_to_attack():
    # 40 riflemen spread around the staging point, nobody "out in the field" yet.
    army = [{"id": 100 + i, "type": "e1", "x": 22 + i % 8, "y": 22 + i // 8, "idle": True, "hp": 100} for i in range(40)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 80}))
    orders = _orders(b.choose_action(_state(_base_obs(units=army)), CTX))
    moves = [o for o in orders if o["cmd"] == "attack_move"]
    forward = {i for o in moves if o["to"][0] >= 30 and o["to"][1] >= 30 for i in o["units"]}
    assert len(forward) >= 20  # the front half heads for the enemy
    assert not any(o["to"][0] < 24 for o in moves)  # nobody is sent backwards into the base


def test_stragglers_near_the_enemy_base_do_not_trigger_an_emergency_retreat():
    home = [{"id": 100 + i, "type": "2tnk", "x": 26 + i % 6, "y": 26 + i // 6, "idle": True, "hp": 100} for i in range(30)]
    straggler = [{"id": 300, "type": "e3", "x": 75, "y": 75, "idle": False, "hp": 60}]
    foes = [{"id": 900 + i, "type": "2tnk", "x": 78, "y": 76, "hp": 100} for i in range(4)]
    theirs = [{"id": 950, "type": "pbox", "x": 77, "y": 74, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 80}))
    b.rules = {"2tnk": {"cost": 850}, "e3": {"cost": 300}, "pbox": {"cost": 600}}
    b.choose_action(_state(_base_obs(units=home + straggler, visible_enemies=foes, visible_enemy_buildings=theirs)), CTX)
    assert b._attacking and "emergency" not in b.status()["main_state"]
    # The main body itself in a hopeless fight: retreat.
    engaged = [{"id": 100 + i, "type": "e1", "x": 74 + i % 3, "y": 74, "idle": False, "hp": 100} for i in range(3)]
    b2 = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 80}))
    b2.rules = {"e1": {"cost": 100}, "2tnk": {"cost": 850}, "pbox": {"cost": 600}}
    b2.choose_action(_state(_base_obs(units=engaged, visible_enemies=foes, visible_enemy_buildings=theirs)), CTX)
    assert not b2._attacking and "emergency" in b2.status()["main_state"]


def test_power_plant_is_refused_while_power_is_plentiful():
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"build": "powr"}))
    obs = _base_obs()
    obs["economy"].update(power_provided=400, power_drained=160)
    d = b.choose_action(_state(obs), CTX)
    orders = d.action["orders"] if isinstance(d, WithReasoning) else []
    assert {"cmd": "build", "item": "powr"} not in orders
    assert any("power margin" in p for p in b.status()["problems"])
    obs["economy"].update(power_provided=200, power_drained=180)  # margin 20: a plant is due
    b._sent.clear()
    assert {"cmd": "build", "item": "powr"} in _orders(b.choose_action(_state(obs), CTX))


def test_general_standing_raid_tops_up_the_commander():
    army = [{"id": 100 + i, "type": "e1", "x": 26, "y": 26, "idle": True, "hp": 100} for i in range(7)]
    army += [{"id": 150 + i, "type": "jeep", "x": 26, "y": 26, "idle": True, "hp": 100} for i in range(3)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 0}))
    b.knobs["standing_raid"] = 3
    b.choose_action(_state(_base_obs(units=army)), CTX)
    assert len(b._raiders) == 3
    # Enemies at our base: the commander's 0 stands, everyone defends.
    foes = [{"id": 900, "type": "3tnk", "x": 24, "y": 24, "hp": 100}]
    b2 = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 0}))
    b2.knobs["standing_raid"] = 3
    b2.choose_action(_state(_base_obs(units=army, visible_enemies=foes)), CTX)
    assert not b2._raiders


def test_strategy_handler_reads_standing_raid(monkeypatch):
    from agent import advisor
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    c = advisor.commanders_from_env(log=lambda m: None)
    c._on_strategy({"posture": "balanced", "unit_focus": "mixed", "defenses": 1, "max_refineries": 2,
                    "build_targets": [], "avoid": [], "standing_raid": 9, "plan": "x"})
    assert c.knobs()["standing_raid"] == 4


def test_carrying_on_past_an_empty_target_sticks_despite_a_stray_enemy():
    army = [{"id": 100 + i, "type": "3tnk", "x": 60, "y": 60, "idle": True, "hp": 100} for i in range(6)]
    stray = [{"id": 900, "type": "e1", "x": 66, "y": 60, "hp": 100}]  # one rifleman: not "something there"
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 60, "main_y": 60}))
    b._enemy_buildings = {950: {"id": 950, "type": "powr", "x": 78, "y": 78}}
    b.choose_action(_state(_base_obs(units=army, visible_enemies=stray)), CTX)
    assert b._onward == (78, 78) and "carrying on" in b.status()["main_state"]
    # The stray is shot (micro), then the army keeps heading for the power plant.
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=army)), CTX))
    moves = [o for o in orders if o["cmd"] == "attack_move" and 100 in o["units"]]
    assert moves and moves[-1]["to"][0] > 62
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=army, visible_enemies=stray)), CTX)
    assert b._onward == (78, 78)  # a stray rifleman does not reset it


def test_situation_reports_the_largest_enemy_group_seen():
    import time as _t
    b = bot()
    b.rules = {"3tnk": {"cost": 1150}, "e1": {"cost": 100}}
    now = _t.monotonic()
    b._enemy_army = {900 + i: {"type": "3tnk", "hp": 100, "x": 70, "y": 70, "t": now} for i in range(3)}
    b._enemy_army[950] = {"type": "e1", "hp": 100, "x": 20, "y": 20, "t": now}
    b._enemy_army_seen_at = now
    strength, cell, _ = b.enemy_largest_group()
    assert strength == 3450 and cell == (70, 70)
    assert "largest enemy group seen: 3450 at [70, 70]" in b._situation()


def test_raid_hit_and_run_cycle():
    import time as _t
    from agent import redalert_bot as rb
    squad = [{"id": 100 + i, "type": "ftrk", "x": 72, "y": 72, "idle": True, "hp": 100} for i in range(3)]
    harv = [{"id": 800, "type": "harv", "x": 74, "y": 73, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_x": 74, "raid_y": 73}))
    b.rules = {"ftrk": {"cost": 600, "speed": 2.76}, "3tnk": {"cost": 1150}, "harv": {"cost": 1100}}
    b.choose_action(_state(_base_obs(units=squad, visible_enemies=harv)), CTX)
    assert b._raiders == {100, 101, 102} and b._raid_phase == "strike"  # at the target: strike
    # Their tanks respond: withdraw, away from them and towards home.
    tanks = [{"id": 900 + i, "type": "3tnk", "x": 78, "y": 72, "hp": 100} for i in range(2)]
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=squad, visible_enemies=harv + tanks)), CTX))
    assert b._raid_phase == "withdraw"
    run = [o for o in orders if o["cmd"] == "move" and set(o["units"]) == {100, 101, 102}]
    assert run and run[0]["to"][0] < 72
    # Clear of them (after running a moment): reset, then approach again on the other flank.
    side = b._raid_side
    b._raid_phase_since -= 3
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=squad)), CTX)
    assert b._raid_phase == "reset"
    b._raid_phase_since -= rb.RAID_RESET_S + 1
    b._sent.clear()
    b._enemy_army = {}  # nothing remembered on either flank: the other flank is taken
    b.choose_action(_state(_base_obs(units=[{**u, "x": 50, "y": 50} for u in squad])), CTX)
    assert b._raid_phase == "approach" and b._raid_side == -(side or 1)
    assert b.status()["raid"]["phase"] == "approach"


def test_raiders_shoot_harvesters_not_tanks():
    squad = [{"id": 100 + i, "type": "ftrk", "x": 72, "y": 72, "idle": True, "hp": 100} for i in range(3)]
    foes = [{"id": 800, "type": "harv", "x": 75, "y": 72, "hp": 100}, {"id": 900, "type": "3tnk", "x": 74, "y": 74, "hp": 40}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_x": 75, "raid_y": 72}))
    b.rules = {"ftrk": {"cost": 600, "speed": 2.76}, "3tnk": {"cost": 1150}, "harv": {"cost": 1100}}
    b._raiders = {100, 101, 102}
    orders = _orders(b.choose_action(_state(_base_obs(units=squad, visible_enemies=foes)), CTX))
    shots = [o for o in orders if o["cmd"] == "attack" and set(o["units"]) & {100, 101, 102}]
    assert shots and all(o["target"] == 800 for o in shots)


def test_raid_runs_when_hurt_and_goes_back_once_they_stop_chasing():
    import time as _t
    squad = [{"id": 100 + i, "type": "jeep", "x": 72, "y": 72, "idle": True, "hp": 100} for i in range(3)]
    harv = [{"id": 800, "type": "harv", "x": 74, "y": 73, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_x": 74, "raid_y": 73}))
    b.rules = {"jeep": {"cost": 500, "speed": 4.0}, "harv": {"cost": 1100}, "e1": {"cost": 100, "speed": 1.32}}
    b.choose_action(_state(_base_obs(units=squad, visible_enemies=harv)), CTX)
    assert b._raid_phase == "strike"
    # Took a few hits (no responder seen yet): run anyway.
    hurt = [{**u, "hp": 80} for u in squad]
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=hurt, visible_enemies=harv)), CTX)
    assert b._raid_phase == "withdraw"
    # Nobody is coming after it: after a moment it resets, then goes straight back in.
    b._raid_last_chased = _t.monotonic() - 5
    b._raid_phase_since -= 3  # ran for 3 s
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=[{**u, "x": 60} for u in hurt])), CTX)
    assert b._raid_phase == "reset"
    b._raid_phase_since -= 4
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=[{**u, "x": 60} for u in hurt])), CTX)
    assert b._raid_phase == "approach"


def test_infantry_is_not_used_for_a_raid_when_fast_units_exist():
    units = [{"id": 100 + i, "type": "e1", "x": 26, "y": 26, "idle": True, "hp": 100} for i in range(5)]
    units += [{"id": 200 + i, "type": "jeep", "x": 27, "y": 26, "idle": True, "hp": 100} for i in range(3)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_unit": "e1"}))
    b.rules = {"jeep": {"cost": 500, "speed": 4.0}, "e1": {"cost": 100, "speed": 1.32}}
    b.choose_action(_state(_base_obs(units=units)), CTX)
    assert b._raiders == {200, 201, 202}
    assert any("too slow" in p for p in b.status()["problems"])


def test_raid_squad_takes_moving_units_not_only_idle_ones():
    units = [{"id": 100 + i, "type": "e3", "x": 40, "y": 40, "idle": False, "hp": 100} for i in range(6)]
    units.append({"id": 200, "type": "jeep", "x": 30, "y": 30, "idle": False, "hp": 100})  # on its way to the front
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "raid_size": 2, "raid_unit": "jeep"}))
    b.rules = {"jeep": {"cost": 500, "speed": 4.0}, "e3": {"cost": 300, "speed": 1.32}}
    b.choose_action(_state(_base_obs(units=units)), CTX)
    assert 200 in b._raiders


def test_raids_never_use_infantry_and_jeeps_get_built():
    units = [{"id": 100 + i, "type": "e3", "x": 40, "y": 40, "idle": True, "hp": 100} for i in range(6)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 2, "raid_unit": "jeep", "vehicle": "3tnk"}))
    b.rules = {"jeep": {"cost": 500, "speed": 4.0}, "e3": {"cost": 300, "speed": 1.32}, "3tnk": {"cost": 1150, "speed": 1.56}}
    obs = _base_obs(units=units)
    obs["available_production"] = obs["available_production"] + ["jeep"]
    orders = _orders(b.choose_action(_state(obs), CTX))
    assert not b._raiders  # no infantry raid
    assert {"cmd": "train", "item": "jeep", "count": 1} in orders  # jeeps first, then the tanks
    assert any("fast units" in p for p in b.status()["problems"])


def test_reinforcements_wait_at_home_until_a_group_can_leave():
    front = [{"id": 100 + i, "type": "3tnk", "x": 70, "y": 70, "idle": True, "hp": 100} for i in range(6)]
    one = [{"id": 300, "type": "e3", "x": 27, "y": 27, "idle": True, "hp": 100}]  # staging is near (27, 27)
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 80}))
    b.rules = {"3tnk": {"cost": 1150}, "e3": {"cost": 300}}
    orders = _orders(b.choose_action(_state(_base_obs(units=front + one)), CTX))
    assert not any(300 in o.get("units", []) for o in orders if o["cmd"] == "attack_move")  # waits
    group = [{"id": 300 + i, "type": "3tnk", "x": 27, "y": 27, "idle": True, "hp": 100} for i in range(4)]
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=front + group)), CTX))
    assert any({300, 301, 302, 303} <= set(o.get("units", [])) for o in orders if o["cmd"] == "attack_move")


def test_focus_fire_stays_on_one_target_until_it_dies():
    army = [{"id": 100 + i, "type": "3tnk", "x": 50, "y": 50, "idle": False, "hp": 100} for i in range(5)]
    foes = [{"id": 900, "type": "2tnk", "x": 54, "y": 50, "hp": 90}, {"id": 901, "type": "2tnk", "x": 55, "y": 51, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 50}))
    b.choose_action(_state(_base_obs(units=army, visible_enemies=foes)), CTX)
    first = b._focus_id
    # The other one is now weaker, but we keep killing the first.
    foes2 = [{**foes[0], "hp": 60}, {**foes[1], "hp": 20}] if first == 900 else [{**foes[0], "hp": 20}, {**foes[1], "hp": 60}]
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=army, visible_enemies=foes2)), CTX))
    assert b._focus_id == first
    assert all(o["target"] == first for o in orders if o["cmd"] == "attack" and set(o["units"]) & {100, 101, 102})


def test_pincer_flanks_round_the_back_then_closes():
    from agent import redalert_bot as rb
    army = [{"id": 100 + i, "type": "3tnk", "x": 30, "y": 50, "idle": True, "hp": 100} for i in range(5)]
    army += [{"id": 200 + i, "type": "1tnk", "x": 30, "y": 50, "idle": True, "hp": 100} for i in range(3)]
    army += [{"id": 300 + i, "type": "jeep", "x": 27, "y": 27, "idle": True, "hp": 100} for i in range(2)]  # the standing raid
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "pincer", "main_x": 60, "main_y": 50}))
    b.rules = {"3tnk": {"cost": 1150, "speed": 1.56}, "1tnk": {"cost": 700, "speed": 2.76}, "jeep": {"cost": 500, "speed": 4.0}}
    orders = _orders(b.choose_action(_state(_base_obs(units=army)), CTX))
    p = b._pincer
    assert set(p["flank"]) == {200, 201, 202}  # the fast tanks go round
    assert p["behind"][0] > 60  # behind the target, seen from us
    flank_move = [o for o in orders if o["cmd"] == "move" and set(o["units"]) == {200, 201, 202}]
    assert flank_move and abs(flank_move[0]["to"][1] - 50) >= 10  # out to a side first
    front = [o for o in orders if o["cmd"] == "attack_move" and set(o["units"]) >= {100, 101}]
    assert front and front[0]["to"][0] < 60  # the front waits short of the target
    assert "pincer" in b.status()["main_state"]
    # Flank at the behind point: both close in.
    b._pincer["leg"] = 1
    moved = [{**u, "x": p["behind"][0], "y": p["behind"][1]} if 200 <= u["id"] < 300 else u for u in army]
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=moved)), CTX))
    assert b._pincer["phase"] == "closing"
    assert any(o["cmd"] == "attack_move" and o["to"] == [60, 50] and {100, 200} <= set(o["units"]) for o in orders)


def test_pincer_window_is_flagged():
    b = bot()
    b._raid_info = {"size": 3, "phase": "withdraw", "at": [40, 40], "responders": 1800, "responders_from_their_base": 26}
    assert "PINCER WINDOW" in b._situation()


def test_many_vs_few_surrounds_from_several_sides():
    army = [{"id": 100 + i, "type": "3tnk", "x": 50 + (i % 3) * 2 - 2, "y": 50 + (i // 3) * 2 - 2, "idle": False, "hp": 100}
            for i in range(9)]
    foes = [{"id": 900 + i, "type": "2tnk", "x": 58, "y": 50 + i, "hp": 100} for i in range(2)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 50}))
    b.rules = {"3tnk": {"cost": 1150, "speed": 1.56}, "2tnk": {"cost": 850, "speed": 1.76}}
    orders = _orders(b.choose_action(_state(_base_obs(units=army, visible_enemies=foes)), CTX))
    assert "surround" in b.status()["main_state"]
    rings = {tuple(o["to"]) for o in orders if o["cmd"] == "attack_move" and set(o["units"]) & set(range(100, 109))}
    focus = [o for o in orders if o["cmd"] == "attack"]
    assert focus or len(rings) >= 2  # units in range focus fire; the rest close in from more than one side


def test_few_vs_many_backs_off_then_hits_back():
    import time as _t
    army = [{"id": 100 + i, "type": "2tnk", "x": 50, "y": 50, "idle": False, "hp": 100} for i in range(3)]
    foes = [{"id": 900 + i, "type": "3tnk", "x": 57, "y": 50, "hp": 100} for i in range(5)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 50}))
    b.rules = {"3tnk": {"cost": 1150, "speed": 1.56}, "2tnk": {"cost": 850, "speed": 1.76}}
    # They come on (seen closer than a moment ago): back off, away from them.
    b._enemy_hist = {900 + i: __import__("collections").deque([(_t.monotonic(), (60, 50))], maxlen=10) for i in range(5)}
    orders = _orders(b.choose_action(_state(_base_obs(units=army, visible_enemies=foes)), CTX))
    assert "backing off" in b.status()["main_state"]
    assert any(o["cmd"] == "move" and o["to"][0] < 50 for o in orders)
    # They stop: turn and hit.
    b._enemy_hist = {}
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=army, visible_enemies=foes)), CTX))
    assert "holding ground" in b.status()["main_state"]


def test_unopposed_raid_finishes_the_harvester():
    import time as _t
    from agent import redalert_bot as rb
    squad = [{"id": 100 + i, "type": "jeep", "x": 72, "y": 72, "idle": True, "hp": 100} for i in range(3)]
    harv = [{"id": 800, "type": "harv", "x": 74, "y": 73, "hp": 40}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_x": 74, "raid_y": 73}))
    b.rules = {"jeep": {"cost": 500, "speed": 4.0}, "harv": {"cost": 1100}}
    b.choose_action(_state(_base_obs(units=squad, visible_enemies=harv)), CTX)
    assert b._raid_phase == "strike"
    b._raid_strike_until -= rb.RAID_STRIKE_S + 1  # the usual strike time is over...
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=squad, visible_enemies=harv)), CTX)
    assert b._raid_phase == "strike"  # ...but nobody defends it: keep hitting until it dies


def test_a_new_raid_squad_forms_after_the_last_one_died_mid_withdraw():
    units = [{"id": 200 + i, "type": "ftrk", "x": 30, "y": 30, "idle": True, "hp": 100} for i in range(3)]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_unit": "ftrk"}))
    b.rules = {"ftrk": {"cost": 600, "speed": 2.76}}
    b._raid_phase = "withdraw"  # the old scout died while running
    b._raiders = set()
    b.choose_action(_state(_base_obs(units=units)), CTX)
    assert b._raiders == {200, 201, 202}


def test_raid_target_avoids_ore_covered_by_a_defense():
    b = bot()
    b._config = {"rules": {"maps": {"m": {"spawns": [[80, 80]], "layout": {
        "origin": [0, 0], "rows": ["." * 100] * 76 + ["." * 76 + "$$$$" + "." * 20] * 4 + ["." * 66 + "$$$" + "." * 31] * 3 + ["." * 100] * 17}}}}}
    b._enemy_buildings = {950: {"id": 950, "type": "pbox", "x": 77, "y": 77}}
    cell = b._raid_cell((80, 80))
    assert cell and cell[0] <= 69  # the uncovered patch, not the one under the pillbox
    b._enemy_buildings[951] = {"id": 951, "type": "gun", "x": 67, "y": 81}
    assert b._raid_cell((80, 80)) is None  # every patch covered: no raid there


def test_a_defense_shooting_the_raid_makes_it_run():
    squad = [{"id": 100 + i, "type": "ftrk", "x": 60, "y": 60, "idle": True, "hp": 100} for i in range(3)]
    pbox = [{"id": 950, "type": "pbox", "x": 63, "y": 60, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"raid_size": 3, "raid_unit": "ftrk", "raid_x": 80, "raid_y": 60}))
    b.rules = {"ftrk": {"cost": 600, "speed": 2.76}, "pbox": {"cost": 600}}
    b.choose_action(_state(_base_obs(units=squad, visible_enemy_buildings=pbox)), CTX)
    assert b._raid_phase == "withdraw"


def test_siege_units_shell_a_defense_from_outside_its_reach():
    v2 = [{"id": 100, "type": "v2rl", "x": 50, "y": 50, "idle": True, "hp": 100}]
    tsla = [{"id": 950, "type": "tsla", "x": 59, "y": 50, "hp": 100}]  # 9 cells: inside V2 range 10, outside tesla 7
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 70, "main_y": 50}))
    b.rules = {"v2rl": {"cost": 900, "speed": 1.76, "weapon": {"weapon": "SCUD", "range_cells": 10.0}}, "tsla": {"cost": 1200}}
    orders = _orders(b.choose_action(_state(_base_obs(units=v2, visible_enemy_buildings=tsla)), CTX))
    assert any(o["cmd"] == "attack" and o["target"] == 950 and 100 in o["units"] for o in orders)
    # Too close (6 cells): back out of the tesla's reach.
    b._kite_state.clear()
    b._sent.clear()
    close = [{**v2[0], "x": 53}]
    orders = _orders(b.choose_action(_state(_base_obs(units=close, visible_enemy_buildings=tsla)), CTX))
    assert any(o["cmd"] == "move" and 100 in o["units"] and o["to"][0] < 53 for o in orders)


def test_apc_raid_loads_drops_and_leaves():
    apc = {"id": 200, "type": "apc", "x": 27, "y": 27, "idle": True, "hp": 100}
    riders = [{"id": 300 + i, "type": "e3", "x": 26, "y": 27, "idle": True, "hp": 100} for i in range(5)]
    fake = FakeCommander({"raid_size": 1, "raid_unit": "apc", "raid_x": 70, "raid_y": 70})
    b = RedAlertBot(log=lambda m: None, advisor=fake)
    b.rules = {"apc": {"cost": 850, "speed": 3.12}, "e3": {"cost": 300, "speed": 1.32}}
    orders = _orders(b.choose_action(_state(_base_obs(units=[apc] + riders)), CTX))
    assert b._apc["phase"] == "load" and len(b._apc["riders"]) == 5
    assert any(o["cmd"] == "enter_transport" and o["target"] == 200 for o in orders)
    # They boarded (gone from view): the APC drives off.
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=[apc])), CTX)
    assert b._apc["phase"] == "go"
    # At the target: unload.
    b._sent.clear()
    orders = _orders(b.choose_action(_state(_base_obs(units=[{**apc, "x": 66, "y": 66}])), CTX))
    assert b._apc["phase"] == "fight" and {"cmd": "unload", "units": [200]} in orders
    # Their tanks come: pick up, then go home.
    tanks = [{"id": 900 + i, "type": "3tnk", "x": 72, "y": 66, "hp": 100} for i in range(3)]
    out = [{**r, "x": 66, "y": 66} for r in riders]
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=[{**apc, "x": 66, "y": 66}] + out, visible_enemies=tanks)), CTX)
    assert b._apc["phase"] == "pickup"
    b._sent.clear()
    b.choose_action(_state(_base_obs(units=[{**apc, "x": 66, "y": 66}], visible_enemies=tanks)), CTX)
    assert b._apc["phase"] == "home"


def test_army_never_charges_a_stronger_force_ahead():
    army = [{"id": 100 + i, "type": "3tnk", "x": 50, "y": 50, "idle": True, "hp": 100} for i in range(5)]  # 5,750
    foes = [{"id": 900 + i, "type": "3tnk", "x": 64, "y": 50, "hp": 100} for i in range(7)]               # 8,050 ahead
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 50}))
    b.rules = {"3tnk": {"cost": 1150, "speed": 1.56}}
    orders = _orders(b.choose_action(_state(_base_obs(units=army, visible_enemies=foes)), CTX))
    assert "outnumbered ahead" in b.status()["main_state"]
    assert not any(o["cmd"] == "attack_move" and o["to"][0] > 52 for o in orders)  # nobody walks forward


def test_no_surrounding_under_an_enemy_defense():
    army = [{"id": 100 + i, "type": "3tnk", "x": 50, "y": 50, "idle": False, "hp": 100} for i in range(8)]
    foes = [{"id": 900, "type": "e3", "x": 56, "y": 50, "hp": 100}]
    b = RedAlertBot(log=lambda m: None, advisor=FakeCommander({"main_action": "attack", "main_x": 80, "main_y": 50}))
    b.rules = {"3tnk": {"cost": 1150, "speed": 1.56}, "e3": {"cost": 300}, "tsla": {"cost": 1200}}
    b._enemy_buildings = {950: {"id": 950, "type": "tsla", "x": 59, "y": 50}}
    b.choose_action(_state(_base_obs(units=army, visible_enemies=foes)), CTX)
    assert "under their tsla" in b.status()["main_state"]
