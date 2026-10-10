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
    attack = [o for o in orders if o["cmd"] == "attack_move"]
    assert attack and attack[0]["to"] == [22, 23] and len(attack[0]["units"]) == 3


def test_attacks_with_big_army_and_respects_command_cap():
    army = [{"id": 100 + i, "type": "e1", "x": 25, "y": 25, "idle": True} for i in range(120)]
    obs = _obs(buildings=[{"id": 1, "type": "fact", "x": 20, "y": 20}], units=army)
    orders = _orders(bot().choose_action(_state(obs), CTX))
    attack = [o for o in orders if o["cmd"] == "attack_move"]
    assert attack and attack[0]["to"] == [80, 80]
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

    def observe(self, raw):
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
    assert any(o["cmd"] == "attack_move" and o["to"] == [70, 30] for o in orders)


def test_tactical_retreat():
    b = RedAlertBot(log=lambda m: None)
    b.choose_action(_state(_army_obs(12)), CTX)
    assert b._attacking
    b.advisor = FakeCommanders(commands=[{"type": "retreat"}])
    orders = _orders(b.choose_action(_state(_army_obs(12)), CTX))
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
    monkeypatch.delenv("FAST_MODEL", raising=False)
    c = advisor.commanders_from_env(log=lambda m: None)
    assert c._layers["strategy"].model == "claude-fable-5-1" and c._layers["strategy"].effort == "medium"
    assert c._layers["tactical"].model == "claude-opus-5-5" and c._layers["tactical"].effort == "low"
    assert "fast" not in c._layers
    sonnet = advisor.Layer("fast", None, "claude-sonnet-5-5", "low", 4, 15, 10, "", {}, print)
    assert sonnet._request_kwargs()["thinking"] == {"type": "between_tools"}
    assert "thinking" not in c._layers["strategy"]._request_kwargs()
    c._on_strategy({"posture": "aggressive", "unit_focus": "tanks", "defenses": 9, "max_refineries": 2, "plan": "rush"})
    c._on_tactical({"attack_size": 2, "action": "retreat", "target_x": -1, "target_y": -1, "note": "x"})
    k = c.knobs()
    assert k["posture"] == "aggressive" and k["defenses"] == 6 and k["attack_size"] == 4 and k["target"] is None
    assert c.pop_commands() == [{"type": "retreat"}]


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
