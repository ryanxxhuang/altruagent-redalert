"""Our tournament agent: a Red Alert bot (``agent/redalert_bot.py``) with an
three Claude command layers (``agent/advisor.py``: Fable strategy, Opus
tactics, Sonnet micro; on when ``ANTHROPIC_API_KEY`` is in ``.env``).

    python -m agent --match        # test matches
    python -m agent --tournament   # tournament day

``create_agent()`` runs once per match, in that match's own process, so each
match gets a fresh bot. Any other game falls back to the first legal action.
"""

from __future__ import annotations

from dotenv import load_dotenv

from altruagent import WAIT, DecisionContext, GameState

from agent.advisor import commanders_from_env
from agent.redalert_bot import RedAlertBot, dump_dir_from_env


class Agent:
    def __init__(self) -> None:
        self._bot: RedAlertBot | None = None

    def _red_alert(self) -> RedAlertBot:
        if self._bot is None:
            self._bot = RedAlertBot(advisor=commanders_from_env(), dump_dir=dump_dir_from_env())
        return self._bot

    def choose_action(self, state: GameState, context: DecisionContext):
        if (context.game_type or state.game_name) == "red_alert":
            return self._red_alert().choose_action(state, context)
        return state.legal_actions[0] if state.legal_actions else WAIT

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        if self._bot is not None:
            self._bot.on_action_result(result, context)


def create_agent():
    load_dotenv()
    return Agent()
