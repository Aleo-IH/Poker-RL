"""RLlib MultiAgentEnv for multi-agent No-Limit Texas Hold'em.

Turn-based: one agent acts per step.  Observations are Dict spaces
containing a feature vector and an action mask for illegal-action filtering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from . import evaluator
from .game import NUM_ACTIONS, PokerGame, Street
from .observation import ObservationBuilder


class PokerEnv(MultiAgentEnv):
    """No-Limit Texas Hold'em for 2-10 players (RLlib MultiAgentEnv).

    All agents share the same observation / action spaces.
    Rewards are zero-sum and sparse (assigned at hand end only).
    """

    metadata = {"render_modes": ["human"], "name": "nlhe_poker_v0"}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        self.num_players = config.get("num_players", 6)
        self.initial_stack = config.get("initial_stack", 1000)
        self.small_blind = config.get("small_blind", 5)
        self.big_blind = config.get("big_blind", 10)
        self.max_history_per_street = config.get("max_history_per_street", 10)
        self.cash_game = config.get("cash_game", False)
        self.max_hands_per_episode = config.get("max_hands_per_episode", 500)

        self._obs_builder = ObservationBuilder(
            self.num_players, self.max_history_per_street
        )
        self._obs_dim = self._obs_builder.obs_dim
        self._game: PokerGame | None = None
        self._active_agents: set[str] = set()
        self._hands_played: int = 0

        self._agent_ids = {f"player_{i}" for i in range(self.num_players)}
        self.agents = list(self._agent_ids)
        self.possible_agents = list(self._agent_ids)

        obs_space = Dict(
            {
                "observation": Box(
                    low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
                ),
                "action_mask": Box(
                    low=0.0, high=1.0, shape=(NUM_ACTIONS,), dtype=np.float32
                ),
            }
        )
        act_space = Discrete(NUM_ACTIONS)
        # New API stack (RLlib) requires per-agent dict spaces
        self.observation_spaces = {aid: obs_space for aid in self._agent_ids}
        self.action_spaces = {aid: act_space for aid in self._agent_ids}
        # Legacy single spaces (for sampling etc.)
        self.observation_space = obs_space
        self.action_space = act_space

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        rng = np.random.default_rng(seed)
        self._game = PokerGame(
            num_players=self.num_players,
            initial_stack=self.initial_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            rng=rng,
            cash_game=self.cash_game,
        )
        self._hands_played = 0
        self._game.new_hand()
        self._active_agents = set()

        if self._game.is_hand_over():
            return {}, {}

        agent = f"player_{self._game.current_player}"
        self._active_agents.add(agent)
        return {agent: self._build_obs(self._game.current_player)}, {agent: {}}

    def start_next_hand(self) -> tuple[dict, dict]:
        """Start the next hand in the same session (cash game only). Call after hand over."""
        if not self.cash_game or self._game is None:
            raise RuntimeError("start_next_hand requires cash_game and an existing game")
        self._game.new_hand()
        self._active_agents = set()
        if self._game.is_hand_over():
            return {}, {}
        agent = f"player_{self._game.current_player}"
        self._active_agents.add(agent)
        return {agent: self._build_obs(self._game.current_player)}, {agent: {}}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action_dict: dict[str, int]) -> tuple[dict, ...]:
        assert self._game is not None and not self._game.is_hand_over()

        acting_agent = f"player_{self._game.current_player}"
        action = action_dict[acting_agent]

        legal = self._game.get_legal_actions()
        if not legal[action]:
            action = next(i for i, ok in enumerate(legal) if ok)

        self._game.apply_action(action)

        obs: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terminateds: dict[str, bool] = {"__all__": False}
        truncateds: dict[str, bool] = {"__all__": False}
        infos: dict[str, dict] = {}

        if self._game.is_hand_over():
            game_rewards = self._game.get_rewards()
            for agent in self._active_agents:
                pid = int(agent.split("_")[1])
                rewards[agent] = game_rewards[pid]

            if self.cash_game:
                session_over = (
                    self._game.count_eligible() <= 1
                    or self._hands_played >= self.max_hands_per_episode
                )
                if not session_over:
                    self._hands_played += 1
                    self._game.new_hand()
                    if self._game.is_hand_over():
                        session_over = True
                    else:
                        next_agent = f"player_{self._game.current_player}"
                        self._active_agents = {next_agent}
                        obs[next_agent] = self._build_obs(self._game.current_player)
                        for a in self._agent_ids:
                            terminateds[a] = False
                            truncateds[a] = False
                        infos[next_agent] = {}
                        return obs, rewards, terminateds, truncateds, infos

                truncateds["__all__"] = (
                    self._hands_played >= self.max_hands_per_episode
                )

            for agent in self._agent_ids:
                terminateds[agent] = True
                truncateds[agent] = False
            terminateds["__all__"] = True
        else:
            next_agent = f"player_{self._game.current_player}"
            self._active_agents.add(next_agent)
            obs[next_agent] = self._build_obs(self._game.current_player)
            rewards[acting_agent] = 0.0
            terminateds[acting_agent] = False
            truncateds[acting_agent] = False
            infos[next_agent] = {}

        return obs, rewards, terminateds, truncateds, infos

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> None:
        if self._game is None:
            print("[No game in progress]")
            return

        g = self._game
        board_str = " ".join(evaluator.card_to_str(c) for c in g.board) or "--"
        print(f"\n{'='*50}")
        print(f"Street: {g.street.name}  |  Board: {board_str}  |  Pot: {g.pot}")
        print("-" * 50)
        for i in range(g.num_players):
            tag = ""
            if i == g.dealer_idx:
                tag += " (D)"
            if g.folded[i]:
                tag += " [FOLD]"
            if g.all_in[i]:
                tag += " [ALL-IN]"
            if not g.is_hand_over() and i == g.current_player:
                tag += " <-- acting"
            cards = (
                " ".join(evaluator.card_to_str(c) for c in g.hole_cards[i])
                if not g.folded[i]
                else "XX XX"
            )
            print(f"  player_{i}: {cards}  stack={g.stacks[i]}  bet={g.bets[i]}{tag}")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_obs(self, player: int) -> dict[str, np.ndarray]:
        obs = self._obs_builder.build(self._game, player)
        mask = self._obs_builder.build_action_mask(self._game).astype(np.float32)
        return {"observation": obs, "action_mask": mask}
