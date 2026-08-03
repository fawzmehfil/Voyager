"""Stage 7B deterministic Civilization interface over the shared Voyager world."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from voyager.envs.civilization import VoyagerCivilizationEnv
from voyager.envs.parallel_env import VoyagerParallelEnv
from voyager.sim.registries_v2 import (
    V2_ENTITY_SLOT_COUNT,
    V2_FLAT_ACTION_COUNT,
    V2_MEANINGFUL_ACTIONS,
    V2_TARGET_COUNT,
    CivilizationV2Action,
    CivilizationV2Argument,
    CivilizationV2Verb,
    unflatten_v2_action,
)

TOOLS = ("axe", "pickaxe", "spear", "torch", "pack")
EQUIPPABLE_TOOLS = TOOLS[:4]
ITEMS = ("food", "wood", "stone", "raw_meat", "cooked_meat")
ENTITY_TYPES = {"agent": 1, "creature": 2, "pile": 3, "structure": 4}


class VoyagerCivilizationV2Env(VoyagerCivilizationEnv):
    """Targeted-action interface for the deterministic Stage 7B core."""

    metadata = {"render_modes": ["ansi"], "name": "VoyagerCivilization-v2"}  # noqa: RUF012

    def __init__(
        self,
        *,
        reward_mode: str = "dense",
        render_mode: str | None = None,
    ) -> None:
        super().__init__(reward_mode=reward_mode, render_mode=render_mode)
        self.world.civilization_version = 2
        observation_space = self._build_observation_space()
        action_space: spaces.Space = spaces.Dict(
            {
                "verb": spaces.Discrete(len(CivilizationV2Verb)),
                "argument": spaces.Discrete(len(CivilizationV2Argument)),
                "target": spaces.Discrete(V2_TARGET_COUNT),
            }
        )
        self.observation_spaces = {
            agent_id: observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {agent_id: action_space for agent_id in self.possible_agents}
        self._cache_tick = -1
        self._action_mask_cache: dict[str, np.ndarray] = {}
        self._entity_slot_cache: dict[str, list[str]] = {}
        self._conservation_cache: dict[str, int] | None = None
        self.performance_seconds = {
            "observation_generation": 0.0,
            "action_mask_generation": 0.0,
            "entity_slot_generation": 0.0,
            "ledger_reconciliation": 0.0,
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        self._clear_step_caches()
        return super().reset(seed=seed, options=options)

    def step(  # type: ignore[override]
        self,
        actions: dict[str, dict[str, int]],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        parsed: dict[str, CivilizationV2Action] = {}
        for agent_id in self.agents:
            payload = actions.get(
                agent_id,
                {
                    "verb": int(CivilizationV2Verb.NOOP),
                    "argument": int(CivilizationV2Argument.NONE),
                    "target": 0,
                },
            )
            if not isinstance(payload, dict) or set(payload) != {
                "verb",
                "argument",
                "target",
            }:
                raise ValueError(
                    "Civilization v2 actions require exactly 'verb', 'argument', and 'target'."
                )
            try:
                parsed[agent_id] = CivilizationV2Action(
                    CivilizationV2Verb(int(payload["verb"])),
                    CivilizationV2Argument(int(payload["argument"])),
                    int(payload["target"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed Civilization v2 action for {agent_id}.") from exc
        return VoyagerParallelEnv.step(self, parsed)  # type: ignore[arg-type,return-value]

    def action_mask(self, agent_id: str) -> np.ndarray:
        self._refresh_step_caches()
        if agent_id not in self._action_mask_cache:
            started = time.perf_counter()
            self._action_mask_cache[agent_id] = self.world.v2_action_mask(agent_id)
            self.performance_seconds["action_mask_generation"] += (
                time.perf_counter() - started
            )
        return self._action_mask_cache[agent_id].copy()

    def _entity_slots(self, agent_id: str) -> list[str]:
        self._refresh_step_caches()
        if agent_id not in self._entity_slot_cache:
            started = time.perf_counter()
            self._entity_slot_cache[agent_id] = self.world.v2_entity_slots(agent_id)
            self.performance_seconds["entity_slot_generation"] += (
                time.perf_counter() - started
            )
        return list(self._entity_slot_cache[agent_id])

    def _conservation(self) -> dict[str, int]:
        self._refresh_step_caches()
        if self._conservation_cache is None:
            started = time.perf_counter()
            self._conservation_cache = self.world.reconcile_v2_ledger()
            self.performance_seconds["ledger_reconciliation"] += (
                time.perf_counter() - started
            )
        return dict(self._conservation_cache)

    def _refresh_step_caches(self) -> None:
        state = self.world.state
        tick = -1 if state is None else state.step_count
        if tick != self._cache_tick:
            self._clear_step_caches(tick)

    def _clear_step_caches(self, tick: int = -1) -> None:
        self._cache_tick = tick
        self._action_mask_cache.clear()
        self._entity_slot_cache.clear()
        self._conservation_cache = None

    def _build_observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "local_tiles": spaces.Box(0, 255, shape=(7, 7, 7), dtype=np.uint8),
                "self_state": spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32),
                "inventory": spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32),
                "tools_owned": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                "tool_equipped": spaces.Box(0, 1, shape=(4,), dtype=np.int8),
                "torch_charge": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "role": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
                "time": spaces.Box(0.0, 2.0, shape=(6,), dtype=np.float32),
                "camp": spaces.Box(0.0, 1.0, shape=(20,), dtype=np.float32),
                "entity_slots": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(V2_ENTITY_SLOT_COUNT, 12),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    0,
                    1,
                    shape=(V2_FLAT_ACTION_COUNT,),
                    dtype=np.int8,
                ),
            }
        )

    def _observation(self, agent_id: str) -> dict[str, np.ndarray]:
        started = time.perf_counter()
        base = VoyagerCivilizationEnv._observation(self, agent_id)
        state = self.world.state
        if state is None:
            raise RuntimeError("Environment must be reset before producing observations.")
        agent = state.agents[agent_id]
        local_tiles = base["local_tiles"].copy()
        for torch_agent in state.agents.values():
            if not (
                torch_agent.equipped_tool == "torch"
                and torch_agent.tool_charges.get("torch", 0) > 0
            ):
                continue
            for y in range(torch_agent.y - 1, torch_agent.y + 2):
                for x in range(torch_agent.x - 1, torch_agent.x + 2):
                    if abs(x - torch_agent.x) + abs(y - torch_agent.y) > 1:
                        continue
                    row, column = y - agent.y + 3, x - agent.x + 3
                    if 0 <= row < 7 and 0 <= column < 7:
                        local_tiles[row, column, 6] = 255
        capacity = 15 if "pack" in agent.tools else 10
        freshness = np.zeros(3, dtype=np.float32)
        for index, kind in enumerate(("berries", "raw_meat", "cooked_meat")):
            expiries = [
                lot.expires_tick
                for lot in agent.food_lots
                if lot.kind == kind and lot.quantity > 0 and lot.expires_tick is not None
            ]
            if expiries:
                lifetime = {"berries": 240, "raw_meat": 90, "cooked_meat": 360}[kind]
                freshness[index] = max(0.0, min(1.0, (min(expiries) - state.step_count) / lifetime))

        camp = np.zeros(20, dtype=np.float32)
        near_camp = abs(agent.x - state.camp.x) <= 3 and abs(agent.y - state.camp.y) <= 3
        if near_camp:
            scale = max(1, 15 * self._configured_num_agents)
            camp[:5] = [state.camp.stockpile.get(item, 0) / scale for item in ITEMS]
            camp[5:10] = [min(1.0, len(state.camp.tool_stockpile.get(tool, [])) / 10.0) for tool in TOOLS]
            for offset, name in enumerate(("workbench", "campfire", "shelter")):
                structure = state.structures[name]
                camp[10 + offset] = structure.progress
                camp[13 + offset] = structure.condition / 100.0
            camp[16] = state.structures["campfire"].fuel / 120.0
            camp[17] = len(state.structures["shelter"].occupants) / 6.0
            camp[18] = state.structures["shelter"].repair_labor / 40.0
            camp[19] = 1.0

        slots = np.zeros((V2_ENTITY_SLOT_COUNT, 12), dtype=np.float32)
        for index, entity_ref in enumerate(self._entity_slots(agent_id)):
            kind, entity_id = entity_ref.split(":", maxsplit=1)
            x, y, condition, active, downed, dead = self._entity_values(kind, entity_id)
            hostility = float(kind == "creature" and entity_id.startswith("stalker"))
            distance = abs(x - agent.x) + abs(y - agent.y)
            eligible = float(distance <= 1)
            slots[index] = np.array(
                [
                    1.0,
                    ENTITY_TYPES[kind] / 4.0,
                    (x - agent.x) / 3.0,
                    (y - agent.y) / 3.0,
                    condition,
                    active,
                    downed,
                    dead,
                    hostility,
                    eligible,
                    float(kind == "structure" and condition < 1.0),
                    float(kind == "agent" and downed == 1.0),
                ],
                dtype=np.float32,
            )

        observation = {
            "local_tiles": local_tiles,
            "self_state": np.array(
                [
                    agent.health / 100.0,
                    agent.hunger / 100.0,
                    agent.energy / 100.0,
                    sum(agent.inventory.values()) / capacity,
                    float(agent.sheltered),
                    float(agent.life_state == "active"),
                    float(agent.life_state == "downed"),
                    float(agent.life_state == "dead"),
                    agent.downed_ticks / 20.0,
                ],
                dtype=np.float32,
            ),
            "inventory": np.concatenate(
                (
                    np.array(
                        [agent.inventory.get(item, 0) / capacity for item in ITEMS],
                        dtype=np.float32,
                    ),
                    freshness,
                )
            ),
            "tools_owned": np.array([tool in agent.tools for tool in TOOLS], dtype=np.int8),
            "tool_equipped": np.array(
                [agent.equipped_tool == tool for tool in EQUIPPABLE_TOOLS],
                dtype=np.int8,
            ),
            "torch_charge": np.array(
                [agent.tool_charges.get("torch", 0) / 30.0], dtype=np.float32
            ),
            "role": base["role"],
            "time": base["time"],
            "camp": camp,
            "entity_slots": slots,
            "action_mask": self.action_mask(agent_id),
        }
        self.performance_seconds["observation_generation"] += (
            time.perf_counter() - started
        )
        return observation

    def _entity_values(
        self, kind: str, entity_id: str
    ) -> tuple[int, int, float, float, float, float]:
        state = self.world.state
        assert state is not None
        if kind == "agent":
            entity = state.agents[entity_id]
            return (
                entity.x,
                entity.y,
                entity.health / 100.0,
                float(entity.life_state == "active"),
                float(entity.life_state == "downed"),
                float(entity.life_state == "dead"),
            )
        if kind == "creature":
            creature = state.creatures[entity_id]
            return creature.x, creature.y, creature.health / creature.max_health, 1.0, 0.0, 0.0
        if kind == "structure":
            structure = state.structures[entity_id]
            return structure.x, structure.y, structure.condition / 100.0, 1.0, 0.0, 0.0
        pile = state.ground_piles[entity_id]
        return pile.x, pile.y, min(1.0, pile.quantity / 2.0), 1.0, 0.0, 0.0

    def _info(
        self,
        agent_id: str,
        event: str,
        reward_components: dict[str, float] | None = None,
        dense_reward_components: dict[str, float] | None = None,
        new_achievements: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        info = VoyagerCivilizationEnv._info(
            self,
            agent_id,
            event,
            reward_components,
            dense_reward_components,
            new_achievements,
        )
        state = self.world.state
        assert state is not None
        agent = state.agents[agent_id]
        info.update(
            {
                "interface_version": 2,
                "entity_slots": self._entity_slots(agent_id),
                "life_state": agent.life_state,
                "downed_ticks": agent.downed_ticks,
                "tool_charges": dict(agent.tool_charges),
                "ledger_entries": len(state.ledger),
                "conservation": self._conservation(),
            }
        )
        info.pop("target_slots", None)
        return info


class CivilizationV2FlattenedActionWrapper(
    ParallelEnv[str, dict[str, np.ndarray], int]
):
    """Expose exactly the v2 registry as a one-dimensional action space."""

    metadata = VoyagerCivilizationV2Env.metadata

    def __init__(self, env: VoyagerCivilizationV2Env | None = None) -> None:
        self.env = env or VoyagerCivilizationV2Env()
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.observation_spaces = dict(self.env.observation_spaces)
        action_space: spaces.Space = spaces.Discrete(V2_FLAT_ACTION_COUNT)
        self.action_spaces = {agent_id: action_space for agent_id in self.possible_agents}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        structured = {
            agent_id: dict(zip(("verb", "argument", "target"), unflatten_v2_action(action), strict=True))
            for agent_id, action in actions.items()
        }
        result = self.env.step(structured)
        self.agents = list(self.env.agents)
        return result

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def state(self) -> dict[str, object]:
        return self.env.global_state()

    @property
    def performance_seconds(self) -> dict[str, float]:
        return self.env.performance_seconds

    def close(self) -> None:
        self.env.close()


assert len(V2_MEANINGFUL_ACTIONS) == V2_FLAT_ACTION_COUNT
