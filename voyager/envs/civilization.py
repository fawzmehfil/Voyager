"""Stage 7A Civilization interface over Voyager's shared multi-agent world."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from voyager.envs.parallel_env import VoyagerParallelEnv
from voyager.sim.constants import ROLE_COUNT, Terrain
from voyager.sim.multi_world import ROLE_IDS
from voyager.sim.registries import (
    ARGUMENT_COUNT,
    FLAT_ACTION_COUNT,
    MEANINGFUL_ACTION_PAIRS,
    TARGET_SLOT_COUNT,
    VERB_COUNT,
    CivilizationAction,
    CivilizationArgument,
    CivilizationVerb,
    unflatten_action,
)
from voyager.sim.scenarios import (
    CIVILIZATION_MAP_SIZE,
    CIVILIZATION_MAX_STEPS,
    CIVILIZATION_SCENARIO_ID,
)

STRUCTURE_CHANNEL = {"workbench": 1, "campfire": 2, "shelter": 3}
CREATURE_CHANNEL = {"island_deer": 1, "night_stalker": 2}


class VoyagerCivilizationEnv(VoyagerParallelEnv):
    """Structured local-observation interface for the handcrafted Stage 7A scenario."""

    metadata = {"render_modes": ["ansi"], "name": "VoyagerCivilization-v1"}  # noqa: RUF012

    def __init__(
        self,
        *,
        reward_mode: str = "dense",
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            num_agents=10,
            map_size=CIVILIZATION_MAP_SIZE,
            max_steps=CIVILIZATION_MAX_STEPS,
            local_view_size=7,
            inventory_capacity=10,
            storm_start_step=10_000,
            storm_interval=0,
            storm_duration=0,
            storm_damage=0.0,
            food_regen_interval=0,
            food_spawn_rate=0.0,
            reward_mode=reward_mode,  # type: ignore[arg-type]
            scenario_id=CIVILIZATION_SCENARIO_ID,
            render_mode=render_mode,
        )
        action_space: spaces.Space = spaces.Dict(
            {
                "verb": spaces.Discrete(VERB_COUNT),
                "argument": spaces.Discrete(ARGUMENT_COUNT),
            }
        )
        self.action_spaces = {agent_id: action_space for agent_id in self.possible_agents}

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
        parsed: dict[str, CivilizationAction] = {}
        for agent_id in self.agents:
            payload = actions.get(
                agent_id,
                {"verb": int(CivilizationVerb.NOOP), "argument": int(CivilizationArgument.NONE)},
            )
            if not isinstance(payload, dict) or set(payload) != {"verb", "argument"}:
                raise ValueError("Civilization actions require exactly 'verb' and 'argument'.")
            try:
                parsed[agent_id] = CivilizationAction(
                    CivilizationVerb(int(payload["verb"])),
                    CivilizationArgument(int(payload["argument"])),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed Civilization action for {agent_id}.") from exc
        return super().step(parsed)  # type: ignore[arg-type,return-value]

    def action_mask(self, agent_id: str) -> np.ndarray:
        return self.world.civilization_action_mask(agent_id)

    def global_state(self) -> dict[str, object]:
        return self.world.global_state()

    def _build_observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "local_tiles": spaces.Box(
                    low=0,
                    high=255,
                    shape=(7, 7, 7),
                    dtype=np.uint8,
                ),
                "self_state": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),
                "inventory": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
                "tools_owned": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                "tool_equipped": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                "role": spaces.Box(low=0.0, high=1.0, shape=(ROLE_COUNT,), dtype=np.float32),
                "time": spaces.Box(low=0.0, high=2.0, shape=(6,), dtype=np.float32),
                "camp": spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32),
                "target_slots": spaces.Box(
                    low=-1.0,
                    high=2.0,
                    shape=(TARGET_SLOT_COUNT, 7),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(VERB_COUNT, ARGUMENT_COUNT),
                    dtype=np.int8,
                ),
            }
        )

    def _observation(self, agent_id: str) -> dict[str, np.ndarray]:
        state = self.world.state
        if state is None:
            raise RuntimeError("Environment must be reset before producing observations.")
        agent = state.agents[agent_id]
        half = 3
        local_tiles = np.zeros((7, 7, 7), dtype=np.uint8)
        local_tiles[:, :, 0] = Terrain.WATER
        occupied = self.world.occupied_positions()
        structures = {(value.x, value.y): value for value in state.structures.values()}
        creatures = {(value.x, value.y): value for value in state.creatures.values() if value.alive}
        ambient = float(self.world.civilization_time()["ambient_light"])
        for row, y in enumerate(range(agent.y - half, agent.y + half + 1)):
            for col, x in enumerate(range(agent.x - half, agent.x + half + 1)):
                if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                    continue
                local_tiles[row, col, 0] = state.terrain[y, x]
                if state.resource_quantities[y, x] > 0:
                    local_tiles[row, col, 1] = state.resource_ids[y, x]
                    local_tiles[row, col, 2] = state.resource_quantities[y, x]
                structure = structures.get((x, y))
                if structure:
                    local_tiles[row, col, 3] = STRUCTURE_CHANNEL[structure.type]
                creature = creatures.get((x, y))
                if creature:
                    local_tiles[row, col, 4] = CREATURE_CHANNEL[creature.type]
                visible_agent = occupied.get((x, y))
                if visible_agent:
                    local_tiles[row, col, 5] = int(ROLE_IDS[state.agents[visible_agent].role]) + 1
                light = 1.0 if self.world._inside_fire_radius(x, y) else ambient
                local_tiles[row, col, 6] = round(light * 255)

        role = np.zeros(ROLE_COUNT, dtype=np.float32)
        role[ROLE_IDS[agent.role]] = 1.0
        time = self.world.civilization_time()
        phase = str(time["phase"])
        phase_one_hot = [
            1.0 if phase == "morning" else 0.0,
            1.0 if phase == "afternoon" else 0.0,
            1.0 if phase == "night" else 0.0,
        ]
        structures_state = state.structures
        near_camp = abs(agent.x - state.camp.x) <= 3 and abs(agent.y - state.camp.y) <= 3
        camp = np.zeros(15, dtype=np.float32)
        if near_camp:
            stockpile_scale = max(1, self.inventory_capacity * self._configured_num_agents)
            camp[:5] = [
                state.camp.stockpile.get(item, 0) / stockpile_scale
                for item in ("food", "wood", "stone", "raw_meat", "cooked_meat")
            ]
            camp[5:8] = [
                structures_state[name].progress for name in ("workbench", "campfire", "shelter")
            ]
            camp[8:11] = [
                float(structures_state[name].complete)
                for name in ("workbench", "campfire", "shelter")
            ]
            camp[11] = structures_state["campfire"].fuel / 120.0
            camp[12] = len(structures_state["shelter"].occupants) / 6.0
            camp[13] = structures_state["shelter"].capacity / 6.0
            camp[14] = 1.0

        target_slots = np.zeros((TARGET_SLOT_COUNT, 7), dtype=np.float32)
        for slot, creature_id in enumerate(self.world.target_slots(agent_id)):
            creature = state.creatures[creature_id]
            target_slots[slot] = np.array(
                [
                    1.0,
                    float(CREATURE_CHANNEL[creature.type]),
                    (creature.x - agent.x) / 3.0,
                    (creature.y - agent.y) / 3.0,
                    creature.health / max(1, creature.max_health),
                    0.0 if creature.type == "island_deer" else 1.0,
                    float(creature.target == agent_id),
                ],
                dtype=np.float32,
            )

        return {
            "local_tiles": local_tiles,
            "self_state": np.array(
                [
                    agent.health / 100.0,
                    agent.hunger / 100.0,
                    agent.energy / 100.0,
                    sum(agent.inventory.values()) / self.inventory_capacity,
                    float(agent.sheltered),
                    float(agent.alive),
                ],
                dtype=np.float32,
            ),
            "inventory": np.array(
                [
                    agent.inventory.get(item, 0) / self.inventory_capacity
                    for item in ("food", "wood", "stone", "raw_meat", "cooked_meat")
                ],
                dtype=np.float32,
            ),
            "tools_owned": np.array([int("spear" in agent.tools)], dtype=np.int8),
            "tool_equipped": np.array([int(agent.equipped_tool == "spear")], dtype=np.int8),
            "role": role,
            "time": np.array(
                [
                    min(2.0, (int(time["day"]) - 1) / 1.0),
                    *phase_one_hot,
                    float(time["phase_progress"]),
                    float(time["ambient_light"]),
                ],
                dtype=np.float32,
            ),
            "camp": camp,
            "target_slots": target_slots,
            "action_mask": self.action_mask(agent_id),
        }

    def _info(
        self,
        agent_id: str,
        event: str,
        reward_components: dict[str, float] | None = None,
        dense_reward_components: dict[str, float] | None = None,
        new_achievements: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        info = super()._info(
            agent_id,
            event,
            reward_components,
            dense_reward_components,
            new_achievements,
        )
        state = self.world.state
        if state is None:
            return info
        agent = state.agents[agent_id]
        info.update(
            {
                "scenario_id": CIVILIZATION_SCENARIO_ID,
                "time": self.world.civilization_time(),
                "tools": sorted(agent.tools),
                "equipped_tool": agent.equipped_tool,
                "sheltered": agent.sheltered,
                "target_slots": self.world.target_slots(agent_id),
                "structures": {
                    key: self.world._structure_payload(value)
                    for key, value in sorted(state.structures.items())
                },
                "creature_events": list(state.events),
                "action_mask": self.action_mask(agent_id),
            }
        )
        return info


class CivilizationFlattenedActionWrapper(ParallelEnv[str, dict[str, np.ndarray], int]):
    """Expose the same Civilization world through a stable discrete action registry."""

    metadata = VoyagerCivilizationEnv.metadata

    def __init__(self, env: VoyagerCivilizationEnv | None = None) -> None:
        self.env = env or VoyagerCivilizationEnv()
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.observation_spaces = {
            agent_id: self._flattened_observation_space() for agent_id in self.possible_agents
        }
        action_space: spaces.Space = spaces.Discrete(FLAT_ACTION_COUNT)
        self.action_spaces = {agent_id: action_space for agent_id in self.possible_agents}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return self._project_observations(observations), infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        structured = {
            agent_id: {"verb": verb, "argument": argument}
            for agent_id, action in actions.items()
            for verb, argument in [unflatten_action(int(action))]
        }
        observations, rewards, terminations, truncations, infos = self.env.step(structured)
        self.agents = list(self.env.agents)
        return (
            self._project_observations(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def state(self) -> dict[str, object]:
        return self.env.global_state()

    def close(self) -> None:
        self.env.close()

    def _flattened_observation_space(self) -> spaces.Dict:
        original = self.env.observation_spaces[self.possible_agents[0]]
        assert isinstance(original, spaces.Dict)
        projected = dict(original.spaces)
        projected["action_mask"] = spaces.Box(
            low=0,
            high=1,
            shape=(FLAT_ACTION_COUNT,),
            dtype=np.int8,
        )
        return spaces.Dict(projected)

    def _project_observations(
        self,
        observations: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        projected: dict[str, dict[str, np.ndarray]] = {}
        for agent_id, observation in observations.items():
            canonical_mask = observation["action_mask"]
            flat_mask = np.array(
                [
                    canonical_mask[int(verb), int(argument)]
                    for verb, argument in MEANINGFUL_ACTION_PAIRS
                ],
                dtype=np.int8,
            )
            projected[agent_id] = {**observation, "action_mask": flat_mask}
        return projected
