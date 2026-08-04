"""Controlled Stage 7C tasks that isolate failures in the full island probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pettingzoo import ParallelEnv

from voyager.envs.civilization_v2 import (
    CivilizationV2FlattenedActionWrapper,
    VoyagerCivilizationV2Env,
)
from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    V2_MEANINGFUL_ACTIONS,
    CivilizationV2Argument,
    CivilizationV2Verb,
)
from voyager.training.civilization_probe import (
    PROBE_REWARD_VERSION,
    CivilizationProbeRewardWrapper,
)

LearningTask = Literal[
    "gather_wood",
    "gather_stone",
    "return_to_camp",
    "delivery",
    "construction",
    "survival",
]
LEARNING_TASKS: tuple[LearningTask, ...] = (
    "delivery",
    "construction",
    "survival",
)
DELIVERY_DIAGNOSTIC_TASKS: tuple[LearningTask, ...] = (
    "gather_wood",
    "gather_stone",
    "return_to_camp",
)
ALL_LEARNING_TASKS: tuple[LearningTask, ...] = (
    *DELIVERY_DIAGNOSTIC_TASKS,
    *LEARNING_TASKS,
)
LEARNING_TASK_CONTRACTS: dict[LearningTask, str] = {
    task: f"civilization_learning_ladder_{task}_v1"
    for task in ALL_LEARNING_TASKS
}
CONTRACT_TO_LEARNING_TASK = {
    contract: task for task, contract in LEARNING_TASK_CONTRACTS.items()
}
WORKBENCH_REQUIREMENTS = {"wood": 6, "stone": 2}


@dataclass(frozen=True, slots=True)
class LearningTaskDefinition:
    """One diagnostic preset over the unchanged Civilization v2 mechanics."""

    task: LearningTask
    start_tick: int
    end_tick: int
    allowed_actions: np.ndarray

    @property
    def horizon(self) -> int:
        return self.end_tick - self.start_tick


def learning_task_definition(task: LearningTask) -> LearningTaskDefinition:
    """Return the immutable action subset and timing for one learning test."""

    allowed = np.zeros(V2_FLAT_ACTION_COUNT, dtype=np.int8)
    if task in {"gather_wood", "gather_stone"}:
        verbs = {
            CivilizationV2Verb.NOOP,
            CivilizationV2Verb.MOVE,
            CivilizationV2Verb.INTERACT,
            CivilizationV2Verb.REST,
        }
        for index, triple in enumerate(V2_MEANINGFUL_ACTIONS):
            if CivilizationV2Verb(triple[0]) in verbs:
                allowed[index] = 1
        return LearningTaskDefinition(task, 0, 100, allowed)
    if task == "return_to_camp":
        verbs = {
            CivilizationV2Verb.NOOP,
            CivilizationV2Verb.MOVE,
            CivilizationV2Verb.REST,
        }
        for index, triple in enumerate(V2_MEANINGFUL_ACTIONS):
            verb = CivilizationV2Verb(triple[0])
            argument = CivilizationV2Argument(triple[1])
            if verb in verbs or (
                verb == CivilizationV2Verb.DEPOSIT
                and argument
                in {CivilizationV2Argument.WOOD, CivilizationV2Argument.STONE}
            ):
                allowed[index] = 1
        return LearningTaskDefinition(task, 0, 60, allowed)
    if task == "delivery":
        verbs = {
            CivilizationV2Verb.NOOP,
            CivilizationV2Verb.MOVE,
            CivilizationV2Verb.INTERACT,
            CivilizationV2Verb.REST,
        }
        for index, triple in enumerate(V2_MEANINGFUL_ACTIONS):
            verb = CivilizationV2Verb(triple[0])
            argument = CivilizationV2Argument(triple[1])
            if verb in verbs or (
                verb == CivilizationV2Verb.DEPOSIT
                and argument
                in {CivilizationV2Argument.WOOD, CivilizationV2Argument.STONE}
            ):
                allowed[index] = 1
        return LearningTaskDefinition(task, 0, 150, allowed)
    if task == "construction":
        verbs = {
            CivilizationV2Verb.NOOP,
            CivilizationV2Verb.MOVE,
            CivilizationV2Verb.REST,
        }
        for index, triple in enumerate(V2_MEANINGFUL_ACTIONS):
            verb = CivilizationV2Verb(triple[0])
            argument = CivilizationV2Argument(triple[1])
            if verb in verbs or (
                verb == CivilizationV2Verb.WORK
                and argument == CivilizationV2Argument.WORKBENCH
            ):
                allowed[index] = 1
        return LearningTaskDefinition(task, 0, 60, allowed)
    if task == "survival":
        verbs = {
            CivilizationV2Verb.NOOP,
            CivilizationV2Verb.MOVE,
            CivilizationV2Verb.INTERACT,
            CivilizationV2Verb.EAT,
            CivilizationV2Verb.REST,
            CivilizationV2Verb.DEFEND,
        }
        for index, triple in enumerate(V2_MEANINGFUL_ACTIONS):
            verb = CivilizationV2Verb(triple[0])
            argument = CivilizationV2Argument(triple[1])
            if verb in verbs or (
                verb in {CivilizationV2Verb.DEPOSIT, CivilizationV2Verb.WITHDRAW}
                and argument == CivilizationV2Argument.FOOD
            ) or (
                verb == CivilizationV2Verb.USE
                and argument == CivilizationV2Argument.SHELTER
            ):
                allowed[index] = 1
        return LearningTaskDefinition(task, 180, 300, allowed)
    raise ValueError(f"Unknown Stage 7C learning task: {task!r}.")


class DiagnosticCivilizationV2Env(VoyagerCivilizationV2Env):
    """Apply a controlled reset preset without changing the public v2 environment."""

    def __init__(self, task: LearningTask) -> None:
        super().__init__(reward_mode="none")
        self.task_definition = learning_task_definition(task)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        super().reset(seed=seed, options=options)
        state = self.world.state
        assert state is not None
        state.creatures.clear()
        state.step_count = self.task_definition.start_tick
        if self.task_definition.task == "construction":
            for item, quantity in WORKBENCH_REQUIREMENTS.items():
                state.camp.stockpile[item] += quantity
            state.ledger.append(
                {
                    "id": f"ledger-{len(state.ledger):08d}",
                    "tick": state.step_count,
                    "event": "diagnostic_initial_sources",
                    "category": "source",
                    "actors": [],
                    "source": "stage7c_construction_test",
                    "target": "camp",
                    "item": None,
                    "tool": None,
                    "quantity": sum(WORKBENCH_REQUIREMENTS.values()),
                    "lot_id": None,
                    "balance": dict(WORKBENCH_REQUIREMENTS),
                    "details": {"training_only": True},
                }
            )
        elif self.task_definition.task == "return_to_camp":
            positions = (
                (24, 25),
                (30, 31),
                (24, 37),
                (18, 31),
                (27, 28),
                (27, 34),
                (21, 34),
                (21, 28),
                (29, 30),
                (19, 32),
            )
            for index, agent_id in enumerate(self.possible_agents):
                agent = state.agents[agent_id]
                agent.x, agent.y = positions[index]
                item = "wood" if index < 5 else "stone"
                agent.inventory[item] = 2
            diagnostic_sources = {"wood": 10, "stone": 10}
            state.ledger.append(
                {
                    "id": f"ledger-{len(state.ledger):08d}",
                    "tick": state.step_count,
                    "event": "diagnostic_initial_sources",
                    "category": "source",
                    "actors": [],
                    "source": "stage7c_return_to_camp_test",
                    "target": "agents",
                    "item": None,
                    "tool": None,
                    "quantity": sum(diagnostic_sources.values()),
                    "lot_id": None,
                    "balance": diagnostic_sources,
                    "details": {"training_only": True},
                }
            )
        elif self.task_definition.task == "survival":
            shelter = state.structures["shelter"]
            shelter.labor = shelter.required_labor
            shelter.condition = 100
            shelter.capacity = 6
            state.camp.shelter_capacity = 6

        self._clear_step_caches(state.step_count)
        self.agents = self.world.alive_agents()
        observations = {
            agent_id: self._observation(agent_id) for agent_id in self.agents
        }
        infos = {
            agent_id: self._info(agent_id, "diagnostic_reset")
            for agent_id in self.agents
        }
        return observations, infos

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
        acting_agents = tuple(self.agents)
        observations, rewards, terminations, truncations, infos = super().step(
            actions
        )
        state = self.world.state
        assert state is not None
        if state.step_count >= self.task_definition.end_tick:
            truncations = {agent_id: True for agent_id in acting_agents}
            self.agents = []
            observations = {}
        return observations, rewards, terminations, truncations, infos

    def action_mask(self, agent_id: str) -> np.ndarray:
        public_mask = super().action_mask(agent_id)
        return np.bitwise_and(public_mask, self.task_definition.allowed_actions)

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
        assert state is not None
        info.update(
            {
                "learning_task": self.task_definition.task,
                "learning_task_step": (
                    state.step_count - self.task_definition.start_tick
                ),
                "learning_task_horizon": self.task_definition.horizon,
            }
        )
        return info


class CivilizationLearningTaskWrapper(
    ParallelEnv[str, dict[str, np.ndarray], int]
):
    """Replace v3 rewards with a bounded reward for one isolated capability."""

    metadata = CivilizationV2FlattenedActionWrapper.metadata

    def __init__(self, task: LearningTask) -> None:
        self.task_definition = learning_task_definition(task)
        diagnostic = DiagnosticCivilizationV2Env(task)
        self.env = CivilizationProbeRewardWrapper(
            CivilizationV2FlattenedActionWrapper(diagnostic),
            reward_version=PROBE_REWARD_VERSION,
        )
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.observation_spaces = dict(self.env.observation_spaces)
        self.action_spaces = dict(self.env.action_spaces)
        self._ledger_cursor = 0
        self._milestones: set[str] = set()
        self._gather_credit: dict[str, float] = {}
        self._camp_high_water: dict[str, float] = {}

    @property
    def world(self) -> Any:
        return self.env.world

    @property
    def performance_seconds(self) -> dict[str, float]:
        return self.env.performance_seconds

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        state = self.world.state
        assert state is not None
        self._ledger_cursor = len(state.ledger)
        self._milestones.clear()
        self._gather_credit.clear()
        self._camp_high_water.clear()
        self._gather_credit.update(
            {item: 0.0 for item in WORKBENCH_REQUIREMENTS}
        )
        self._camp_high_water.update(
            {item: 0.0 for item in WORKBENCH_REQUIREMENTS}
        )
        for info in infos.values():
            info["reward_version"] = LEARNING_TASK_CONTRACTS[
                self.task_definition.task
            ]
        return observations, infos

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
        acting_agents = tuple(self.env.agents)
        state = self.world.state
        assert state is not None
        before_progress = state.structures["workbench"].progress
        observations, _source_rewards, terminations, truncations, infos = (
            self.env.step(actions)
        )
        self.agents = list(self.env.agents)
        state = self.world.state
        assert state is not None
        new_ledger = state.ledger[self._ledger_cursor :]
        self._ledger_cursor = len(state.ledger)
        shared, individual = self._reward_components(
            acting_agents=acting_agents,
            new_ledger=new_ledger,
            before_progress=before_progress,
            infos=infos,
        )
        shared_reward = float(sum(shared.values()))
        rewards: dict[str, float] = {}
        for agent_id in acting_agents:
            agent_components = individual.get(agent_id, {})
            individual_reward = float(sum(agent_components.values()))
            rewards[agent_id] = shared_reward + individual_reward
            info = infos[agent_id]
            info["source_probe_reward_components"] = dict(
                info.get("reward_components", {})
            )
            info["shared_reward_components"] = dict(shared)
            info["individual_reward_components"] = dict(agent_components)
            info["reward_components"] = {
                **{f"shared_{key}": value for key, value in shared.items()},
                **{
                    f"individual_{key}": value
                    for key, value in agent_components.items()
                },
            }
            info["shared_reward"] = shared_reward
            info["individual_reward"] = individual_reward
            info["reward_version"] = LEARNING_TASK_CONTRACTS[
                self.task_definition.task
            ]
            info["learning_task_success"] = self.success()
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> Any:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Any:
        return self.action_spaces[agent]

    def state(self) -> dict[str, object]:
        return self.env.state()

    def close(self) -> None:
        self.env.close()

    def success(self) -> bool:
        state = self.world.state
        assert state is not None
        if self.task_definition.task == "delivery":
            return all(
                self._camp_high_water[item] >= requirement
                for item, requirement in WORKBENCH_REQUIREMENTS.items()
            )
        if self.task_definition.task == "gather_wood":
            return self._gather_credit["wood"] >= 6
        if self.task_definition.task == "gather_stone":
            return self._gather_credit["stone"] >= 2
        if self.task_definition.task == "return_to_camp":
            return all(
                self._camp_high_water[item] >= requirement
                for item, requirement in WORKBENCH_REQUIREMENTS.items()
            )
        if self.task_definition.task == "construction":
            return state.structures["workbench"].complete
        return (
            state.step_count >= self.task_definition.end_tick
            and sum(
                agent.life_state == "active" for agent in state.agents.values()
            )
            >= 6
        )

    def score(self) -> float:
        state = self.world.state
        assert state is not None
        if self.task_definition.task == "delivery":
            fractions = [
                min(1.0, self._camp_high_water[item] / requirement)
                for item, requirement in WORKBENCH_REQUIREMENTS.items()
            ]
            return float(np.mean(fractions))
        if self.task_definition.task == "gather_wood":
            return min(1.0, self._gather_credit["wood"] / 6)
        if self.task_definition.task == "gather_stone":
            return min(1.0, self._gather_credit["stone"] / 2)
        if self.task_definition.task == "return_to_camp":
            return float(
                np.mean(
                    [
                        min(1.0, self._camp_high_water[item] / requirement)
                        for item, requirement in WORKBENCH_REQUIREMENTS.items()
                    ]
                )
            )
        if self.task_definition.task == "construction":
            return state.structures["workbench"].progress
        return sum(
            agent.life_state == "active" for agent in state.agents.values()
        ) / len(state.agents)

    def _reward_components(
        self,
        *,
        acting_agents: tuple[str, ...],
        new_ledger: list[dict[str, object]],
        before_progress: float,
        infos: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        state = self.world.state
        assert state is not None
        shared: dict[str, float] = {}
        individual: dict[str, dict[str, float]] = {
            agent_id: {} for agent_id in acting_agents
        }
        for agent_id in acting_agents:
            if bool(infos[agent_id].get("invalid_action", False)):
                individual[agent_id]["invalid"] = -0.02

        if self.task_definition.task in {"gather_wood", "gather_stone"}:
            item = (
                "wood"
                if self.task_definition.task == "gather_wood"
                else "stone"
            )
            requirement = 6 if item == "wood" else 2
            gathered = self._actor_quantities(new_ledger, "gather")
            self._credit_capped_units(
                individual,
                gathered.get(item, {}),
                cap=requirement,
                counter=self._gather_credit,
                item=item,
                component=f"gather_{item}",
                value_per_unit=0.10,
            )
            if self.success():
                self._award_once(shared, f"gather_{item}_complete", 2.0)
            return shared, individual

        if self.task_definition.task == "return_to_camp":
            deposited = self._actor_quantities(new_ledger, "deposit")
            for item, requirement in WORKBENCH_REQUIREMENTS.items():
                previous = self._camp_high_water[item]
                current = min(requirement, state.camp.stockpile.get(item, 0))
                self._camp_high_water[item] = max(previous, current)
                self._distribute_credit(
                    individual,
                    deposited.get(item, {}),
                    self._camp_high_water[item] - previous,
                    component=f"deliver_{item}",
                    value_per_unit=0.20,
                )
            if self.success():
                self._award_once(shared, "return_to_camp_complete", 2.0)
            return shared, individual

        if self.task_definition.task == "delivery":
            gathered = self._actor_quantities(new_ledger, "gather")
            deposited = self._actor_quantities(new_ledger, "deposit")
            for item, requirement in WORKBENCH_REQUIREMENTS.items():
                self._credit_capped_units(
                    individual,
                    gathered.get(item, {}),
                    cap=requirement,
                    counter=self._gather_credit,
                    item=item,
                    component=f"gather_{item}",
                    value_per_unit=0.05,
                )
                previous = self._camp_high_water[item]
                current = min(requirement, state.camp.stockpile.get(item, 0))
                self._camp_high_water[item] = max(previous, current)
                self._distribute_credit(
                    individual,
                    deposited.get(item, {}),
                    self._camp_high_water[item] - previous,
                    component=f"deliver_{item}",
                    value_per_unit=0.20,
                )
            if self.success():
                self._award_once(shared, "delivery_complete", 2.0)
            return shared, individual

        if self.task_definition.task == "construction":
            progress = state.structures["workbench"].progress
            progress_delta = max(0.0, progress - before_progress)
            if progress_delta:
                shared["workbench_progress"] = 2.0 * progress_delta
            for entry in new_ledger:
                if (
                    str(entry.get("event", "")) != "construction_labor"
                    or str(entry.get("target", "")) != "workbench"
                ):
                    continue
                actors = entry.get("actors", [])
                if not isinstance(actors, list):
                    continue
                for actor in actors:
                    actor_id = str(actor)
                    if actor_id in individual:
                        individual[actor_id]["valid_work"] = 0.02
            if self.success():
                self._award_once(shared, "workbench_complete", 2.0)
            return shared, individual

        active_fraction = sum(
            agent.life_state == "active" for agent in state.agents.values()
        ) / len(state.agents)
        shared["active_population"] = 0.001 * active_fraction
        within_day = state.step_count % 300
        if 200 <= within_day < 300:
            sheltered_fraction = sum(
                agent.life_state == "active" and agent.sheltered
                for agent in state.agents.values()
            ) / len(state.agents)
            shared["night_shelter"] = 0.005 * sheltered_fraction
        shared["downed"] = -0.10 * sum(
            event.get("type") == "agent_downed" for event in state.events
        )
        shared["death"] = -0.50 * sum(
            event.get("type") == "agent_died" for event in state.events
        )
        if state.step_count >= self.task_definition.end_tick:
            self._award_once(
                shared,
                "first_night_complete",
                2.0 * active_fraction,
            )
        return shared, individual

    @staticmethod
    def _actor_quantities(
        entries: list[dict[str, object]],
        event_name: str,
    ) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}
        for entry in entries:
            if str(entry.get("event", "")) != event_name:
                continue
            item = str(entry.get("item", ""))
            quantity = entry.get("quantity", 0)
            actors = entry.get("actors", [])
            if not isinstance(quantity, int) or not isinstance(actors, list):
                continue
            if quantity <= 0 or not actors:
                continue
            each = quantity / len(actors)
            item_values = values.setdefault(item, {})
            for actor in actors:
                actor_id = str(actor)
                item_values[actor_id] = item_values.get(actor_id, 0.0) + each
        return values

    def _credit_capped_units(
        self,
        components: dict[str, dict[str, float]],
        actor_quantities: dict[str, float],
        *,
        cap: int,
        counter: dict[str, float],
        item: str,
        component: str,
        value_per_unit: float,
    ) -> None:
        remaining = max(0.0, cap - counter.get(item, 0.0))
        credit = min(remaining, sum(actor_quantities.values()))
        self._distribute_credit(
            components,
            actor_quantities,
            credit,
            component=component,
            value_per_unit=value_per_unit,
        )
        counter[item] = counter.get(item, 0.0) + credit

    @staticmethod
    def _distribute_credit(
        components: dict[str, dict[str, float]],
        actor_quantities: dict[str, float],
        credit: float,
        *,
        component: str,
        value_per_unit: float,
    ) -> None:
        total = sum(actor_quantities.values())
        if credit <= 0 or total <= 0:
            return
        for actor, quantity in actor_quantities.items():
            if actor not in components:
                continue
            components[actor][component] = (
                components[actor].get(component, 0.0)
                + value_per_unit * credit * quantity / total
            )

    def _award_once(
        self,
        components: dict[str, float],
        name: str,
        value: float,
    ) -> None:
        if name in self._milestones:
            return
        self._milestones.add(name)
        components[name] = value
