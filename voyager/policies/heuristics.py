"""Scripted baseline policies for Voyager."""

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from voyager.policies.base import Info, Observation, Policy
from voyager.sim.constants import ACTION_COUNT, Action, Resource

MOVE_ACTIONS = (
    Action.MOVE_RIGHT,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_UP,
)


@dataclass(slots=True)
class RandomPolicy(Policy):
    """Uniform random action baseline."""

    seed: int | None = None
    action_count: int = ACTION_COUNT
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        _ = agent_id, observation, info
        return int(self.rng.integers(0, self.action_count))


class GreedySurvivalPolicy(Policy):
    """Individual survival baseline with no explicit cooperation objective."""

    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        inventory = _inventory(info)
        hunger = _hunger(info)
        energy = _energy(info)

        if hunger >= 55.0 and inventory["food"] > 0:
            return int(Action.EAT)
        if _at_camp(info) and hunger >= 65.0 and _camp_stockpile(info)["food"] > 0:
            return int(Action.WITHDRAW_FOOD)
        if energy <= 25.0:
            return int(Action.REST)
        if _center_resource(observation) != Resource.NONE:
            return int(Action.GATHER)
        if hunger >= 55.0:
            action = _move_toward_visible_resource(observation, Resource.FOOD)
            if action is not None:
                return action
            if _camp_stockpile(info)["food"] > 0:
                return _move_toward_camp(info)

        action = _move_toward_visible_resource(observation, None)
        if action is not None:
            return action
        return _explore(agent_id, info)


class CooperativePolicy(Policy):
    """Shared-camp baseline that contributes surplus and prioritizes shelter."""

    def act(self, agent_id: str, observation: Observation, info: Info) -> int:
        inventory = _inventory(info)
        hunger = _hunger(info)
        energy = _energy(info)
        role = str(info.get("role", "forager"))
        at_camp = _at_camp(info)
        camp = _camp_stockpile(info)
        shelter_progress = _shelter_progress(info)

        if hunger >= 70.0 and inventory["food"] > 0:
            return int(Action.EAT)
        if at_camp and hunger >= 60.0 and camp["food"] > 0:
            return int(Action.WITHDRAW_FOOD)
        if energy <= 25.0:
            return int(Action.REST)

        if at_camp:
            if (
                role == "builder"
                and shelter_progress < 1.0
                and (inventory["wood"] > 0 or inventory["stone"] > 0)
            ):
                return int(Action.BUILD_SHELTER)
            if inventory["food"] > 1 and hunger < 55.0:
                return int(Action.DEPOSIT_FOOD)
            if inventory["wood"] > 0:
                return int(Action.DEPOSIT_WOOD)
            if inventory["stone"] > 0:
                return int(Action.DEPOSIT_STONE)

        if _should_return_to_camp(inventory):
            return _move_toward_camp(info)

        center_resource = _center_resource(observation)
        if center_resource != Resource.NONE:
            return int(Action.GATHER)

        target_resource = _role_resource(role, hunger)
        action = _move_toward_visible_resource(observation, target_resource)
        if action is not None:
            return action
        action = _move_toward_visible_resource(observation, None)
        if action is not None:
            return action
        return _explore(agent_id, info)


def _inventory(info: Info) -> dict[str, int]:
    raw = info.get("inventory", {})
    if not isinstance(raw, dict):
        return {"food": 0, "wood": 0, "stone": 0}
    return {
        "food": int(raw.get("food", 0)),
        "wood": int(raw.get("wood", 0)),
        "stone": int(raw.get("stone", 0)),
    }


def _camp_stockpile(info: Info) -> dict[str, int]:
    camp = info.get("camp", {})
    if not isinstance(camp, dict):
        return {"food": 0, "wood": 0, "stone": 0}
    stockpile = camp.get("stockpile", {})
    if not isinstance(stockpile, dict):
        return {"food": 0, "wood": 0, "stone": 0}
    return {
        "food": int(stockpile.get("food", 0)),
        "wood": int(stockpile.get("wood", 0)),
        "stone": int(stockpile.get("stone", 0)),
    }


def _shelter_progress(info: Info) -> float:
    camp = info.get("camp", {})
    if not isinstance(camp, dict):
        return 0.0
    return _real_value(camp.get("shelter_progress", 0.0), default=0.0)


def _hunger(info: Info) -> float:
    return _real_value(info.get("hunger", 0.0), default=0.0)


def _energy(info: Info) -> float:
    return _real_value(info.get("energy", 100.0), default=100.0)


def _at_camp(info: Info) -> bool:
    position = info.get("position")
    camp = info.get("camp", {})
    if not isinstance(camp, dict):
        return False
    return position == camp.get("position")


def _move_toward_camp(info: Info) -> int:
    position = info.get("position")
    camp = info.get("camp", {})
    if not isinstance(position, tuple) or not isinstance(camp, dict):
        return _explore("agent_0", info)
    camp_position = camp.get("position")
    if not isinstance(camp_position, tuple):
        return _explore("agent_0", info)
    return _move_toward(position, camp_position)


def _move_toward(start: tuple[int, int], target: tuple[int, int]) -> int:
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return int(Action.MOVE_RIGHT if dx > 0 else Action.MOVE_LEFT)
    if dy != 0:
        return int(Action.MOVE_DOWN if dy > 0 else Action.MOVE_UP)
    return int(Action.NOOP)


def _center_resource(observation: Observation) -> Resource:
    local_view = observation["local_view"]
    center = local_view.shape[0] // 2
    return Resource(int(local_view[center, center, 1]))


def _move_toward_visible_resource(
    observation: Observation,
    resource: Resource | None,
) -> int | None:
    local_view = observation["local_view"]
    center = local_view.shape[0] // 2
    candidates: list[tuple[int, int, int]] = []
    for row in range(local_view.shape[0]):
        for col in range(local_view.shape[1]):
            visible_resource = Resource(int(local_view[row, col, 1]))
            if visible_resource == Resource.NONE:
                continue
            if resource is not None and visible_resource != resource:
                continue
            distance = abs(row - center) + abs(col - center)
            candidates.append((distance, row, col))

    if not candidates:
        return None

    _distance, row, col = min(candidates)
    return _move_toward((center, center), (col, row))


def _role_resource(role: str, hunger: float) -> Resource:
    if hunger >= 55.0:
        return Resource.FOOD
    if role == "woodcutter" or role == "builder":
        return Resource.WOOD
    return Resource.FOOD


def _should_return_to_camp(inventory: dict[str, int]) -> bool:
    return inventory["food"] > 1 or inventory["wood"] > 0 or inventory["stone"] > 0


def _explore(agent_id: str, info: Info) -> int:
    try:
        agent_index = int(agent_id.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        agent_index = 0
    step = int(_real_value(info.get("step", 0), default=0.0))
    return int(MOVE_ACTIONS[(step + agent_index) % len(MOVE_ACTIONS)])


def _real_value(value: object, default: float) -> float:
    if isinstance(value, Real):
        return float(value)
    return default
