"""Acceptance coverage for the Stage 7B deterministic Civilization core."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np

from voyager.envs.civilization_v2 import (
    CivilizationV2FlattenedActionWrapper,
    VoyagerCivilizationV2Env,
)
from voyager.replay.loader import ReplayLoader
from voyager.replay.serialization import sha256_value
from voyager.sim.constants import Resource
from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    flatten_v2_action,
    unflatten_v2_action,
)
from voyager.sim.registries_v2 import (
    CivilizationV2Argument as Argument,
)
from voyager.sim.registries_v2 import (
    CivilizationV2Verb as Verb,
)
from voyager.sim.state import FoodLot


def action(verb: Verb, argument: Argument = Argument.NONE, target: int = 0) -> dict[str, int]:
    return {"verb": int(verb), "argument": int(argument), "target": target}


def reset() -> VoyagerCivilizationV2Env:
    env = VoyagerCivilizationV2Env()
    env.reset(seed=7)
    return env


def relocate_resource(env: VoyagerCivilizationV2Env, agent_id: str, item: str, quantity: int) -> None:
    state = env.world.state
    assert state is not None
    resource = {"food": Resource.FOOD, "wood": Resource.WOOD, "stone": Resource.STONE}[item]
    remaining = quantity
    for y, x in np.argwhere(state.resource_ids == resource):
        taken = min(remaining, int(state.resource_quantities[y, x]))
        state.resource_quantities[y, x] -= taken
        if state.resource_quantities[y, x] == 0:
            state.resource_ids[y, x] = Resource.NONE
        remaining -= taken
        if remaining == 0:
            break
    assert remaining == 0
    state.agents[agent_id].inventory[item] = quantity


def test_v2_is_registered_and_flat_registry_round_trips() -> None:
    env = gym.make("VoyagerCivilization-v2")
    observations, _infos = env.reset(seed=7)
    assert observations["agent_0"]["action_mask"].shape == (V2_FLAT_ACTION_COUNT,)
    assert observations["agent_0"] in env.observation_space("agent_0")
    for index in range(V2_FLAT_ACTION_COUNT):
        assert flatten_v2_action(*unflatten_v2_action(index)) == index
    flattened = CivilizationV2FlattenedActionWrapper()
    flat_observations, _ = flattened.reset(seed=7)
    assert np.array_equal(
        observations["agent_0"]["action_mask"],
        flat_observations["agent_0"]["action_mask"],
    )


def test_committed_replay_22_deeply_reconstructs() -> None:
    replay = Path(__file__).parents[1] / "runs/replays/civilization_deterministic_core_v1"
    loader = ReplayLoader(replay)
    assert loader.manifest.versions.replay == "stage7_replay_2.2.0"
    assert loader.manifest.terminal_summary["invalid_scripted_actions"] == 0
    assert loader.manifest.terminal_summary["conservation"] == {}
    assert loader.manifest.terminal_summary["downings"] >= 1
    assert loader.manifest.terminal_summary["revivals"] >= 1
    assert loader.validate(deep=True)["checked_ticks"] == 601


def test_submission_order_is_byte_identical() -> None:
    def run(reverse: bool) -> tuple[str, list[dict[str, object]]]:
        env = reset()
        noop = action(Verb.NOOP)
        for _tick in range(205):
            agents = list(env.agents)
            if reverse:
                agents.reverse()
            env.step({agent_id: deepcopy(noop) for agent_id in agents})
        state = env.world.state
        assert state is not None
        return sha256_value(env.global_state()), state.events

    forward_hash, forward_events = run(False)
    reverse_hash, reverse_events = run(True)
    assert forward_hash == reverse_hash
    assert forward_events == reverse_events


def test_movement_conflicts_swaps_cycles_and_blocked_chains() -> None:
    env = reset()
    state = env.world.state
    assert state is not None
    positions = [(23, 23), (24, 23), (24, 24), (23, 24)]
    for index, position in enumerate(positions):
        state.agents[f"agent_{index}"].x, state.agents[f"agent_{index}"].y = position

    env.step(
        {
            "agent_0": action(Verb.MOVE, Argument.EAST),
            "agent_1": action(Verb.MOVE, Argument.WEST),
        }
    )
    assert (state.agents["agent_0"].x, state.agents["agent_0"].y) == positions[1]
    assert (state.agents["agent_1"].x, state.agents["agent_1"].y) == positions[0]

    env.step(
        {
            "agent_0": action(Verb.MOVE, Argument.SOUTH),
            "agent_1": action(Verb.MOVE, Argument.EAST),
            "agent_2": action(Verb.MOVE, Argument.WEST),
            "agent_3": action(Verb.MOVE, Argument.NORTH),
        }
    )
    assert (state.agents["agent_0"].x, state.agents["agent_0"].y) == positions[2]
    assert (state.agents["agent_1"].x, state.agents["agent_1"].y) == positions[1]

    before = (state.agents["agent_0"].x, state.agents["agent_0"].y)
    env.step(
        {
            "agent_0": action(Verb.MOVE, Argument.NORTH),
            "agent_2": action(Verb.MOVE, Argument.NORTH),
        }
    )
    assert (state.agents["agent_0"].x, state.agents["agent_0"].y) == before


def test_tool_crafting_transfer_storage_charge_and_conservation() -> None:
    env = reset()
    state = env.world.state
    assert state is not None
    workbench = state.structures["workbench"]
    workbench.labor = workbench.required_labor
    giver = state.agents["agent_0"]
    receiver = state.agents["agent_1"]
    giver.x, giver.y = workbench.x, workbench.y
    receiver.x, receiver.y = workbench.x + 1, workbench.y
    relocate_resource(env, "agent_0", "wood", 2)
    relocate_resource(env, "agent_0", "stone", 1)

    env.step({"agent_0": action(Verb.CRAFT, Argument.AXE_RECIPE)})
    assert "axe" in giver.tools
    target = env.world.v2_entity_slots("agent_0").index("agent:agent_1") + 1
    env.step({"agent_0": action(Verb.GIVE, Argument.AXE, target)})
    assert "axe" not in giver.tools and "axe" in receiver.tools

    receiver.x, receiver.y = state.camp.x, state.camp.y
    env.step({"agent_1": action(Verb.DEPOSIT, Argument.AXE)})
    assert len(state.camp.tool_stockpile["axe"]) == 1
    env.step({"agent_1": action(Verb.WITHDRAW, Argument.AXE)})
    assert "axe" in receiver.tools
    assert env.world.reconcile_v2_ledger() == {}


def test_food_expiry_boundary_and_first_expiry_selection() -> None:
    env = reset()
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    relocate_resource(env, "agent_0", "food", 2)
    agent.inventory["food"] = 0
    agent.food_lots = [
        FoodLot("later", "berries", 1, "map", "tile-a", 0, 3, "fresh"),
        FoodLot("first", "berries", 1, "map", "tile-b", 0, 2, "fresh"),
    ]
    env.step({"agent_0": action(Verb.EAT, Argument.FOOD)})
    assert [lot.id for lot in agent.food_lots] == ["later"]
    env.step({})
    assert [lot.id for lot in agent.food_lots] == ["later"]
    env.step({})
    assert agent.food_lots == []
    assert state.spoiled_resources["food"] == 1
    assert env.world.reconcile_v2_ledger() == {}


def test_oversubscribed_withdrawal_fails_as_one_symmetric_group() -> None:
    env = reset()
    state = env.world.state
    assert state is not None
    ration = state.camp.food_lots[0]
    ration.quantity = 1
    state.agents["agent_9"].food_lots.append(
        FoodLot(
            "reserved-rations",
            "wreck_ration",
            9,
            "wreck",
            "initial_wreck",
            0,
            None,
            "ration",
        )
    )
    for agent_id in ("agent_0", "agent_1"):
        state.agents[agent_id].x, state.agents[agent_id].y = state.camp.x, state.camp.y
    _observations, _rewards, _terms, _truncs, infos = env.step(
        {
            "agent_0": action(Verb.WITHDRAW, Argument.FOOD),
            "agent_1": action(Verb.WITHDRAW, Argument.FOOD),
        }
    )
    assert infos["agent_0"]["invalid_action"]
    assert infos["agent_1"]["invalid_action"]
    assert state.camp.food_lots[0].quantity == 1
    assert env.world.reconcile_v2_ledger() == {}


def test_joint_repair_consumes_one_material_and_restores_threshold() -> None:
    env = reset()
    state = env.world.state
    assert state is not None
    shelter = state.structures["shelter"]
    shelter.labor = shelter.required_labor
    shelter.condition = 40
    shelter.capacity = 3
    relocate_resource(env, "agent_0", "wood", 1)
    state.agents["agent_0"].inventory["wood"] = 0
    state.camp.stockpile["wood"] += 1
    repairers = ("agent_0", "agent_1", "agent_2")
    for agent_id, position in zip(
        repairers,
        ((shelter.x - 1, shelter.y), (shelter.x + 1, shelter.y), (shelter.x, shelter.y - 1)),
        strict=True,
    ):
        state.agents[agent_id].x, state.agents[agent_id].y = position
    repair_actions = {}
    for agent_id in repairers:
        slot = env.world.v2_entity_slots(agent_id).index("structure:shelter") + 1
        repair_actions[agent_id] = action(Verb.REPAIR, target=slot)
    env.step(repair_actions)
    assert shelter.condition == 40 and shelter.repair_material_reserved
    env.step(repair_actions)
    assert shelter.condition == 60 and shelter.capacity == 6
    assert state.camp.stockpile["wood"] == 0
    assert env.world.reconcile_v2_ledger() == {}


def test_downed_agent_is_restricted_and_jointly_revived() -> None:
    env = reset()
    state = env.world.state
    assert state is not None
    target = state.agents["agent_0"]
    target.health = 0
    target.x, target.y = 24, 24
    for agent_id, position in zip(
        ("agent_1", "agent_2", "agent_3"),
        ((23, 24), (25, 24), (24, 23)),
        strict=True,
    ):
        state.agents[agent_id].x, state.agents[agent_id].y = position
    state.agents["agent_1"].food_lots.append(state.camp.food_lots.pop())
    env.step({})
    assert target.life_state == "downed" and target.downed_ticks == 20
    assert env.action_mask("agent_0").sum() == 1

    revive_actions = {}
    for agent_id in ("agent_1", "agent_2", "agent_3"):
        slot = env.world.v2_entity_slots(agent_id).index("agent:agent_0") + 1
        revive_actions[agent_id] = action(Verb.REVIVE, target=slot)
    env.step(revive_actions)
    assert target.life_state == "active"
    assert (target.health, target.hunger, target.energy) == (30, 50, 20)
    assert any(entry["event"] == "revival_labor" for entry in state.ledger)
    assert env.world.reconcile_v2_ledger() == {}
