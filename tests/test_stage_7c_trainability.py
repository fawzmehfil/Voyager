"""Stage 7C trainer, outcome evaluator, and hot-path acceptance tests."""

from __future__ import annotations

import numpy as np
import pytest

from voyager.envs.civilization_v2 import VoyagerCivilizationV2Env
from voyager.sim.constants import Resource
from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    CivilizationV2Argument,
    CivilizationV2Verb,
    flatten_v2_action,
)
from voyager.training.civilization_evaluation import (
    CivilizationEpisodeResult,
    compare_against_random,
    pilot_continuation,
    run_civilization_episode,
    summarize_civilization_results,
)
from voyager.training.civilization_probe import (
    PROBE_REWARD_V2,
    CivilizationProbeRewardWrapper,
)
from voyager.training.environments import (
    CIVILIZATION_PROBE_REWARD_CONTRACT,
    CIVILIZATION_PROBE_V1_REWARD_CONTRACT,
    CIVILIZATION_PROBE_V2_REWARD_CONTRACT,
    CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
    CIVILIZATION_V2_TRAINING_ENVIRONMENT,
    make_training_environment,
)
from voyager.training.obs import flat_observation_size, flatten_observation
from voyager.training.ppo import PPOConfig, PPOTrainer


def test_v3_training_adapter_exposes_navigation_actor_contract() -> None:
    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_REWARD_CONTRACT,
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    observations, infos = env.reset(seed=0)

    observation = observations["agent_0"]
    encoded = flatten_observation(
        observation,
        training_environment.observation_encoder,
    )
    assert encoded.shape == (604,)
    assert encoded.shape == (
        flat_observation_size(
            env.observation_space("agent_0"),
            training_environment.observation_encoder,
        ),
    )
    assert env.action_space("agent_0").n == V2_FLAT_ACTION_COUNT
    assert np.asarray(infos["agent_0"]["action_mask"]).shape == (
        V2_FLAT_ACTION_COUNT,
    )
    assert observation["agent_identity"].shape == (10,)
    assert observation["agent_identity"].sum() == 1
    assert observation["agent_identity"][0] == 1
    assert observation["camp_bearing"].shape == (3,)
    assert "team_objective" not in observation
    assert np.allclose(observation["camp_bearing"], 0.0)
    assert np.all(np.isfinite(encoded))


def test_v4_team_objective_is_shared_bounded_and_tracks_reward_state() -> None:
    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_V4_REWARD_CONTRACT,
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    observations, _infos = env.reset(seed=0)

    assert training_environment.observation_encoder == (
        "civilization_v4_team_objective_flat_v4"
    )
    assert flatten_observation(
        observations["agent_0"], training_environment.observation_encoder
    ).shape == (610,)
    expected_initial = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    for observation in observations.values():
        np.testing.assert_allclose(observation["team_objective"], expected_initial)

    state = env.world.state
    assert state is not None
    wood_y, wood_x = np.argwhere(
        (state.resource_ids == Resource.WOOD) & (state.resource_quantities > 0)
    )[0]
    stone_y, stone_x = np.argwhere(
        (state.resource_ids == Resource.STONE) & (state.resource_quantities > 0)
    )[0]
    state.agents["agent_0"].x, state.agents["agent_0"].y = int(wood_x), int(
        wood_y
    )
    state.agents["agent_1"].x, state.agents["agent_1"].y = int(stone_x), int(
        stone_y
    )
    noop = flatten_v2_action(
        CivilizationV2Verb.NOOP, CivilizationV2Argument.NONE, 0
    )
    gather = flatten_v2_action(
        CivilizationV2Verb.INTERACT, CivilizationV2Argument.NONE, 0
    )
    actions = {agent_id: noop for agent_id in env.agents}
    actions.update({"agent_0": gather, "agent_1": gather})
    observations, _rewards, _terminations, _truncations, _infos = env.step(actions)
    expected_gathered = np.array([1 / 6, 1 / 2, 0, 0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(
        observations["agent_0"]["team_objective"], expected_gathered
    )
    np.testing.assert_allclose(
        observations["agent_1"]["team_objective"], expected_gathered
    )

    state = env.world.state
    assert state is not None
    for agent_id in ("agent_0", "agent_1"):
        state.agents[agent_id].x = state.camp.x
        state.agents[agent_id].y = state.camp.y
    actions = {agent_id: noop for agent_id in env.agents}
    actions["agent_0"] = flatten_v2_action(
        CivilizationV2Verb.DEPOSIT, CivilizationV2Argument.WOOD, 0
    )
    actions["agent_1"] = flatten_v2_action(
        CivilizationV2Verb.DEPOSIT, CivilizationV2Argument.STONE, 0
    )
    observations, _rewards, _terminations, _truncations, _infos = env.step(actions)
    expected_delivered = np.array(
        [1 / 6, 1 / 2, 1 / 6, 1 / 2, 0, 1], dtype=np.float32
    )
    np.testing.assert_allclose(
        observations["agent_0"]["team_objective"], expected_delivered
    )
    assert np.all(observations["agent_0"]["team_objective"] >= 0)
    assert np.all(observations["agent_0"]["team_objective"] <= 1)


def test_v2_probe_contract_retains_601_value_identity_observation() -> None:
    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_V2_REWARD_CONTRACT,
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    observations, _infos = env.reset(seed=0)

    assert "agent_identity" in observations["agent_0"]
    assert "camp_bearing" not in observations["agent_0"]
    assert flatten_observation(
        observations["agent_0"],
        training_environment.observation_encoder,
    ).shape == (601,)


def test_v1_probe_contract_retains_591_value_observation() -> None:
    training_environment = make_training_environment(
        environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
        reward_contract=CIVILIZATION_PROBE_V1_REWARD_CONTRACT,
        num_agents=10,
        map_size=48,
        max_steps=600,
        reward_mode="none",
        disabled_reward_components=(),
        mask_role_observation=False,
    )
    env = training_environment.env
    observations, _infos = env.reset(seed=0)

    assert "agent_identity" not in observations["agent_0"]
    assert flatten_observation(
        observations["agent_0"],
        training_environment.observation_encoder,
    ).shape == (591,)


def test_probe_reward_is_shared_and_rewards_each_finite_gathered_unit() -> None:
    env = CivilizationProbeRewardWrapper(reward_version=PROBE_REWARD_V2)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    wood_y, wood_x = np.argwhere(
        (state.resource_ids == Resource.WOOD) & (state.resource_quantities > 0)
    )[0]
    agent.x, agent.y = int(wood_x), int(wood_y)
    gather = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )

    _observations, rewards, _terms, _truncs, infos = env.step({"agent_0": gather})

    assert len(set(rewards.values())) == 1
    assert infos["agent_0"]["reward_components"]["gather_wood"] == pytest.approx(0.10)
    agent.energy = 100
    _observations, _rewards, _terms, _truncs, second_infos = env.step(
        {"agent_0": gather}
    )
    assert second_infos["agent_0"]["reward_components"]["gather_wood"] == pytest.approx(
        0.10
    )


def test_delivery_reward_cannot_be_repeated_by_withdraw_and_redeposit() -> None:
    env = CivilizationProbeRewardWrapper(reward_version=PROBE_REWARD_V2)
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    wood_y, wood_x = np.argwhere(
        (state.resource_ids == Resource.WOOD) & (state.resource_quantities > 0)
    )[0]
    agent.x, agent.y = int(wood_x), int(wood_y)
    gather = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )
    deposit = flatten_v2_action(
        CivilizationV2Verb.DEPOSIT,
        CivilizationV2Argument.WOOD,
        0,
    )
    withdraw = flatten_v2_action(
        CivilizationV2Verb.WITHDRAW,
        CivilizationV2Argument.WOOD,
        0,
    )
    env.step({"agent_0": gather})
    agent.x, agent.y = state.camp.x, state.camp.y

    _obs, _rewards, _terms, _truncs, first_infos = env.step({"agent_0": deposit})
    env.step({"agent_0": withdraw})
    _obs, _rewards, _terms, _truncs, second_infos = env.step({"agent_0": deposit})

    assert first_infos["agent_0"]["reward_components"][
        "first_delivery_wood"
    ] == pytest.approx(0.15)
    assert "first_delivery_wood" not in second_infos["agent_0"]["reward_components"]


def test_v3_camp_bearing_tracks_signed_direction_and_distance() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    state.agents["agent_0"].x = 0
    state.agents["agent_0"].y = 0
    noop = flatten_v2_action(
        CivilizationV2Verb.NOOP,
        CivilizationV2Argument.NONE,
        0,
    )

    observations, _rewards, _terms, _truncs, _infos = env.step(
        {"agent_0": noop}
    )

    assert observations["agent_0"]["camp_bearing"] == pytest.approx(
        np.array([24 / 47, 31 / 47, 55 / 94], dtype=np.float32)
    )


def test_v3_gather_credit_is_globally_capped_at_workbench_requirements() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    gather = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )
    individual_wood = 0.0
    wood_requirement_awards = 0.0

    for _index in range(7):
        wood_y, wood_x = np.argwhere(
            (state.resource_ids == Resource.WOOD)
            & (state.resource_quantities > 0)
        )[0]
        agent.x, agent.y = int(wood_x), int(wood_y)
        agent.energy = 100
        _obs, _rewards, _terms, _truncs, infos = env.step(
            {"agent_0": gather}
        )
        individual_wood += infos["agent_0"][
            "individual_reward_components"
        ].get("gather_wood", 0.0)
        wood_requirement_awards += infos["agent_0"][
            "shared_reward_components"
        ].get("gather_wood_requirement", 0.0)

    assert individual_wood == pytest.approx(0.30)
    assert wood_requirement_awards == pytest.approx(0.25)
    assert "gather_wood" not in infos["agent_0"][
        "individual_reward_components"
    ]


def test_v3_same_tick_credit_is_proportional_without_agent_id_priority() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    env._v3_gather_credit["wood"] = 5
    wood_nodes = np.argwhere(
        (state.resource_ids == Resource.WOOD) & (state.resource_quantities > 0)
    )[:2]
    for agent_id, (wood_y, wood_x) in zip(
        ("agent_0", "agent_1"),
        wood_nodes,
        strict=True,
    ):
        state.agents[agent_id].x = int(wood_x)
        state.agents[agent_id].y = int(wood_y)
    gather = flatten_v2_action(
        CivilizationV2Verb.INTERACT,
        CivilizationV2Argument.NONE,
        0,
    )

    _obs, rewards, _terms, _truncs, infos = env.step(
        {"agent_0": gather, "agent_1": gather}
    )

    assert infos["agent_0"]["individual_reward_components"][
        "gather_wood"
    ] == pytest.approx(0.025)
    assert infos["agent_1"]["individual_reward_components"][
        "gather_wood"
    ] == pytest.approx(0.025)
    assert rewards["agent_0"] == pytest.approx(rewards["agent_1"])


def test_v3_delivery_high_water_cannot_be_farmed() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    agent.x, agent.y = state.camp.x, state.camp.y
    agent.inventory["wood"] = 6
    state.ledger.append({"event": "test_source", "balance": {"wood": 6}})
    deposit = flatten_v2_action(
        CivilizationV2Verb.DEPOSIT,
        CivilizationV2Argument.WOOD,
        0,
    )
    withdraw = flatten_v2_action(
        CivilizationV2Verb.WITHDRAW,
        CivilizationV2Argument.WOOD,
        0,
    )
    individual_delivery = 0.0
    camp_requirement_awards = 0.0
    for _index in range(6):
        _obs, _rewards, _terms, _truncs, infos = env.step(
            {"agent_0": deposit}
        )
        individual_delivery += infos["agent_0"][
            "individual_reward_components"
        ].get("deliver_wood", 0.0)
        camp_requirement_awards += infos["agent_0"][
            "shared_reward_components"
        ].get("camp_wood_requirement", 0.0)
    env.step({"agent_0": withdraw})
    _obs, _rewards, _terms, _truncs, repeated = env.step(
        {"agent_0": deposit}
    )

    assert individual_delivery == pytest.approx(0.60)
    assert camp_requirement_awards == pytest.approx(0.50)
    assert "deliver_wood" not in repeated["agent_0"][
        "individual_reward_components"
    ]


def test_v3_invalid_penalties_are_individual() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    state.agents["agent_0"].x, state.agents["agent_0"].y = 10, 10
    state.agents["agent_1"].x, state.agents["agent_1"].y = 12, 10
    east = flatten_v2_action(
        CivilizationV2Verb.MOVE,
        CivilizationV2Argument.EAST,
        0,
    )
    west = flatten_v2_action(
        CivilizationV2Verb.MOVE,
        CivilizationV2Argument.WEST,
        0,
    )

    _obs, rewards, _terms, _truncs, infos = env.step(
        {"agent_0": east, "agent_1": west}
    )

    assert infos["agent_0"]["individual_reward_components"]["invalid"] == -0.02
    assert infos["agent_1"]["individual_reward_components"]["invalid"] == -0.02
    assert "invalid" not in infos["agent_2"]["individual_reward_components"]
    assert rewards["agent_0"] == pytest.approx(rewards["agent_1"])
    assert rewards["agent_2"] - rewards["agent_0"] == pytest.approx(0.02)
    for agent_id in ("agent_0", "agent_1", "agent_2"):
        assert sum(infos[agent_id]["reward_components"].values()) == pytest.approx(
            rewards[agent_id]
        )


def test_v3_workbench_and_tool_rewards_are_bounded_and_once_only() -> None:
    env = CivilizationProbeRewardWrapper()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    builder = state.agents["agent_2"]
    workbench = state.structures["workbench"]
    builder.x, builder.y = workbench.x, workbench.y
    state.camp.stockpile.update(wood=6, stone=2)
    state.ledger.append(
        {"event": "test_source", "balance": {"wood": 6, "stone": 2}}
    )
    work = flatten_v2_action(
        CivilizationV2Verb.WORK,
        CivilizationV2Argument.WORKBENCH,
        0,
    )
    totals: dict[str, float] = {}
    for _index in range(16):
        _obs, _rewards, _terms, _truncs, infos = env.step({"agent_2": work})
        for name, value in infos["agent_2"]["shared_reward_components"].items():
            totals[name] = totals.get(name, 0.0) + value

    assert workbench.complete
    assert totals["workbench_materials_reserved"] == pytest.approx(1.0)
    assert totals["workbench_progress"] == pytest.approx(2.0)
    assert totals["workbench_complete"] == pytest.approx(2.0)

    crafter = state.agents["agent_0"]
    crafter.x, crafter.y = workbench.x, workbench.y
    crafter.inventory.update(wood=2, stone=1)
    state.ledger.append(
        {"event": "test_source", "balance": {"wood": 2, "stone": 1}}
    )
    craft_axe = flatten_v2_action(
        CivilizationV2Verb.CRAFT,
        CivilizationV2Argument.AXE_RECIPE,
        0,
    )
    _obs, _rewards, _terms, _truncs, first = env.step(
        {"agent_0": craft_axe}
    )
    crafter.x, crafter.y = state.camp.x, state.camp.y
    second_crafter = state.agents["agent_1"]
    second_crafter.x, second_crafter.y = workbench.x, workbench.y
    second_crafter.inventory.update(wood=2, stone=1)
    state.ledger.append(
        {"event": "test_source", "balance": {"wood": 2, "stone": 1}}
    )
    _obs, _rewards, _terms, _truncs, second = env.step(
        {"agent_1": craft_axe}
    )

    assert first["agent_0"]["shared_reward_components"][
        "first_tool_crafted"
    ] == pytest.approx(0.75)
    assert first["agent_0"]["shared_reward_components"][
        "first_axe_crafted"
    ] == pytest.approx(0.25)
    assert "first_tool_crafted" not in second["agent_1"][
        "shared_reward_components"
    ]
    assert "first_axe_crafted" not in second["agent_1"][
        "shared_reward_components"
    ]


def test_work_mask_requires_reservable_materials() -> None:
    env = VoyagerCivilizationV2Env()
    env.reset(seed=0)
    state = env.world.state
    assert state is not None
    agent = state.agents["agent_0"]
    workbench = state.structures["workbench"]
    agent.x, agent.y = workbench.x, workbench.y
    work = flatten_v2_action(
        CivilizationV2Verb.WORK,
        CivilizationV2Argument.WORKBENCH,
        0,
    )

    assert not env.world.v2_action_mask("agent_0")[work]
    state.camp.stockpile.update(wood=6, stone=2)
    assert env.world.v2_action_mask("agent_0")[work]


def test_v2_step_reuses_masks_slots_and_conservation_audits() -> None:
    env = VoyagerCivilizationV2Env()
    mask_calls = 0
    slot_calls = 0
    conservation_calls = 0
    original_mask = env.world.v2_action_mask
    original_slots = env.world.v2_entity_slots
    original_conservation = env.world.reconcile_v2_ledger

    def counted_mask(agent_id: str) -> np.ndarray:
        nonlocal mask_calls
        mask_calls += 1
        return original_mask(agent_id)

    def counted_slots(agent_id: str) -> list[str]:
        nonlocal slot_calls
        slot_calls += 1
        return original_slots(agent_id)

    def counted_conservation() -> dict[str, int]:
        nonlocal conservation_calls
        conservation_calls += 1
        return original_conservation()

    env.world.v2_action_mask = counted_mask  # type: ignore[method-assign]
    env.world.v2_entity_slots = counted_slots  # type: ignore[method-assign]
    env.world.reconcile_v2_ledger = counted_conservation  # type: ignore[method-assign]
    env.reset(seed=0)

    assert mask_calls == 10
    assert slot_calls == 10
    assert conservation_calls == 1


def test_outcome_summary_and_paired_gate_use_independent_metrics() -> None:
    def result(policy: str, seed: int, successes: int) -> CivilizationEpisodeResult:
        flags = [index < successes for index in range(5)]
        return CivilizationEpisodeResult(
            policy=policy,
            seed=seed,
            world_steps=600,
            agent_transitions=6_000,
            shared_return=999.0,
            gathered_workbench_bundle_by_100=flags[0],
            workbench_materials_available_by_300=flags[1],
            workbench_complete=flags[2],
            any_tool_crafted=flags[3],
            majority_active_at_300=flags[4],
            active_at_300=10 if flags[4] else 0,
            final_active=10,
            deaths=0,
            invalid_actions=0,
            submitted_actions=6_000,
        )

    learned = [result("ppo", seed, 5) for seed in range(4)]
    random = [result("random", seed, 0) for seed in range(4)]

    assert summarize_civilization_results(learned)["composite"] == 1.0
    comparison = compare_against_random(
        learned,
        random,
        bootstrap_samples=100,
    )
    assert comparison["overall_passed"] is True
    assert comparison["composite_difference_ci95"] == [1.0, 1.0]


def test_v3_pilot_gate_remains_deterministic_when_stochastic_would_pass() -> None:
    failed_summary: dict[str, object] = {
        "capability_rates": {
            "gathered_workbench_bundle_by_100": 0.0,
            "workbench_materials_available_by_300": 0.0,
            "workbench_complete": 0.0,
            "any_tool_crafted": 0.0,
        },
        "invalid_action_rate": 0.20,
    }
    passing_summary: dict[str, object] = {
        "capability_rates": {
            "gathered_workbench_bundle_by_100": 0.60,
            "workbench_materials_available_by_300": 0.10,
            "workbench_complete": 0.0,
            "any_tool_crafted": 0.0,
        },
        "invalid_action_rate": 0.05,
    }

    decision = pilot_continuation(
        failed_summary,
        {"composite_difference": -0.10},
        passing_summary,
        {"composite_difference": 0.10},
    )

    assert decision["continue"] is False
    assert decision["failure_mode"] == "deterministic_coordination_collapse"
    stochastic = decision["seeded_stochastic_diagnostic"]
    assert isinstance(stochastic, dict)
    assert stochastic["would_continue"] is True


def test_episode_exports_action_resource_and_rejection_diagnostics() -> None:
    result = run_civilization_episode(
        policy_name="legal_random",
        policy="legal_random",
        seed=0,
    )

    assert sum(result.selected_verbs.values()) == result.submitted_actions
    assert sum(result.selected_actions.values()) == result.submitted_actions
    assert sum(result.rejection_reasons.values()) == result.invalid_actions
    assert {"food", "wood", "stone"} <= result.peak_camp_stockpile.keys()
    assert result.total_agent_return == pytest.approx(
        sum(result.per_agent_returns.values())
    )
    assert result.shared_reward_component_totals


def test_model_evaluation_modes_are_seed_reproducible() -> None:
    class UniformModel:
        def __call__(
            self,
            observations: np.ndarray,
            *,
            training: bool,
        ) -> tuple[np.ndarray, np.ndarray]:
            del training
            return (
                np.zeros(
                    (len(observations), V2_FLAT_ACTION_COUNT),
                    dtype=np.float32,
                ),
                np.zeros((len(observations), 1), dtype=np.float32),
            )

    model = UniformModel()
    for deterministic in (True, False):
        first = run_civilization_episode(
            policy_name="uniform",
            policy="model",
            seed=91,
            model=model,
            deterministic=deterministic,
        )
        second = run_civilization_episode(
            policy_name="uniform",
            policy="model",
            seed=91,
            model=model,
            deterministic=deterministic,
        )
        assert first.as_dict() == second.as_dict()


def test_tensorflow_ppo_collects_v2_masked_rollout() -> None:
    pytest.importorskip("tensorflow")
    trainer = PPOTrainer(
        PPOConfig(
            total_steps=20,
            environment_id=CIVILIZATION_V2_TRAINING_ENVIRONMENT,
            reward_contract=CIVILIZATION_PROBE_REWARD_CONTRACT,
            rollout_steps=2,
            num_agents=10,
            map_size=48,
            max_steps=600,
            hidden_sizes=(8,),
            train_epochs=1,
            minibatch_size=16,
            checkpoint_dir=None,
            reward_mode="none",
        )
    )

    batch = trainer.collect_rollout()

    assert batch.observations.shape == (20, 604)
    assert batch.action_masks.shape == (20, V2_FLAT_ACTION_COUNT)
    assert np.all(
        batch.action_masks[np.arange(batch.actions.shape[0]), batch.actions]
    )
