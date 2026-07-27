# Stages 2-4 Technical Implementation Plan

This document defines the execution plan for Voyager Stages 2, 3, and 4. The goal is to move from the Stage 1 single-agent Gymnasium prototype to a usable multi-agent survival economy with baseline policies and reproducible evaluation scripts.

## Stage 2: Multi-Agent Environment

### Goal

Implement a PettingZoo `ParallelEnv` named `VoyagerParallelEnv` and make `VoyagerSurvival-v0` return that environment. This stage proves that multiple agents can act simultaneously in a shared island world without introducing the full survival economy yet.

### Environment API

- Add `voyager.envs.parallel_env.VoyagerParallelEnv`.
- Export it from `voyager.envs`.
- Register `VoyagerSurvival-v0` to `voyager.envs.parallel_env:VoyagerParallelEnv`.
- Constructor defaults:
  - `num_agents=10`
  - `map_size=32`
  - `max_steps=1000`
  - `local_view_size=7`
  - `inventory_capacity=10`
  - `render_mode=None | "ansi"`

### World Model

- Add multi-agent state under `voyager/sim/`.
- Agents should be keyed as `agent_0`, `agent_1`, etc.
- Agents have role metadata:
  - `forager`
  - `woodcutter`
  - `builder`
- Role is represented in observations as a one-hot vector.
- Agents spawn on valid land tiles near the center without overlapping.
- Simultaneous actions are resolved in a stable agent-id order.
- Movement cannot enter water, leave bounds, or move into an occupied tile.
- Dead agents are removed from `env.agents` but remain in `possible_agents`.

### Observation Space

Each agent receives a `spaces.Dict`:

- `local_view`: `uint8`, shape `(7, 7, 4)`
  - terrain id
  - resource id
  - agent/occupancy marker
  - camp marker
- `stats`: normalized `[health, hunger, energy]`
- `inventory`: normalized `[food, wood, stone]`
- `role`: one-hot `[forager, woodcutter, builder]`
- `camp`: normalized camp summary placeholder `[food, wood, stone, shelter_progress]`
- `progress`: normalized step progress

Stage 2 should include camp position/summary in observations, but camp mechanics remain minimal until Stage 3.

### Tests

- `VoyagerSurvival-v0` creates a PettingZoo-style parallel environment.
- `reset(seed=0)` returns observations/infos for all agents.
- Same seed reproduces maps, resources, roles, and spawns.
- Agents cannot overlap at reset.
- Random actions for all agents run at least 100 steps without crashing.
- Movement is blocked by water, bounds, and occupied tiles.
- Dead agents are removed from active `agents`.
- `render_mode="ansi"` returns a non-empty map with multiple agent markers.
- Existing `VoyagerSingleAgent-v0` tests still pass.

## Stage 3: Survival Economy Mechanics

### Goal

Turn the multi-agent island into a real survival economy by adding shared camp storage, deposit/withdraw actions, shelter construction, storms, food spawn pressure, and achievements. This stage makes cooperation and scarcity observable.

### New Actions

Extend the shared action enum in a backward-compatible way:

- `DEPOSIT_FOOD`
- `DEPOSIT_WOOD`
- `DEPOSIT_STONE`
- `WITHDRAW_FOOD`
- `BUILD_SHELTER`

Single-agent may keep using the same enum, but new actions can be invalid/no-op there unless intentionally supported later.

### Camp And Shelter

- Camp state:
  - `x`, `y`
  - stockpile: food, wood, stone
  - `shelter_progress` from `0.0` to `1.0`
  - `shelter_capacity`
- Deposit removes one resource from inventory and adds it to camp.
- Withdraw food removes one camp food and adds it to inventory.
- Build shelter consumes one wood or stone contribution and increases progress.
- Builders should receive a modest role bonus for shelter work.

### Storms And Scarcity

- Add deterministic storm schedule based on seed:
  - Storm every `storm_interval` steps after `storm_start_step`.
  - Storm lasts `storm_duration` steps.
- Unsheltered living agents lose health during active storms.
- Shelter progress reduces storm damage for up to `shelter_capacity` agents.
- Add optional food regeneration every `food_regen_interval` steps.
- Add `food_spawn_rate` constructor option for scenario pressure.

### Achievements And Metrics

- Track global achievements:
  - first deposit
  - first food withdrawal
  - shelter 25/50/100 percent
  - first storm survived
  - all active agents alive at 100 steps
- Add `env.metrics()` returning:
  - step
  - active agents
  - deaths
  - camp stockpile
  - shelter progress
  - achievements
  - total deposits/withdrawals/build actions
- Include event strings in `infos[agent]["event"]`.

### Tests

- Deposit and withdraw update inventory/camp state.
- Shelter construction consumes material and increases progress.
- Storms damage unsheltered agents.
- Shelter reduces storm damage.
- Food regeneration can add food resources.
- Achievements are set at expected milestones.
- Metrics return stable JSON-like data.
- Multi-agent random rollout still runs.

## Stage 4: Baseline Policies And Evaluation

### Goal

Add reproducible non-learning baselines so Voyager has useful comparison runs before PPO. Stage 4 should make it easy to compare random, greedy survival, and cooperative scripted behavior.

### Policy API

Add `voyager/policies/base.py`:

```python
class Policy:
    def act(self, agent_id: str, observation: dict, info: dict) -> int:
        ...
```

Implement:

- `RandomPolicy`
- `GreedySurvivalPolicy`
- `CooperativePolicy`

### Policy Behavior

- `RandomPolicy`: samples from the action space.
- `GreedySurvivalPolicy`:
  - eat when hungry and carrying food
  - withdraw food at camp when hungry
  - gather on useful resource tiles
  - move toward visible/known food when hungry
  - rest when low energy
- `CooperativePolicy`:
  - deposits surplus food/wood/stone at camp
  - builders prioritize shelter when carrying wood/stone
  - withdraw food only when hungry
  - otherwise gathers role-relevant resources

Policies can use observation plus info fields. They do not need pathfinding beyond a deterministic step toward camp or local visible resources.

### Evaluation Script

Add `examples/evaluate_baselines.py`.

The script should:

- Run random, greedy, and cooperative policies.
- Use fixed seeds.
- Run configurable episodes.
- Print a compact table:
  - policy
  - mean reward
  - mean survivors
  - mean deaths
  - mean shelter progress
  - mean camp food
  - achievements count
- Exit successfully without writing generated artifacts by default.

### Documentation

- Add `docs/stage_4.md`.
- Update README with:
  - PettingZoo API now implemented.
  - Baseline evaluation command.
  - Stage 4 as current.

### Tests

- Every policy returns valid actions.
- Evaluation can run one episode per policy quickly.
- Cooperative policy deposits resources when configured state makes that valid.
- Greedy policy eats when hungry with food.
- Baseline evaluation returns deterministic metrics for fixed seeds.
- Full test suite, lint, mypy, random rollout, baseline evaluation, and web build all pass.

## Periodic Commit Plan

- Commit 1: Add this Stage 2-4 technical plan.
- Commit 2: Complete Stage 2 multi-agent environment.
- Commit 3: Complete Stage 3 survival economy mechanics.
- Commit 4: Complete Stage 4 baseline policies/evaluation.

Each commit should pass the relevant Python checks before pushing. Stage 4 completion requires the full acceptance suite.
