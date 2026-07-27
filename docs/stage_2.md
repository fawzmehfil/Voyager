# Stage 2: Multi-Agent Environment

Stage 2 adds Voyager's first PettingZoo-style multi-agent environment: `VoyagerParallelEnv`. It keeps the Stage 1 single-agent Gymnasium environment stable while introducing multiple agents acting in one shared island world.

## Implemented

- `VoyagerParallelEnv` with PettingZoo `ParallelEnv` reset/step signatures.
- `VoyagerSurvival-v0` Gymnasium id points to the parallel environment with Gym wrappers disabled.
- Multiple agents named `agent_0`, `agent_1`, etc.
- Stable roles: forager, woodcutter, builder.
- Seeded non-overlapping land spawns near the camp.
- Simultaneous actions resolved in stable agent-id order.
- Movement blocked by water, map bounds, and occupied tiles.
- Dead agents are removed from active `env.agents`.
- Camp position and placeholder stockpile appear in observations and info.
- ANSI render mode for quick map inspection.

## Observation Space

Each live agent receives:

- `local_view`: `uint8` array with shape `(7, 7, 4)`.
  - terrain id.
  - visible resource id.
  - live-agent occupancy marker.
  - camp marker.
- `stats`: normalized `[health, hunger, energy]`.
- `inventory`: normalized `[food, wood, stone]`.
- `role`: one-hot `[forager, woodcutter, builder]`.
- `camp`: normalized `[food, wood, stone, shelter_progress]`.
- `progress`: normalized episode progress.

## Excluded

- Camp deposit/withdraw mechanics.
- Shelter construction.
- Storms.
- Food regeneration.
- Achievements and economy metrics.
- Baseline policies.
- PPO/training.

## Stage 3 Handoff

Stage 3 should turn the camp placeholder into a survival economy institution. It should add deposit/withdraw actions, shelter building, storms, food regeneration, achievements, and environment-level metrics.
