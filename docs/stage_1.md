# Stage 1: Single-Agent Survival Prototype

Stage 1 replaces the `VoyagerSingleAgent-v0` placeholder with a real Gymnasium environment. The environment is a deterministic, single-agent island-survival prototype.

## Implemented

- Seeded 32x32 island generation.
- Terrain ids: water, beach, grass, forest, quarry.
- Resource ids: food, wood, stone.
- One agent with position, health, hunger, energy, and inventory.
- Actions: noop, move up/down/left/right, gather, eat, rest.
- Hunger and health survival pressure.
- Movement blocked by water and map bounds.
- Inventory capacity per resource.
- Gymnasium `spaces.Dict` observations.
- Simple reward function.
- ANSI text render mode.

## Observation Space

`VoyagerSingleAgent-v0` returns:

- `local_view`: `uint8` array with shape `(7, 7, 3)`.
  - channel 0: terrain id.
  - channel 1: visible resource id.
  - channel 2: agent marker at the center.
- `stats`: normalized `[health, hunger, energy]`.
- `inventory`: normalized `[food, wood, stone]`.
- `progress`: normalized step progress through the episode.

## Reward Rules

- Small alive reward every step.
- Positive reward for successful gather.
- Positive reward for eating when meaningfully hungry.
- Small positive reward for useful rest when energy is low.
- Small penalty for invalid or wasted actions.
- Small ongoing hunger pressure penalty.
- Large death penalty.

The reward is intentionally simple. PPO reward tuning begins later.

## Excluded

- Multi-agent PettingZoo environment.
- Shared camp.
- Shelter construction.
- Weather and storms.
- Crafting tree.
- Recorder and replay artifacts.
- TensorFlow PPO.
- Phaser/web map rendering.
- LLM policy layer.

## Stage 2 Handoff

Stage 2 should introduce the true multi-agent environment around the same simulation style. It should keep the single-agent environment stable while adding multiple agents, shared camp state, simultaneous actions, and PettingZoo `ParallelEnv` support.
