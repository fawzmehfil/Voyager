# Stage 3: Survival Economy Mechanics

Stage 3 turns Voyager's multi-agent island into a small survival economy. Agents now interact with a shared camp stockpile, build shelter, experience deterministic storms, and expose achievement/metrics data for evaluation.

## Implemented

- New actions:
  - `DEPOSIT_FOOD`
  - `DEPOSIT_WOOD`
  - `DEPOSIT_STONE`
  - `WITHDRAW_FOOD`
  - `BUILD_SHELTER`
- Camp stockpile for food, wood, and stone.
- Food withdrawal from camp.
- Shelter construction from carried wood or stone.
- Builder role bonus for shelter construction.
- Deterministic storm schedule.
- Shelter progress reduces storm damage.
- Periodic food regeneration.
- Global achievements.
- `VoyagerParallelEnv.metrics()`.

## Camp Rules

Deposit, withdraw, and build actions require the agent to stand on the camp tile. This keeps the shared economy spatial and gives scripted/learned policies a clear target location.

## Metrics

`metrics()` returns a JSON-like dictionary with:

- current step.
- active agents.
- deaths.
- camp stockpile and shelter progress.
- storm status.
- achievements.
- total deposits, withdrawals, and build actions.

## Excluded

- Learned PPO policies.
- Episode artifact recorder.
- Web replay.
- Currency or market economy.
- Explicit agent communication.

## Stage 4 Handoff

Stage 4 should add baseline policies and evaluation tooling so random, greedy survival, and cooperative behaviors can be compared before PPO training.
