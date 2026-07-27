# Stage 4: Baseline Policies And Evaluation

Stage 4 adds reproducible non-learning baselines for Voyager. These policies make the environment useful before PPO by giving random, selfish, and cooperative behaviors to compare.

## Implemented

- `Policy` protocol.
- `RandomPolicy`.
- `GreedySurvivalPolicy`.
- `CooperativePolicy`.
- `examples/evaluate_baselines.py`.
- Tests for policy validity and deterministic evaluation.

## Baseline Behaviors

- Random samples uniformly from the current action space.
- Greedy survival eats when hungry, gathers visible resources, withdraws food at camp when needed, and explores deterministically.
- Cooperative policy deposits surplus resources at camp, builders construct shelter when carrying materials, withdraws food only when hungry, and otherwise gathers role-relevant resources.

## Evaluation

Run:

```bash
python examples/evaluate_baselines.py --episodes 3 --max-steps 300 --num-agents 10
```

The script prints mean reward, survivors, deaths, shelter progress, camp food, and achievement count for each policy.

## Excluded

- PPO/TensorFlow training.
- Learned policies.
- Replay artifact writer.
- Web replay.
- Policy leaderboard persistence.

## Stage 5 Handoff

Stage 5 should use these baselines as non-learning comparisons while implementing PPO. The policy/evaluation interface should stay stable enough that PPO can be evaluated in the same table.
