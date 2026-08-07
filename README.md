# Voyager

Voyager is a two-agent reinforcement-learning benchmark for cooperative capability
acquisition in a finite stranded-island economy. It asks whether a learning method can train
a decentralized pair to explore, gather, return resources, build public infrastructure,
survive repeated nights, and complete a rescue signal on unseen islands.

The canonical `VoyagerIsland-v1` contract deliberately keeps the research task smaller than
the underlying game engine:

- Two symmetric agents sharing one policy.
- Deterministic simultaneous actions and conflict resolution.
- A 48×48 island with a 1,200-tick, four-night horizon.
- A fixed nineteen-action categorical interface and 373-value structured actor input.
- Finite food, wood, stone, deer, camp storage, two tools, four public structures, and one or
  two dangerous night stalkers.
- Fifteen outcome achievements and a reward-independent geometric-mean evaluation score.
- Frozen train, development, and held-out test seed manifests.
- Legal-random, public-observation scripted-oracle, feed-forward PPO, and recurrent PPO
  baseline paths.
- Replay 2.3 recording through the existing web viewer and conservation ledger.

The older `VoyagerCivilization-v1` and `VoyagerCivilization-v2` environments remain supported
advanced sandboxes over the same simulation. Their ten-agent population, 270-action targeted
interface, spoilage, transfers, repair, and revival are not part of the official v1 benchmark.
The failed Stage 7C experiments on that broader interface remain documented as evidence for
why the benchmark was narrowed rather than silently discarded.

## Why This Exists

Voyager is intended to compare learning methods for decentralized exploration, temporal
composition, shared resource allocation, public investment, memory, and cooperative credit
assignment. Its evaluation reports every semantic achievement before aggregating them, so a
method cannot hide a missing capability behind one dense return.

The environment is implemented, deterministic, procedurally seeded, replayable, and wired to
the existing PPO trainers. A seed-0 fixed-island 250K run now establishes early-economy
separation from legal random, but this is not yet a scientifically validated benchmark result:
the predeclared multi-seed procedural baselines must still demonstrate generalization and
late-game capability. The frozen contract and remaining acceptance gates are in
[`docs/stage_7_island_benchmark_v1.md`](docs/stage_7_island_benchmark_v1.md).

## Python API

Canonical multi-agent benchmark:

```python
from voyager.envs import VoyagerIslandEnv

env = VoyagerIslandEnv(procedural=True)
observations, infos = env.reset(seed=0)

while env.agents:
    actions = {
        agent_id: env.action_space(agent_id).sample(mask=infos[agent_id]["action_mask"])
        for agent_id in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

The canonical interface follows PettingZoo's parallel API. `gym.make("VoyagerIsland-v1")`
registers the same environment, while `VoyagerIslandCentralized-v1` is an optional
Gymnasium wrapper for a controller that emits both actions.

Gymnasium-style single-agent prototype:

```python
import gymnasium as gym
import voyager

env = gym.make("VoyagerSingleAgent-v0")
obs, info = env.reset(seed=0)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

PettingZoo-style multi-agent environment:

```python
from voyager.envs import VoyagerParallelEnv

env = VoyagerParallelEnv(num_agents=10, max_steps=300)
obs, infos = env.reset(seed=0)

while env.agents:
    actions = {
        agent_id: env.action_space(agent_id).sample()
        for agent_id in env.agents
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)
```

The multi-agent environment supports seeded reset/step loops, roles, shared map state, collision handling, camp deposits/withdrawals, shelter construction, deterministic storms, food regeneration, metrics, ANSI rendering, scripted baselines, and shared-policy PPO training. Each agent's `info["action_mask"]` identifies actions that are currently legal and useful.

## Development Stages

- Stage 0: Project skeleton.
- Stage 1: Single-agent island-survival prototype.
- Stage 2: Multi-agent environment.
- Stage 3: Survival economy mechanics.
- Stage 4: Random, greedy, and cooperative baseline policies.
- Stage 5: TensorFlow PPO training.
- Stage 5.5: Action masking, entropy decay, economy/group reward shaping, and three reference training runs. Complete.
- Stage 5.6: Held-out seeds, achievement success rates, civilization score, benchmark exports, and ablation support. Complete.
- Stage 6: Versioned recorder, random-access loader, API, curated run library, comparison, presentation mode, and interactive pixel-art viewer. Complete.
- Stage 7: Complete `VoyagerIsland-v1`: two-agent deterministic economy, compact public API, fixed-island trainability gate, procedural seed splits, rescue, PPO/recurrent PPO baselines, frozen evaluation, and Replay 2.3 presentation.
- Stage 8: Optional research extensions such as pixels, communication, mixed incentives, variable populations, additional algorithms, or accelerated backends.
- Stage 9: Public research release, reproducibility package, benchmark tables, report, and polished replay showcase. Voyager may be called complete here.
- Stage 10: Optional LLM interface only if a concrete later use case justifies it.

## Backend Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Basic checks:

```bash
python -m compileall voyager examples
python -c "import voyager; print(voyager.__version__)"
python examples/random_rollout.py
python examples/evaluate_baselines.py --episodes 1 --max-steps 100 --num-agents 5
```

## PPO Training

TensorFlow does not currently support every newest Python release. Use Python 3.11 or 3.12 for training:

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
```

Run a small smoke training job:

```bash
python examples/train_ppo.py \
  --total-steps 5000 \
  --rollout-steps 64 \
  --num-agents 5 \
  --max-steps 150 \
  --checkpoint-dir checkpoints/stage5
```

Run a more meaningful first experiment:

```bash
python examples/train_ppo.py \
  --total-steps 1000000 \
  --rollout-steps 128 \
  --num-agents 10 \
  --max-steps 300 \
  --entropy-coef-start 0.02 \
  --entropy-coef-end 0.001 \
  --checkpoint-dir checkpoints/stage5
```

Evaluate the trained checkpoint against the Stage 4 baselines:

```bash
python examples/evaluate_baselines.py \
  --episodes 3 \
  --max-steps 300 \
  --num-agents 10 \
  --ppo-checkpoint checkpoints/stage5/latest
```

When a PPO checkpoint is provided, evaluation always prints separate `ppo_deterministic` and `ppo_stochastic` rows. The environment masks impossible or currently useless actions during training and inference, including empty gathers, eating without food, invalid camp transactions, and shelter building without material.

`--total-steps` counts agent transitions, not only world ticks. With `10` agents and `128` rollout steps, one PPO update collects up to `1280` training samples. Entropy decays linearly across those agent transitions from `--entropy-coef-start` to `--entropy-coef-end`.

Run the reduced Stage 7 fixed-island trainability gate:

```bash
.venv-train/bin/python examples/run_stage7_island_fixed_gate.py \
  --total-agent-transitions 250000 \
  --seed 0 \
  --dev-episodes 10 \
  --eval-episodes 20 \
  --evaluation-milestones 50000 100000 150000 200000 250000 \
  --output-dir results/stage7/island_progression_v4_250k_seed0
```

This trains shared feed-forward PPO on `VoyagerIsland-v1`, evaluates stochastic and
deterministic inference at fixed milestones on development seeds, selects by stochastic
achievement score with invalid-action rate as its tie-breaker, and applies the predeclared
gate once to that locked checkpoint on held-out test seeds. The final checkpoint is also
reported as a collapse diagnostic but cannot replace the development-selected checkpoint.
Do not begin the longer procedural baseline runs if this gate fails. The runner uses the bounded
`voyager_island_progression_reward_v4` contract while public evaluation remains entirely
achievement-based. V4 preserves the 48x48 island and nineteen-action interface, but exposes
one technology branch at a time, reduces construction cost, rewards only currently useful
deposits and applied labor, and turns beacon completion into a 100-tick extraction window.
The original v1 reward gate failed because PPO learned wood gathering but not stone,
returning, delivery, or construction; its artifact remains preserved at
`results/stage7/island_fixed_gate_seed0`. The earlier ten-agent
Stage 7C diagnostic commands and results remain in the historical Stage 7 documentation;
they are not part of the canonical benchmark workflow.

The verified seed-0 run selects its 200K checkpoint and passes the held-out gate: achievement
score `0.122` versus legal random `0.019`, wood/stone delivery `70%/75%`, workbench completion
`50%`, and first-night achievement `90%`. This validates early-economy trainability, not
complete-game mastery.

Evaluate the public-observation safety oracle on the frozen held-out split:

```bash
.venv-train/bin/python examples/evaluate_stage7_island.py \
  --policy oracle \
  --split test \
  --output results/stage7/island_oracle_test_v1.json
```

The current oracle passes the solvability gate on all 100 test seeds: every achievement is
completed on at least 96% of islands, rescue succeeds on 96%, and the achievement geometric
mean is `0.986`. This is a solvability check, not a learned benchmark result.

Validate one official procedural run without creating artifacts or loading TensorFlow:

```bash
.venv-train/bin/python examples/run_stage7_island_procedural.py train \
  --algorithm feed_forward_ppo \
  --training-seed 0 \
  --dry-run
```

The official procedural matrix contains six separate one-million-transition commands:

```bash
.venv-train/bin/python examples/run_stage7_island_procedural.py train --algorithm feed_forward_ppo --training-seed 0
.venv-train/bin/python examples/run_stage7_island_procedural.py train --algorithm feed_forward_ppo --training-seed 1
.venv-train/bin/python examples/run_stage7_island_procedural.py train --algorithm feed_forward_ppo --training-seed 2
.venv-train/bin/python examples/run_stage7_island_procedural.py train --algorithm recurrent_ppo --training-seed 0
.venv-train/bin/python examples/run_stage7_island_procedural.py train --algorithm recurrent_ppo --training-seed 1
.venv-train/bin/python examples/run_stage7_island_procedural.py train --algorithm recurrent_ppo --training-seed 2
```

Each command trains only on the frozen `0-999` manifest, evaluates five checkpoints on all
50 development islands, and locks the strongest seeded-stochastic achievement score. It does
not access held-out test islands. After all six selections are locked, intentionally open the
test split once with:

```bash
.venv-train/bin/python examples/run_stage7_island_procedural.py finalize
```

Use `--smoke` on a training command for a nonofficial 2,560-transition integration check.
Smoke runs default to `results/stage7/procedural_baselines_smoke_v1`, separate from the
official experiment root. Generated runs remain under `results/`; they are not benchmark
evidence until the complete official matrix is finalized and statistically analyzed.

## Stage 5.6 Benchmark

The final benchmark evaluates random, greedy, cooperative, and all three frozen PPO checkpoints on seeds `10000000` through `10000099`. Deterministic PPO is the official learned-policy result; stochastic PPO is reported separately.

Official deterministic PPO family results across 300 episodes:

- Mean survivors: `10.00/10` (all episodes).
- Mean shelter progress: `1.00` (all episodes completed the shelter).
- Mean achievements: `14.46/16`.
- Civilization score: `89.42`, hierarchical-bootstrap 95% CI `84.19-93.17`.
- Individual checkpoint scores: `93.16`, `90.80`, and `83.74`.

Baseline civilization scores were `29.45` for random, `24.93` for greedy survival, and `33.02` for cooperative scripted play. The benchmark exports episode-level JSONL, aggregate JSON, achievement CSV, policy CSV, reward/action diagnostics, metric curves, and artifact checksums. Frozen checkpoints, manifests, and the compact final result are under `benchmarks/`.

Run or reproduce it with:

```bash
python examples/run_benchmark.py \
  --manifest benchmarks/manifests/stage5_6_final.json \
  --output results/benchmark/stage5_6_final_v1 \
  --resume
```

The environment also exposes `VoyagerReward-v0`, `VoyagerAchievement-v0`, and `VoyagerNoReward-v0`, plus training flags for action-mask, reward-component, and role-observation ablations.

Stage 7's canonical task is the four-cycle, 1,200-tick `VoyagerIsland-v1` contract described
above. Fishing, renewable ecology, larger populations, and MAPPO are optional extensions,
not prerequisites for a useful v1 benchmark.

## Stage 6 Replay Platform

Stage 6 records simulation output into `stage6_replay_2.0.0`, a portable directory format:

```text
manifest.json
initial.json.gz
timeline/000001-000100.json.gz
snapshots/000000.json.gz
metrics.json.gz
camera.json
```

Timeline chunks are 100 ticks, reconstruction snapshots are every 25 ticks, and every artifact has a SHA-256 checksum. Readers do not import TensorFlow or execute environment rules. Unknown minor fields and namespaced extensions are safe, while unknown major versions fail clearly.

Record and inspect a versioned policy/seed pair:

```bash
voyager-replay record \
  --manifest benchmarks/manifests/stage5_6_final.json \
  --policy ppo_seed0_deterministic \
  --seed 10000010 \
  --tag local

voyager-replay list
voyager-replay inspect runs/replays/<replay-id>
voyager-replay validate --deep runs/replays/<replay-id>
```

An arbitrary compatible training checkpoint can be recorded with `--checkpoint checkpoints/stage5/latest`. Training also supports `--record-final-replay`; evaluation and benchmark commands support explicit `--record-replay POLICY:SEED` hooks. Training records only a post-training evaluation, never the training trajectory.

Five validated seed-matched recordings live under `benchmarks/replays/stage6_curated_v1/`: random, greedy, cooperative, deterministic PPO, and stochastic PPO. Their compressed total is approximately 1.1 MB. The deterministic PPO replay is the default showcase; the cooperative replay is intentionally retained as an honest failure case. The original Stage 6A JSON remains a permanent compatibility and visual-regression fixture.

Run the unified read-only application:

```bash
voyager-web
```

It serves the API and the production React build from `http://127.0.0.1:8000`. Useful routes are:

- `/` — polished default showcase.
- `/replays/{replay_id}` — general viewer with optional inspection.
- `/compare?left=random&right=ppo_seed0_deterministic` — synchronized same-seed comparison.
- `/present/{replay_id}` — fixed, chrome-free capture mode.
- `/api/v1/replays` — filtered and cursor-paginated replay catalog.

Configure catalog roots with a path-separated `VOYAGER_REPLAY_ROOTS` value. `VOYAGER_HOST`, `VOYAGER_PORT`, `VOYAGER_FRONTEND_DIR`, and `VOYAGER_REPLAY_CACHE_SIZE` configure the server. The web service is deliberately read-only: new recordings enter through the recorder CLI or workflow hooks.

## Frontend Development

```bash
cd web
npm install
npm run dev
```

The local dev server should run at:

```text
http://localhost:5173
```

For a production build:

```bash
npm run build
```

Vite proxies `/api/v1` and `/healthz` to a local `voyager-web` process. The renderer uses crisp procedural 16-pixel art, nearest-neighbor scaling, modular chibi agents, semantic action/event animation, deterministic camera direction, exact seeking, and no browser-side simulation.

Generate the Stage 7A reachability artifact with:

```bash
python examples/generate_stage7a_replay.py
voyager-web
```

Then open `/replays/civilization_vertical_slice_v1`. The replay remains simulation-free in
the browser: structures, shelter occupancy, deer, stalkers, night lighting, attacks,
mitigation, and defeats are all rendered from recorded facts.

Generate and inspect the Stage 7B deterministic-core artifact with:

```bash
python examples/generate_stage7b_replay.py
voyager-web
```

Then open `/replays/civilization_deterministic_core_v1`. Its Replay 2.2 bundle includes a
separate `ledger.json.gz` artifact and deeply reconstructable state hashes for all 601
states (initial state plus 600 ticks).

Generate the canonical reduced Stage 7 benchmark replay with:

```bash
.venv-train/bin/python examples/generate_stage7_island_replay.py
voyager-web
```

Then open `/replays/island_benchmark_oracle_v1`. Replay 2.3 shows the complete two-agent
achievement chain, night defense, beacon construction, and final rescue from recorded facts.

Build and run the single-process production container:

```bash
docker build -t voyager .
docker run --rm -p 8000:8000 voyager
```
