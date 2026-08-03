# Voyager

Voyager is an auditable cooperative multi-agent reinforcement-learning benchmark under
development, built around a bounded stranded-island economy and paired with a web replay
viewer. Its central experiment is: under the same interaction budget, which RL algorithm
can most reliably convert finite, spatially distributed resources into population survival
and rescue on unseen islands?

Stage 5.6 validates three frozen two-million-agent-step TensorFlow PPO policies on 100 held-out seeds each. Across the 300 official deterministic PPO episodes, every run retained 10/10 survivors and completed the shelter. The hierarchical-bootstrap civilization score is 89.42 (95% CI 84.19-93.17). Stage 6 turns those trajectories into a versioned saved-replay platform and a game-like web viewer.

Stage 7A expands that same simulator into a deterministic 48×48 handcrafted island without
removing the compact benchmark. `VoyagerCivilization-v1` adds structured actions, a
workbench-to-spear-to-hunt-to-cooking progression, joint construction, a fueled campfire,
six-person shelter occupancy, and one-or-two seeded night stalkers whose 25-damage attacks
can be mitigated or suppressed through coordinated defense. The committed Replay 2.1
vertical slice runs for 600 ticks at two ticks per second.

Stage 7B keeps that same island and shared simulation engine. `VoyagerCivilization-v2`
adds deterministic intent resolution, targeted local entity slots, five owned tools,
food-lot provenance and spoilage, ground piles, structure damage and repair, downing and
revival, and an append-only conservation/contribution ledger. Replay 2.2 records those
facts without changing the Stage 7A replay or `VoyagerCivilization-v1` contract.

The first Stage 7C trainability probe completed two million transitions and failed: PPO
learned no production capability and scored below legal-random play. That result is kept as
`civilization_trainability_probe_v1`. The v2 remediation makes construction masks honest,
adds a non-privileged agent identity for shared-policy symmetry breaking, uses bounded
dense progression rewards, and replaces saturated metrics with timed economic outcomes.
Procedural generation remains gated on a successful trainability result.

## Why This Exists

Voyager is intended to compare learning methods for decentralized planning, resource
allocation, public investment, recovery, memory, and cooperative credit assignment. Ten
agents must acquire a compact production chain, physically redistribute private tools and
food, build shared infrastructure, survive repeated nights, and maintain a rescue beacon.
Deterministic simultaneous resolution and an append-only provenance ledger make resource
flows and claimed cooperation auditable.

The planned public benchmark, `VoyagerCollective-v1`, uses a procedural 48x48 island,
2,400 world ticks across eight day/night cycles, local structured observations, one shared
reward contract, eighteen outcome achievements, and held-out island seeds. The current
Stage 7A/7B implementation is a strong deterministic substrate, not yet a validated
benchmark: Stage 7C-7F must still demonstrate learnability, solvability, cooperation
dependence, closed-loop control, generalization, algorithm headroom, and practical compute.
The full design and its failure gates are in
[`docs/stage_7_civilization_benchmark_plan.md`](docs/stage_7_civilization_benchmark_plan.md).

## Python API

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
- Stage 7: Complete `VoyagerCollective-v1`: deterministic cooperative economy, trainability gate, procedural islands, fishing and rescue, PPO/recurrent PPO/MAPPO baselines, frozen evaluation, and final replay presentation.
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

Run the Stage 7C v2 250K remediation pilot first:

```bash
.venv-train/bin/python examples/run_stage7c_trainability_probe.py \
  --total-agent-transitions 250000 \
  --seed 0 \
  --dev-episodes 10 \
  --test-episodes 20 \
  --evaluation-milestones 250000 \
  --output-dir results/stage7c/ppo_probe_v2_250k_seed0
```

This pilot is a continuation decision, not the full pass gate. Continue to two million
transitions only if PPO exceeds random, gathers the workbench bundle promptly, produces
nonzero camp or construction progress, and keeps invalid actions below ten percent. The
output includes config, history, checkpoints, action distributions, rejection reasons,
resource flows, capability outcomes, paired comparisons, and timing.

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

Stage 7 now targets one fixed eight-cycle, 2,400-tick benchmark rather than a ladder of
ever-longer campaigns. The compact progression adds one renewable investment, a fishing
net, and one terminal public project, a rescue beacon. The task should be easy to begin,
difficult to complete, and calibrated so survival is learnable, majority rescue requires
the shared economy, and perfect rescue remains exceptional. Feed-forward learning is
tested before further content is added; recurrent PPO and MAPPO are trained only after
that gate passes.

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

Build and run the single-process production container:

```bash
docker build -t voyager .
docker run --rm -p 8000:8000 voyager
```
