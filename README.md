# Voyager

Voyager is a Python-first multi-agent reinforcement learning environment for a stranded-island survival economy, paired with a web-based replay/demo layer. The project is inspired by the compact survival benchmark shape of Crafter: a small world, clear environment stepping, achievements, recorded runs, and behavior that can be inspected visually.

Stage 5 adds shared-policy TensorFlow PPO training for the multi-agent survival economy. A recorder, Phaser rendering, and web replay are planned for later stages.

## Why This Exists

Voyager is meant to become an RL environment where agents learn to survive under scarcity. The first real environment will model agents stranded on an island with food, wood, stone, hunger, energy, shared camp storage, shelter construction, storms, and role specialization. The same runs should also be easy to showcase in a browser, with replay controls, agent panels, event timelines, and metrics.

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

env = VoyagerParallelEnv(num_agents=10, days=30)
obs, infos = env.reset(seed=0)

while env.agents:
    actions = {
        agent_id: env.action_space(agent_id).sample()
        for agent_id in env.agents
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)
```

The multi-agent environment supports seeded reset/step loops, roles, shared map state, collision handling, camp deposits/withdrawals, shelter construction, deterministic storms, food regeneration, metrics, ANSI rendering, scripted baselines, and shared-policy PPO training.

## Development Stages

- Stage 0: Project skeleton.
- Stage 1: Single-agent Crafter-style prototype.
- Stage 2: Multi-agent environment.
- Stage 3: Survival economy mechanics.
- Stage 4: Random, greedy, and cooperative baseline policies.
- Stage 5: TensorFlow PPO training. Current.
- Stage 6: Web replay viewer.
- Stage 7: Notable runs page.
- Stage 8: Optional LLM policy layer.

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
  --total-steps 200000 \
  --rollout-steps 128 \
  --num-agents 10 \
  --max-steps 300 \
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

`--total-steps` counts agent transitions, not only world ticks. With `10` agents and `128` rollout steps, one PPO update collects up to `1280` training samples.

## Frontend Setup

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
