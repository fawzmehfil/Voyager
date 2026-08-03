# Stage 6: Recorder And Web Replay

## Goal

Make Voyager runs scientifically inspectable and visually understandable. The Python
simulation remains authoritative; the browser reads versioned replay artifacts and never
implements environment rules.

## Phase 6A: Replay Contract

Define a versioned manifest containing:

- Replay, environment, reward, observation, action, and scenario versions.
- Seed and complete environment configuration.
- Policy type, checkpoint metadata, training seed, and inference mode.
- Episode status, terminal metrics, and artifact paths.

Record:

- Initial terrain, resources, camp, and agents.
- Every submitted action and action mask.
- State-changing events and named reward components.
- Resource gathering, deposits, withdrawals, consumption, and construction.
- Storm transitions, deaths, and new achievements.
- Periodic complete snapshots for seeking and recovery.

Use JSON for manifests/events and compressed NumPy or equivalent storage for large snapshots.
Do not optimize storage until the schema and reconstruction tests are stable.

## Phase 6B: Python Recorder And Loader

- Policy-agnostic recorder wrapper.
- Atomic run completion marker so partial runs are detectable.
- Schema validation with useful version errors.
- Replay loader with random access by tick.
- Deterministic reconstruction test comparing terminal metrics with the original run.
- CLI to record random, scripted, deterministic PPO, and stochastic PPO episodes.

## Phase 6C: Replay API

FastAPI endpoints:

- List runs and filter by policy, scenario, seed, and status.
- Return run manifest and summary.
- Return initial state, event ranges, snapshot ranges, and metric series.
- Stream large artifacts without loading an entire run into memory.

Live WebSocket simulation is optional. Saved replay is the required first path.

## Phase 6D: Web Viewer

Build the actual replay experience as the first screen:

- Pixel-art island map with visible terrain, resources, camp, structures, and agents.
- Play, pause, speed control, single-step, and timeline scrubber.
- Clickable agents with role, inventory, health, hunger, energy, and current action.
- Event timeline for gathering, transactions, construction, storms, achievements, and deaths.
- Survivor, hunger, food reserve, shelter, and reward-component charts.
- Policy comparison tabs using equivalent scenario seeds.
- Stable responsive layouts for desktop and mobile.

The interface should be understandable without knowing PPO. Technical metrics can be
available in secondary panels rather than covering the primary simulation.

## Representative Runs

Record at least:

- Random baseline.
- Greedy survival baseline.
- Cooperative scripted baseline.
- Best deterministic PPO checkpoint.
- Seed-matched stochastic PPO checkpoint.
- One failure or stress case if available.

Do not fabricate narrative summaries. Generate them from recorded events and metrics.

## Acceptance Criteria

- A saved episode can be replayed without executing the environment.
- Scrubbing reconstructs the same state at any tick.
- Terminal replay metrics match the original episode.
- Incomplete and incompatible runs fail clearly.
- Viewer renders the map, agents, controls, event timeline, and charts without overlap.
- Random, scripted, and PPO runs can be compared on a shared scenario.
- Backend tests, frontend tests, lint, typing, production build, and browser verification pass.

## Stage 7 Handoff

The replay schema must allow new structures, day phases, resources, achievements, and event
types without breaking old Stage 5.5 runs. Stage 7 uses that foundation to build one
auditable cooperative-planning benchmark: ten agents on a procedural 48x48 island, an
eight-cycle workbench-to-rescue economy, deterministic resource provenance, held-out
islands, and feed-forward PPO, recurrent PPO, and MAPPO comparisons under one fixed
interaction budget.
