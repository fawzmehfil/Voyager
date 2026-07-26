# Stage 0: Project Skeleton

Stage 0 establishes the repository shape for Voyager. It creates the Python package, a placeholder Gymnasium environment registration, a basic frontend app, an example rollout script, run-output conventions, and project documentation.

## Included

- `voyager` Python package with version metadata.
- Environment registration for planned Gymnasium IDs.
- Placeholder environment that raises `NotImplementedError`.
- `examples/random_rollout.py` showing the intended future API.
- Vite React TypeScript web app placeholder.
- `runs/.gitkeep` for future run artifacts.
- README setup and stage documentation.
- `.gitignore` rules for caches, generated runs, logs, checkpoints, and local planning notes.

## Excluded

- Real map generation.
- Real agent state.
- Real action resolution.
- Real reward function.
- PPO or TensorFlow training.
- Recorder implementation.
- Phaser map rendering.
- Live simulation server.

## Acceptance Criteria

- `import voyager` succeeds.
- The package exposes `voyager.__version__`.
- Planned env IDs are registered with Gymnasium.
- Calling the placeholder env clearly says Stage 1 is not implemented yet.
- Frontend builds and renders a static Voyager placeholder.
- Generated runs, model checkpoints, logs, and local planning notes are ignored by Git.

## Stage 1 Handoff

Stage 1 should replace the placeholder behavior with the first real single-agent environment. It should add a deterministic 32x32 island, basic resources, hunger/health/energy state, simple actions, and a working Gymnasium loop for `VoyagerSingleAgent-v0`.
