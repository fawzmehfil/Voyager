# Voyager Technical Report Progress Notes

## Stage 7 Scope Reset: From Sandbox Complexity To Benchmark Clarity

The complete Stage 7C diagnostic sequence established a consistent result: the ten-agent,
270-action Civilization interface exposed many individually learnable primitives but ordinary
PPO did not compose gathering, returning, depositing, and construction reliably. Identity,
camp bearing, bounded rewards, a public objective board, recurrent memory, and factorized
action heads each clarified part of the failure without producing a useful learned economy.
The final factorized recurrent test therefore triggered the predeclared task/interface
simplification branch rather than MAPPO or a longer blind run.

`VoyagerIsland-v1` keeps the deterministic Stage 7A/7B engine, 48x48 island, public camp,
finite economy, night threats, ledger, and replay platform. It narrows the official benchmark
to two symmetric agents, nineteen categorical actions, a 373-value actor observation,
four nights, two tools, four structures, and fifteen semantic achievements. The broader
Civilization interfaces remain reproducible sandboxes.

This change is scientifically motivated rather than cosmetic: the benchmark now has one
falsifiable question, a fixed one-million-transition budget, disjoint procedural seed splits,
reward-independent scoring, legal-random and public-observation oracle comparators, and
predeclared trainability and release gates. The environment implementation is complete, but
the later seed-0 fixed-island pass establishes only early-economy trainability. No report
should claim benchmark validity until the multi-seed procedural and release gates pass.

Status: tracked historical notes for the final technical report. Superseded diagnoses remain
here deliberately so the report can distinguish failed experiments from current conclusions.

The reduced environment's public-observation safety oracle was then evaluated once on the
frozen 100-seed held-out split. It achieved at least 90% on every semantic achievement,
98% rescue, 98% joint survival, and a `0.976` geometric-mean score. This establishes that
unseen generated islands are solvable without privileged simulator state. It does not yet
establish learned-policy separation; that begins with the fixed-island 250K PPO gate.

### Reduced Fixed-Island Gate V1 Failure And V2 Remediation

The first reduced 250K gate also failed, but much more specifically than the earlier
ten-agent experiments. PPO learned food in 85% of evaluation episodes, wood in 100%, and
first-night survival in 60%, with only a 0.07% invalid-action rate. It never collected stone,
deposited materials, or built the workbench. Its score was `0.014` versus legal random's
`0.023`.

Policy tracing found that agent 0 performed every gathering interaction while agent 1 never
interacted. Material carriers never came within five tiles of camp and never selected
`DEPOSIT_ALL`. The fully shared achievement reward let one actor produce reward for both
training samples, while the return-to-camp dependency had no intermediate bounded signal.

`voyager_island_trainability_reward_v2` keeps the public world and evaluator frozen. It adds
one-time `+0.50` causal-contributor bonuses, three nonrepeatable `+0.20` material-return
distance milestones, and deterministic seed-based spawn-slot swapping. The repeated 250K
gate retains the exact same evaluation thresholds, so any improvement must appear in public
achievements rather than shaped training return.

### Reduced Fixed-Island V2 Outcome And Checkpoint Protocol Correction

V2 produced real but insufficient improvement. Seeded-stochastic PPO scored `0.025` versus
legal random's `0.023`, collected food in 85% of episodes, wood in 95%, stone in 15%, and
deposited wood in 10%. It never deposited stone or completed the workbench. The run therefore
failed the predeclared gate and procedural calibration remained blocked.

The training curve also exposed an experimental-design problem rather than a new world-design
conclusion. Entropy fell from roughly 1.4 during much of training to `0.013` at the final
checkpoint. Deterministic inference from that checkpoint selected almost entirely no-op and
scored `0.004`. Since the runner retained and evaluated only the last update, it could not tell
whether an earlier checkpoint had been stronger before the collapse.

The corrected runner now saves 50K-transition milestones, evaluates each on development seeds,
and selects the highest seeded-stochastic public achievement score, breaking ties by lower
invalid-action rate. It then evaluates that locked checkpoint once on disjoint held-out test
seeds. Deterministic inference and the final checkpoint remain diagnostics only. The world,
reward v2, action space, observation, architecture, budget, public score, and pass thresholds
are unchanged, so the rerun isolates checkpoint selection from environment changes.

## Project Story In One Paragraph

Voyager began as a small single-agent survival game, became a ten-agent shared survival
economy, gained trainable PPO baselines and a reproducible replay viewer, and then expanded
into a larger stranded-island benchmark. The compact environment proved that the training,
evaluation, and replay pipeline works. The present Stage 7 challenge is harder: agents can
learn individual skills such as gathering, returning to camp, and construction, but ordinary
feed-forward PPO has not yet reliably composed them into a complete gather-deliver-build
strategy. Current work is measuring partial capability correctly and testing whether memory
through recurrent PPO closes that gap before more content is added.

## Stage-By-Stage Progression

### Stage 0 — Repository Skeleton

- Created the Python package, planned environment registrations, frontend placeholder,
  examples, tests, and artifact conventions.
- No real simulation existed yet.
- Main result: established the intended installable RL-environment shape.

### Stage 1 — Single-Agent Survival

- Added a deterministic 32x32 island with terrain, food, wood, stone, hunger, health,
  energy, inventory, movement, gathering, eating, and resting.
- Exposed a Gymnasium observation/action loop.
- Main result: proved the core survival simulation could function as a standard RL task.

### Stage 2 — Multi-Agent Environment

- Added ten-agent PettingZoo-style parallel play, local observations, stable roles, shared
  world state, collisions, and a camp placeholder.
- Main result: changed the subject from one surviving policy to a population acting in one
  economy.

### Stage 3 — Shared Survival Economy

- Added camp deposits and withdrawals, shelter construction, storms, food regeneration,
  achievements, and group metrics.
- Main result: introduced spatial public investment: agents had to bring resources to a
  shared location and convert them into group protection.

### Stage 4 — Non-Learning Baselines

- Added random, greedy-survival, and cooperative scripted policies with reproducible
  evaluation.
- Main result: created behavioral reference points before attempting RL training.

### Stage 5/5.5 — Shared-Policy PPO

- Added TensorFlow PPO, GAE, action masking, checkpointing, entropy scheduling, named reward
  components, and deterministic/stochastic evaluation.
- All agents shared one neural network; every agent transition became a training sample.
- Three two-million-transition policies achieved 10/10 survival and full shelter completion
  on the compact task.
- Important limitation: early evaluation overlapped training seeds, so success demonstrated
  trainability but was not yet a defensible benchmark result.

### Stage 5.6 — Frozen Compact Benchmark

- Added disjoint held-out seeds, frozen contracts/checkpoints, achievement success rates,
  geometric-mean scoring, uncertainty reporting, and benchmark artifacts.
- Across the three trained policies, the official deterministic result retained 10/10
  survivors, completed shelter, and produced a civilization score of 89.42 with a reported
  95% confidence interval of 84.19–93.17.
- Main result: proved Voyager could train, freeze, evaluate, and reproduce a real RL
  benchmark row.
- Limitation motivating Stage 7: the compact task was comparatively short and easy; it did
  not strongly test long dependency chains, memory, or richer resource allocation.

### Stage 6 — Replay And Viewer

- Added versioned saved replays, deterministic reconstruction, an API, and a web viewer.
- The browser displays recorded simulation facts rather than reimplementing game mechanics.
- Main result: made training outcomes auditable and visually demonstrable.

### Stage 7A — Larger Civilization Island

- Expanded the same project and engine to a handcrafted 48x48 island with ten agents and a
  600-tick/two-cycle episode.
- Added workbench, private tools, hunting, cooking, campfire, shelter, joint construction,
  and one or two dangerous night stalkers.
- Main result: created a richer survival technology chain without replacing the compact
  environment.

### Stage 7B — Deterministic And Auditable Core

- Added order-independent intent resolution, targeted actions, ownership and transfers,
  food provenance/spoilage, ground piles, structure damage/repair, downing/revival, and an
  append-only conservation/contribution ledger.
- Main result: made simultaneous multi-agent outcomes reproducible and resource flows
  auditable enough for research evaluation.

## Stage 7C — Current Trainability Problem

The current question is not whether the game runs. It is whether a standard RL algorithm
can learn meaningful progression in the richer environment under a practical budget.

### Full-Environment Probe V1

- Feed-forward PPO trained for roughly two million agent transitions.
- It performed below legal random, selected many invalid actions, and learned no reliable
  production progression.
- Lesson: the first observations, masks, and reward signals were not sufficient.

### Remediation V2

- Improved masks, added agent identity, and revised progression rewards.
- PPO discovered a dominant stone-mining strategy because repeated mining was an easier
  source of reward than completing the economy.
- Lesson: reward was being optimized correctly but represented the wrong objective—classic
  reward hacking/specification failure.

### Remediation V3

- Capped gathering/delivery credit, assigned individual credit and invalid penalties, and
  added a nonprivileged bearing back to camp.
- Sampled PPO gathered useful wood and stone and greatly reduced invalid actions.
- It still failed to deposit stone or begin the workbench consistently. Deterministic
  argmax inference often synchronized agents into no-op behavior.
- Lesson: primitive behavior improved, but the complete sequence still did not emerge.

### Diagnostic Learning Ladder

Separate temporary tasks tested the pieces of the dependency chain:

- Wood gathering showed learning above random.
- Stone gathering learned a faster directed route, although its raw success rate was close
  to random.
- Returning pre-carried materials to camp reached 100% success versus 0% for random.
- Pre-stocked workbench construction reached 100% success.
- Basic first-night survival was achievable, although the weak survival task did not show a
  strong advantage over random.

The original report incorrectly called return-to-camp a capability failure because it mixed
success with a strict collision-efficiency condition. This was corrected.

### Current Diagnosis

The evidence supports:

```text
individual skills are learnable
              but
feed-forward PPO does not yet compose them reliably
```

Likely sources of difficulty include:

- Partial observability: resources and camp can leave the local view.
- Memory: an agent must remember what it was collecting and why it is travelling.
- Delayed credit: gathering only becomes strategically useful after returning, depositing,
  and building.
- Shared-policy symmetry: similar agents can choose the same action and collide or collapse
  to the same deterministic behavior.
- Large action registry: PPO must select the right verb, argument, and sometimes target.
- Multi-agent credit assignment: group progress depends on several agents' temporally
  separated contributions.

This does **not** yet prove that the world is too complex or untrainable. It proves that the
current feed-forward baseline has not learned the full composition, while controlled tests
show that the mechanics themselves support learning.

## What We Changed In Response

### Achievement-Spectrum Evaluation

The earlier all-or-nothing continuation composite could hide partial learning. The new
`civilization_achievement_benchmark_v1` evaluates fifteen independent population
achievements across:

- Gathering.
- Camp delivery.
- Workbench progression.
- Tool crafting and transfer.
- First-night and terminal survival.

Every success rate is reported and combined with a smoothed geometric mean. The score gives
partial learning visibility while still penalizing policies that only master easy skills.
Training reward and evaluation achievements are separate contracts, so algorithms are
compared by outcomes rather than by how much shaped reward they collected.

Seeded-stochastic PPO is now the primary evaluation because PPO trains a categorical action
distribution. Deterministic argmax remains recorded as a useful synchronization-collapse
diagnostic.

### Recurrent PPO

Recurrent PPO adds GRU memory to ordinary PPO. All ten agents still share network weights,
but each maintains its own private memory state. It is trained directly on the complete
600-tick environment with the full action registry—no diagnostic curriculum or privileged
global observation.

The intended test is:

```text
legal random < feed-forward PPO < recurrent PPO
```

This asks whether memory helps compose gathering, return, delivery, and construction.

The first 250K recurrent pilot failed this test. Legal random scored 0.139,
feed-forward PPO 0.049, and recurrent PPO 0.022. The recurrent population learned a strong
wood-gathering habit but made no deposits in twenty held-out episodes, never started the
workbench, and suffered almost complete population collapse. Its invalid-action rate also
rose above both feed-forward and random. The result shows that memory alone does not repair
the hidden team-state and composition problem.

### Team-Objective V4

V3 rewards cumulative team gathering and camp high-water milestones, but agents away from
camp cannot observe those shared counters. The next controlled contract adds six globally
shared normalized values: gathered wood/stone progress, delivered wood/stone progress,
workbench progress, and active-population fraction. It adds no locations or private
inventories. Fresh feed-forward and recurrent policies will be trained under the identical
world, reward amounts, action registry, horizon, seeds, and budget. Evaluation now also
records action distributions, carried-resource travel, homeward movement, camp opportunities,
deposit choices, and gathered-to-deposited conversion.

## Current Decision Gate

1. Train fresh feed-forward and recurrent PPO policies with the v4 team-objective board.
2. Evaluate both against legal random on identical held-out seeds and inspect the new
   inventory/return diagnostics.
3. Resume procedural maps and gameplay content only if at least one learned baseline beats
   random on the aggregate score and meaningful delivery/construction/tool achievements.
4. If neither learned baseline separates, factorize verb, argument, and target outputs and
   repeat the short comparison.
5. Add MAPPO and centralized training-only state only if evidence still points to a genuine
   multi-agent credit-assignment limitation.

Procedural generation, fishing, rescue, and longer episodes remain intentionally blocked
until this calibration produces useful baseline separation.

### V4 Outcome And Factorized-Action Test

The v4 paired pilot improved resource selection but not economic composition. Recurrent PPO
gathered food, wood, stone, and the complete workbench resource bundle in all held-out
episodes, yet neither learned policy returned wood or stone to camp, deposited either
resource, or began construction. Recurrent PPO selected `GIVE` roughly 521 times per episode
and usually died. Scores remained below legal random: random 0.119, feed-forward 0.033, and
recurrent 0.060.

This isolates two issues. First, the population still does not switch from exploration to
return/delivery. Second, the atomic 270-way actor gives targeted verbs many separate entries,
so a verb such as `GIVE` can dominate because it has many argument-target combinations.

The next controlled implementation factorizes only the reference PPO policy into verb,
legal argument conditional on verb, and legal target conditional on both. The environment
still receives the same flattened public action. PPO uses the summed component log
probability as one joint action probability. A 250K feed-forward run tests this representation
before spending time on factorized recurrence or MAPPO.

### Factorized Feed-Forward Outcome And Final Composition Test

Factorization corrected the action-multiplicity pathology: average `GIVE` use fell from about
521 per recurrent-v4 episode to below one, while the held-out gathered-bundle rate reached
95%. The score improved over atomic feed-forward PPO from 0.033 to 0.048 but remained below
legal random at 0.119. Only one of twenty held-out episodes deposited wood, no episode
deposited stone, development seeds contained no material deposits, no workbench was started,
and terminal survival averaged 0.8 agents.

The pilot runner exposed two measurement problems. It treated one deposit as sufficient
emergence, and camp-return diagnostics used distance zero even though camp actions are legal
within Manhattan distance one. Diagnostics now match the environment, and a behavior must
occur in at least 20% of both development and held-out episodes to count as repeatable.

The final controlled test combines factorized action heads with per-agent GRU memory for
250K transitions. If it creates repeatable delivery and meaningful separation above random,
the baseline is replicated across training seeds. If it fails, stop algorithm stacking and
simplify the task or interface. MAPPO is not the automatic next step.

## Technical-Report Framing

The useful story is not “PPO failed several times.” It is:

1. A compact multi-agent economy established a reproducible training/evaluation/replay
   pipeline.
2. Expanding the dependency chain exposed reward exploitation, policy symmetry, partial
   observability, delayed credit, and skill-composition problems hidden by the easier task.
3. Controlled diagnostics separated mechanics that are individually learnable from the
   unresolved end-to-end composition problem.
4. Evaluation was changed from a brittle binary gate to an achievement spectrum.
5. Feed-forward and recurrent baselines now test whether memory creates measurable
   algorithmic headroom before benchmark scope expands.

The final report must not claim Voyager is a validated benchmark until learned policies
meaningfully exceed random on the current island and later generalize to held-out procedural
islands.

## Fixed-Island Scope Tightening — Progression V3/V4

The checkpoint-selected two-agent island probe later produced the first meaningful partial
chain. Its selected 150K PPO checkpoint beat legal random (`0.048` versus `0.019`), gathered
all three resources in every held-out episode, deposited wood in 30%, deposited stone in 20%,
completed one workbench, and survived the first night in 75%. Raising the PPO entropy floor
to `0.005` improved this further: wood delivery reached 95%, stone delivery 40%, workbench
completion 15%, first-night survival 100%, and score `0.106` versus random `0.019`. The gate
still failed narrowly because its fixed thresholds required 50% stone delivery and 20%
workbench completion.

This changed the diagnosis. Voyager was no longer failing to start learning; it was failing
to compose stages reliably. Adding more algorithms or gameplay content would not address
that setup problem. The controlled response keeps the 48x48 island, two agents, 373-value
observation, nineteen actions, finite economy, achievements, procedural generator, threats,
ledger, replay, and public score, while tightening the intended dependency graph:

```text
workbench -> axe and spear -> campfire -> hunted/cooked meal
          -> shelter -> beacon -> 100-tick extraction -> rescue
```

Structure costs and labor were reduced, later construction branches are masked until their
prerequisite is achieved, duplicate team tools are prevented, and deer hunting now requires
the spear. The v3 training wrapper adds bounded individual credit only for deposits that fill
the current stage's unmet materials and for labor actually applied. Extraction has three
small, one-time survival milestones. None of these signals changes the reward-independent
achievement scorer.

The rescue condition now waits for both a 100-tick post-beacon extraction window and first-
night survival, then terminates early. This permits a strong policy to beat the game rather
than waiting arbitrarily for tick 1,200, while preserving the full horizon for slower
policies.

Mechanics were verified before another PPO experiment. Focused acceptance tests pass, the
canonical replay completes every achievement, and the public-observation oracle rescues on
96 of 100 held-out procedural islands with zero invalid actions, 100% joint survival, and a
`0.986` geometric-mean score. This proves mechanical solvability, not learned trainability.
The v3 250K run still failed. Its selected policy scored `0.053` versus random `0.019`, but
wood/stone delivery fell to `25%/5%` and workbench completion to `5%`. Diagnostics showed
that all new stage deposit and labor components contributed only about 13 reward units across
training, while health loss and deaths contributed roughly -1,831. Re-evaluating the earlier
v2 checkpoint on the tightened world preserved its behavior, so the mechanics were not the
regression.

V4 changed no mechanic or score. It increased only useful-deposit credit from `0.10` to
`0.50` per needed unit and applied-labor credit from `0.10` to `0.25` per ten labor. Both
remain finite and non-repeatable. The 250K v4 run selected its 200K development checkpoint
and passed every held-out gate: score `0.122` versus random `0.019`, food/wood/stone gathering
`100%/100%/95%`, wood/stone delivery `70%/75%`, workbench completion `50%`, tools `15-25%`,
campfire `5%`, first-night achievement `90%`, and invalid actions `0.29%`.

This closes the immediate setup problem: ordinary feed-forward PPO can now reliably enter
and advance the economy. It does not solve the full task, and procedural calibration or new
content should not be conflated with this fixed-island result.

## Procedural Baseline Runner

The next implementation converts procedural support into an auditable experiment protocol.
Feed-forward and recurrent PPO now accept a manifest-backed episode scheduler rather than
deriving every island from `training_seed + reset_count`. Each seed visits only the frozen
0-999 training split in deterministic shuffled cycles, and matching algorithm runs receive
matching island orders. Earlier trainers retain their original reset behavior when no
schedule is supplied.

The runner handles one algorithm/seed per command, evaluates 200K milestones only on the 50
development islands, and locks the best seeded-stochastic achievement score with invalid rate
and earlier milestone as tie-breakers. A separate suite finalizer is the only path that opens
the 100 held-out test islands, and it refuses to run until all six one-million-transition
selections are complete and contract-identical. Exact configs, episode seeds, histories, raw
episodes, checkpoint hashes, and artifact hashes are recorded. Tiny feed-forward and
recurrent smoke profiles validate the machinery but are explicitly excluded from benchmark
claims.
