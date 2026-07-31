# Stage 7A — Civilization Expansion and Handcrafted Vertical Slice

## 1. Outcome and Guiding Decisions

Stage 7A expands the existing Voyager island, simulation, RL interfaces, replay system, and
viewer into the first playable slice of the Civilization campaign. It is not a replacement
runtime and does not begin a separate project.

The guiding rule is:

> Stage 7A should look and behave like the current Voyager game made larger, more alive,
> more dangerous, and more cooperative.

The completed phase will provide:

- The current authoritative Python simulation extended in place.
- The current compact 300-step scenario retained as a regression and compatibility target.
- A deterministic 48x48 handcrafted island and ten-agent, two-day Civilization scenario.
- Versioned extensions to the current actions, observations, masks, and privileged state.
- A workbench, spear crafting, campfire, cooking, shelter occupancy, huntable animals,
  night stalkers, joint construction, and joint defense.
- One or two night stalkers selected and placed through seeded randomness at the start of
  each night.
- Dangerous monster attacks that make an exposed agent, shelter, fire, and group defense
  mechanically consequential.
- Additive `stage6_replay_2.1.0` support while existing Stage 6 recordings continue to load.
- Existing viewer assets and behavior retained and extended with the new entities, lighting,
  structures, tools, and events.
- A deterministic five-minute scripted replay: 600 ticks at two ticks per second.

Stage 7A does not produce the official Civilization score, procedural campaign, ecology,
rescue objective, or learned-policy result. Its purpose is to prove the expanded gameplay
loop and the compatibility of the existing Voyager stack before later Stage 7 work scales it.

### Verified starting point

The `stage6-complete` Git tag points to the clean Stage 6 revision at `4a8826d`. It is a
lightweight historical bookmark, not an archive branch or alternate runtime.

Before implementation, record the current baseline without moving or deleting artifacts:

- Python tests and optional-training skips.
- Ruff and mypy results.
- Frontend tests, TypeScript checking, and production build.
- Deep validation of the existing curated replays.
- Stage 5.6 reference and checkpoint hash verification.

Any mismatch should be understood before Stage 7A mechanics are introduced, but Stage 7A
does not require a dedicated archive commit, isolated historical worktree, remote tag gate,
or full benchmark rerun before development can begin.

## 2. Expansion and Compatibility Strategy

### 2.1 Systems that remain in place

Stage 7A preserves and builds on:

- `voyager/sim/multi_world.py`, `state.py`, `mapgen.py`, `constants.py`, `rewards.py`, and
  `achievements.py`.
- The current single-agent and PettingZoo parallel environment registrations.
- Existing heuristic and PPO policy code, training utilities, checkpoints, evaluation code,
  and benchmark runner.
- Existing actions and observations for the compact Stage 5/6 scenario.
- The replay recorder, schema, loader, adapters, catalog, CLI, server, camera, and viewer.
- Current terrain, resource, camp, shelter, role, weather, and agent visuals.
- Existing tests, examples, manifests, reference results, and curated replays.

Working components may be generalized or split into focused modules when necessary, but
they are not removed simply to make a clean Stage boundary.

### 2.2 One authoritative simulation

There must be one simulation path:

```text
current Voyager world
  -> generalized map and state support
  -> additional items, structures, and creatures
  -> compact or Civilization scenario configuration
  -> existing environment, policy, replay, and viewer integrations
```

Do not create a parallel `progression_world.py` or a second implementation of movement,
survival, gathering, roles, camp storage, storms, shelter, rewards, or action resolution.
`multi_world.py` remains the world orchestrator. New modules should contain genuinely new
mechanics or extracted logic shared by every scenario.

Likely incremental modules are:

```text
voyager/sim/
  items.py
  recipes.py
  structures.py
  creatures.py
  time.py
  combat.py
  registries.py
  scenarios.py
```

The exact extraction should follow the current code rather than forcing this layout before
implementation begins.

### 2.3 Compact-scenario compatibility

The existing 300-step world remains a supported configuration of the shared engine. When
the compact scenario is selected:

- Existing mechanics retain their meaning.
- Existing action and observation contracts remain available.
- Existing policies and checkpoints continue to use their declared contracts.
- Current regression tests remain active.
- No Stage 7A creature or structure must appear unless enabled by the scenario.

This is scenario compatibility, not a frozen duplicate simulator.

## 3. Stage 7A Contracts

### 3.1 Versioning approach

Stage 7A records every interface version in environment manifests and replays. The intended
identifiers are:

| Contract | Stage 7A identifier |
| --- | --- |
| Handcrafted scenario | `voyager_civilization_vertical_slice_v1` |
| Structured actions | `structured_actions_v1` |
| Flattened actions | `flattened_actions_v1` |
| Local observation | `structured_local_v1` |
| Privileged state | `civilization_global_state_v1` |
| Replay | `stage6_replay_2.1.0` |

`VoyagerCivilization-v1` may be registered at the end of Stage 7A if the vertical slice has
stabilized. If registered, it must wrap the same authoritative simulation as the existing
environments; it is a versioned configuration and interface, not a different game engine.

Only Stage 7A mechanics receive runtime registry entries. Storage, fishing nets, rescue,
revival, future tools, and other later-phase mechanics remain documented future work rather
than inactive placeholders permanently occupying the 7A action space.

### 3.2 Registries

Add typed specifications for the new Stage 7A concepts while preserving existing numeric
and string IDs:

- Terrain additions needed by the handcrafted island, such as beach, forest, rocky
  highland, and cave.
- Items: existing food, wood, and stone plus wreck ration, raw meat, and cooked meat where
  not already represented.
- Tool: spear.
- Recipes: spear and cooked meat.
- Structures: existing camp and shelter plus workbench and campfire.
- Creatures: island deer and night stalker.
- New actions, events, reward components, achievements, and metrics used in 7A.

New IDs are appended and never change the meaning of existing IDs. Validate uniqueness and
reference integrity at import or test time. Freeze the Stage 7A registry fingerprint when
the scripted replay is finalized, not before the mechanics have been playtested.

### 3.3 Actions

The Civilization scenario needs the following conceptual verbs:

```text
NOOP, MOVE, INTERACT, EAT, REST, DEPOSIT, WITHDRAW,
CRAFT, WORK, USE, ATTACK, DEFEND
```

Arguments identify a direction, item, recipe, structure, tool, or visible target slot.
Target slots are derived from the pre-step local observation and have deterministic ordering.

Implementation requirements:

- Preserve the current compact-scenario action contract.
- Append or version new action meanings instead of renumbering existing actions.
- Provide a structured PettingZoo-compatible form for the Civilization scenario.
- Provide a deterministic flattened wrapper and legal-action mask for later PPO use.
- Reject malformed action objects clearly.
- Convert well-formed but currently illegal actions into a recorded invalid action and
  deterministic no-op.
- Derive and lock the exact flattened-action count after the implemented 7A pairs are known.

Stage 7A must not reserve flattened actions for unimplemented later-stage mechanics.

### 3.4 Local observations and privileged state

Extend the current observation pipeline with a versioned local observation containing:

- A 7x7 local tile window with terrain, resource, structure, creature, visible-agent role,
  and recorded local-light channels.
- Health, hunger, energy, inventory load, shelter occupancy, and alive status.
- Stage 7A item inventory and spear ownership/equipment.
- Existing role information.
- Day, phase, phase progress, and ambient light.
- Locally available camp, campfire, workbench, and shelter information.
- Deterministically ordered visible target slots.
- The legal-action mask for the declared action form.

No exact global resource distances, hidden agents, or hidden creatures appear in actor
observations.

Expose a separately versioned `global_state()` snapshot for scripted validation, replay
recording, debugging, and later centralized critics. It must never be embedded in a
decentralized actor observation.

Observation and action spaces remain stable throughout an episode. If the expanded shape
cannot preserve a current contract, introduce a new version and retain the old version for
its compact scenario rather than silently changing checkpoint inputs.

## 4. Handcrafted Vertical-Slice World

### 4.1 Larger island

Add a committed 48x48 handcrafted blueprint using the existing map-generation and terrain
pipeline. The map must feel like an expansion of the current island and reuse its visual
language.

It contains:

- A connected island with a water border and navigable beach.
- A wreck and camp near the center-south.
- Workbench, campfire, and shelter construction sites near camp.
- Berry-bearing terrain near the early safe area.
- A larger forest region with wood and deer habitat.
- Rocky highland with stone.
- Multiple cave, dense-forest, or perimeter tiles eligible for night-stalker spawning.
- Traversable routes between every required progression resource and the camp.
- Enough open space around camp for legible construction and defense.

Validate dimensions, legend, required sites, ten spawn locations, resource sufficiency,
reachability, and monster-spawn eligibility during reset.

The scenario seed controls tie-breaking, deer behavior, and night-spawn selection. The map
geometry itself remains fixed for this vertical slice.

### 4.2 Population and starting state

- Ten agents begin near the wreck using the existing role system.
- Roles continue to modify efficiency without hard-locking actions.
- Existing forager, woodcutter, and builder behavior is retained.
- The camp begins with ten wreck rations.
- Existing inventory-capacity and resource-accounting rules remain the baseline unless a
  documented Stage 7A mechanic requires an additive change.

### 4.3 Time and survival

The two-day slice uses 300 ticks per day:

```text
ticks 0-99: morning
ticks 100-199: afternoon
ticks 200-299: night
day = tick // 300 + 1
```

The replay records every phase transition, including dusk, night, and dawn. Time affects
recorded lighting, campfire value, shelter value, creature behavior, and safe work schedules.

Initial balance targets remain subject to playtesting:

- Hunger rises continuously and high hunger damages health.
- Movement, gathering, construction, and combat consume energy.
- Exposed rest restores less energy than resting near a lit campfire.
- Shelter rest is strongest.
- Berries and wreck rations provide immediate food.
- Raw meat is edible but inefficient and harmful.
- Cooked meat is the strongest Stage 7A food.

Downing and revival are not introduced in Stage 7A. Zero health causes death using the
existing termination semantics.

## 5. Progression Mechanics

### 5.1 Public projects

Stage 7A adds three projects to the current camp-and-shelter foundation:

| Structure | Initial materials | Initial labor | Effect |
| --- | ---: | ---: | --- |
| Workbench | 6 wood, 2 stone | 240 | Enables spear crafting |
| Campfire | 4 wood, 4 stone | 160 | Cooking, light, recovery, and monster deterrence |
| Shelter | 12 wood, 6 stone | 600 | Six protected occupants and improved rest |

These values are starting balance constants, not permanent benchmark values.

The current building system should be generalized rather than replaced. When the first
valid work batch begins, required materials are atomically reserved from camp. Labor applies
only after the complete material cost has been reserved.

Multiple same-tick contributors receive a modest joint-work multiplier capped at 1.5x.
Record contributor IDs, roles, raw labor, multiplier, and applied labor. The joint-
construction achievement requires at least two agents with different roles.

Completed structures have recorded condition. General structure damage and repair remain
outside 7A.

### 5.2 Spear, hunting, and cooking

- A spear initially costs two wood and one stone.
- Crafting requires adjacency to a completed workbench.
- Inputs are removed atomically from the crafter and permitted camp stock.
- The spear is privately owned and must be equipped before use.
- Tools do not have durability in 7A.
- Deer flee nearby agents using deterministic seeded movement.
- A spear can kill a deer through the configured attack interaction.
- A successful hunt produces exactly two raw-meat units with an explicit source event.
- Cooking at a lit campfire converts one raw meat into one cooked meat.
- All creation, transfer, conversion, consumption, and loss reconcile in the resource ledger.

Animal regeneration, migration, and ecological sustainability remain outside 7A.

### 5.3 Campfire

- Fueling the campfire consumes camp wood and adds a fixed number of fuel ticks.
- Fuel has a tested cap and decrements once per active tick.
- A lit fire provides visible light and improved recovery within a recorded radius.
- Night stalkers normally refuse to enter protected fire-radius tiles.
- The fire does not make the entire island or every exposed worker safe.
- Maintaining fire through a complete night unlocks a Stage 7A achievement.

### 5.4 Shelter occupancy

- An adjacent agent may enter shelter while capacity remains.
- Occupancy persists only for no-op, rest, or another shelter-use action.
- Moving, working, crafting, attacking, or defending exits shelter before resolution.
- Occupants cannot be selected as normal night-stalker targets.
- Shelter occupants receive the strongest rest rate.
- Entry, exit, occupancy, rejection, and protection are recorded explicitly.

The initial shelter capacity is six, deliberately leaving part of the ten-agent population
responsible for fire maintenance and camp defense.

## 6. Night Stalkers and Cooperative Defense

### 6.1 Seeded-random nightly spawning

Night stalkers do not appear at one hard-coded coordinate or as a fixed single spawn.
At the transition into each night:

1. Use the scenario RNG to choose a spawn count of one or two with equal initial probability.
2. Build the eligible spawn set from cave, dense-forest, and designated island-perimeter
   tiles.
3. Exclude tiles inside the active campfire radius, immediately adjacent to camp, occupied
   by an agent, visible as a guaranteed unavoidable attack, or unreachable from exposed
   areas.
4. Select spawn tiles without replacement using the seeded RNG.
5. Record the candidate-set fingerprint, RNG outcome, chosen count, and spawn coordinates.

The same scenario seed and joint-action sequence must always produce identical spawn counts,
locations, events, and state hashes. Different seeds should produce visibly different but
valid night pressure.

If only one valid spawn tile remains, spawn one stalker and record the reduced count. Map
validation should normally guarantee at least two eligible tiles.

### 6.2 Stalker threat level

Initial night-stalker balance:

- Six health.
- Movement every two ticks using deterministic shortest-path pursuit.
- Target the closest visible exposed agent, with seeded rotating tie-breaking.
- Avoid protected campfire tiles.
- Deal **25 health damage** on a successful adjacent attack.
- Retreat and be removed at dawn.

With 100 maximum health, four unmitigated hits are lethal, and existing hunger or prior
damage can reduce that margin. One stalker is a serious threat to an isolated agent; two
stalkers can overwhelm an unprepared exposed group. The goal is pressure that visibly
justifies shelter, fire, spears, and coordinated defense—not random unavoidable deaths.

The 25-damage starting value must be tuned only if scripted and adversarial tests show that
prepared defense is either trivial or consistently impossible. Any change must preserve the
design target above.

### 6.3 Defense

- `DEFEND` targets an adjacent stalker.
- Each defender initially supplies eight points of damage reduction to adjacent allies
  against that stalker for the current tick.
- Two or more agents defending against the same stalker stagger it and suppress its attack
  for that tick.
- Spear attackers can damage stalkers while other agents defend.
- Emit a `joint_defense` event containing defender IDs, roles, target, prevented damage,
  and whether the stalker was staggered.
- Defeating a stalker while a different agent is actively defending unlocks the ally-defense
  achievement.

Defense values are resolved before stalker damage and use integer arithmetic. Campfire
deterrence, shelter protection, and defense must compose deterministically.

## 7. Deterministic Resolution

Every tick follows one documented phase order:

1. Snapshot legal actions and target-slot bindings.
2. Decode and validate actions and reserve scarce costs.
3. Resolve movement intents using deterministic rotating initiative.
4. Resolve gathering, deposits, withdrawals, eating, resting, equipping, crafting, and
   structure entry or exit.
5. Aggregate public work, attacks, and defense.
6. Update deer and stalkers, including pursuit and attacks.
7. Advance fuel, survival pressure, time, achievements, metrics, and rewards.
8. Produce the canonical snapshot, state hash, and ordered replay events.

New resolution logic should extend the current transition loop or be extracted from it for
shared use. It must not create a Civilization-only duplicate of existing movement or economy
resolution.

Contested movement uses rotating initiative so a permanent agent ID is not privileged.
Swaps and movement chains may remain unsupported during 7A if they are rejected consistently
and recorded clearly.

Canonical snapshots sort entity collections by stable ID and include authoritative RNG
state. Identical configuration, seed, and joint actions must yield identical events and
state hashes.

## 8. Rewards, Achievements, and Metrics

Preserve existing reward components and achievements. Add only the Stage 7A concepts needed
to explain and train the vertical slice:

- Tool progression.
- Food preparation.
- Public infrastructure.
- Joint work.
- Defense and prevented damage.
- Invalid actions.

Stage 7A population achievements are:

- Workbench completed.
- First spear crafted.
- First successful hunt.
- First cooked meal.
- Campfire maintained through a complete night.
- Shelter completed.
- Full shelter protected through a complete night.
- Joint construction by multiple roles.
- First stalker defeated.
- First stalker defeated while defending another agent.

Do not register future rescue, fishing, ecology, storage, revival, or signal achievements as
inactive 7A placeholders.

Dense, achievement, and no-reward modes must produce identical world transitions. Named
reward components, achievements, and metrics are recorded even when the selected reward mode
returns zero.

The vertical slice is a reachability and integration demonstration, not an official
benchmark result or evidence of emergent cooperation.

## 9. Replay and Viewer Expansion

### 9.1 Additive replay support

Extend the existing recorder and replay v2 schema rather than cutting them over to a new
system. Replay 2.1 adds:

- Day, phase, phase progress, and ambient/local light.
- Generalized item stockpiles.
- Workbench, campfire, and shelter progress, occupancy, fuel, and condition.
- Creatures and equipped spears.
- Targeted structured actions.
- Contributor lists for work and defense.
- Creature spawn count and coordinates, flee, pursuit, attack, prevented damage, defeat,
  and dawn-retreat events.
- Registry fingerprint and privileged-interface classification.
- Terminal structure, hunt, monster, survivor, and state-hash summaries.

The loader continues accepting existing replay 2.0 manifests. Missing Stage 7A collections
default to empty, unknown minor fields are preserved, and unknown major versions fail
clearly. Replay reconstruction remains simulation-independent.

### 9.2 Viewer expansion

The viewer keeps the current layout, controls, camera, sprites, animation language, terrain,
resources, camp, shelter, roles, and agent art. Stage 7A adds coherent assets or temporary
in-style visuals for:

- Workbench and campfire construction and completion.
- Lit and unlit fire states and the recorded protection radius.
- Spear ownership and equipment.
- Deer movement and hunting.
- One or two simultaneously active night stalkers.
- Night lighting.
- Shelter occupancy.
- Stalker attacks, prevented damage, stagger, defeat, and retreat.
- Joint-work and joint-defense events.

The browser renders recorded facts and never invents time, movement, combat, targeting,
fuel, protection, or random spawning. Final Civilization art, analytics panels, and campaign
polish remain Stage 7G work, but the vertical slice must already feel like an expansion of
the existing game rather than a generic fallback renderer.

## 10. Scripted Five-Minute Artifact

Create `civilization_vertical_slice_script_v1` as a privileged scripted policy:

- It may use `global_state()` for navigation and deterministic task assignment.
- It submits only public environment actions.
- It never mutates world state, teleports agents, grants resources, selects monster RNG
  outcomes, or invokes mechanics directly.
- It uses state predicates and pathfinding rather than injecting a fixed state sequence.
- Its privileged status is explicit in replay metadata.

Narrative progression:

- Morning 1: agents gather existing food, wood, and stone and return materials to camp.
- Afternoon 1: agents jointly finish the workbench, craft and equip spears, construct and
  fuel the campfire, hunt a deer, cook meat, and begin shelter improvements.
- Night 1: six agents occupy shelter while exposed agents maintain the fire and defend
  against the naturally seeded one-or-two-stalker spawn.
- Day 2: agents finish or replenish infrastructure, food, and fuel.
- Night 2: the completed camp handles a second independently sampled one-or-two-stalker
  spawn through shelter, fire, and coordinated defense.

Select and record a showcase seed whose natural deterministic RNG schedule demonstrates the
mechanics clearly. Prefer a seed that produces one stalker on one night and two on the other,
but do not alter the spawn result inside the script to achieve that narrative.

The committed artifact contains:

- 600 world ticks.
- A playback rate of two ticks per second.
- Ten starting agents and a recorded final survivor count.
- Completed workbench, campfire, and shelter.
- At least one spear craft, hunt, cooked meal, full-night fire, full shelter occupancy,
  joint multi-role work event, joint-defense event, and defended stalker defeat.
- Both nightly RNG spawn decisions and all monster damage recorded.
- No invalid scripted actions.
- Deep replay validation and deterministic regeneration checksums.

Do not require ten survivors at all costs. Prepared scripted play should normally preserve
the population, but the artifact must not weaken monster damage, override randomness, or
silently repair mistakes to guarantee the outcome.

## 11. Test Plan

### 11.1 Stage 5/6 regression

- The compact scenario still resets, steps, and terminates as declared.
- Existing actions retain their meanings and masks.
- Existing policy, PPO, checkpoint, evaluation, and benchmark imports remain available.
- Existing replay fixtures and curated recordings still validate and render.
- Existing viewer visuals remain stable for known Stage 6 entities.
- The complete pre-Stage-7 test suite remains active unless a test is deliberately updated
  to assert the generalized shared behavior.

### 11.2 Contracts

- Existing and new registry IDs are unique and stable.
- Every Stage 7A recipe, action argument, event, achievement, and structure reference resolves.
- Compact and Civilization action/observation versions are explicit.
- Structured and flattened actions round-trip exactly.
- Canonical and flattened masks are equivalent.
- Spaces remain stable during an episode.
- No later-stage placeholder actions are exposed.

### 11.3 Simulation

- Morning, afternoon, night, and dawn boundaries are exact.
- Blueprint dimensions, required sites, resources, routes, and eligible spawn regions validate.
- Same seed and joint actions produce identical events, RNG outcomes, and state hashes.
- Different seeds exercise both one- and two-stalker nights across a seed sweep.
- Spawn tiles are unique, reachable, outside protected or invalid areas, and selected without
  replacement.
- Work reserves exactly the declared materials.
- Joint-work arithmetic and contribution recording are exact.
- Workbench completion gates spear crafting.
- Crafting conserves materials and private ownership.
- Deer flee deterministically and hunting produces exactly two raw meat.
- Cooking performs a one-for-one conversion.
- Campfire fuel, light, recovery, radius, and deterrence are exact.
- Shelter entry, exit, capacity, persistence, rest, and targeting protection are exact.
- Each unmitigated stalker attack removes exactly 25 health.
- One defender reduces damage by exactly eight; two valid defenders stagger and suppress the
  targeted stalker's attack.
- Two stalkers resolve independently without double-consuming actions or corrupting target
  slots.
- Stalkers path, avoid fire, attack, take damage, die, and retreat correctly.
- Resource ledgers reconcile inventory, camp stock, projects, consumption, cooking, and
  generated animal output.

### 11.4 Environment and replay

- PettingZoo ParallelEnv compliance passes for supported environments.
- Observations are members of declared spaces at reset and every step.
- No hidden creature or resource data appears in local observations.
- Legal-action masks never mark an illegal pair legal.
- Dense, achievement, and no-reward modes preserve identical world transitions.
- Replay 2.0 artifacts continue to reconstruct.
- Replay 2.1 reconstructs the exact state hash at every tick.
- Night spawn counts, spawn locations, attacks, mitigation, defeats, and retreat events
  reconstruct exactly.
- Corrupt, partial, missing, and checksum-invalid artifacts fail clearly.
- Re-recording the scripted slice is byte-identical except for explicitly excluded metadata.

### 11.5 Frontend and end-to-end

- Existing Stage 6 replay visuals remain valid.
- New terrain, structures, items, tools, and creatures retain their recorded IDs.
- One and two simultaneous stalkers render and update independently.
- Night overlay, fire radius, occupancy, attack, stagger, damage, and defeat states render.
- The viewer can play, pause, scrub, and seek across both night boundaries.
- Agent inspection reports health changes, shelter state, equipped spear, and contributions.
- Frontend unit tests, TypeScript checking, production build, and browser smoke verification
  pass.

## 12. Stage 7A Acceptance Gate

Stage 7A is complete only when:

- `stage6-complete` identifies the clean pre-expansion milestone.
- The current simulator has been expanded rather than replaced.
- There is one authoritative world transition path for compact and Civilization scenarios.
- Existing environment, policy, PPO, training, benchmark, replay, server, and viewer code
  remains available and covered by regression tests.
- The compact scenario and existing replays still run through their declared interfaces.
- The 48x48 handcrafted scenario is reachable and deterministic.
- Each night samples and records one or two stalkers through seeded randomness.
- Stalker attacks are dangerous, deterministic, preventable through preparation, and not
  arbitrary unavoidable damage.
- Workbench, spear, hunting, cooking, campfire, shelter, joint work, and joint defense form
  one coherent visible progression.
- The committed five-minute replay satisfies its mechanical and narrative assertions.
- Python tests, Ruff, mypy, frontend tests, TypeScript checking, production build, clean
  installation, replay validation, and browser verification pass.

## 13. Explicit Stage 7A Boundaries

- The handcrafted map is fixed; procedural 48x48 generation remains Stage 7C.
- Existing storms remain supported, but Stage 7A does not add new weather systems or
  structure damage.
- Animal regeneration and ecological sustainability remain Stage 7C.
- General item transfers, spoilage, damage and repair, downing, and revival remain Stage 7B.
- Storage, cooking rack, fishing net, rescue signal, final defense, and rescue remain Stage 7D.
- PPO retraining, curriculum runs, and recurrent policies remain Stage 7E.
- Official manifests, benchmark scores, confidence intervals, and benchmark claims remain
  Stage 7F.
- Final art, campaign analytics, comparative panels, and curated benchmark runs remain Stage
  7G.
- No natural-language agent layer, market, governance, or unrelated platform work is part
  of Stage 7A.
- The vertical slice is a development artifact and reachability demonstration, not a
  benchmark result.
- Python remains authoritative; every frontend effect comes from recorded state.
