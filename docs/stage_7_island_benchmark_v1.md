# Stage 7 — VoyagerIsland-v1 Benchmark Contract

## Status And Research Claim

`VoyagerIsland-v1` is the canonical Stage 7 reinforcement-learning environment. It evaluates
whether a learning method can train a decentralized two-agent policy to acquire a cooperative
capability chain in a finite stranded-island economy:

```text
explore -> gather -> return -> deposit -> build -> craft
        -> hunt -> cook -> shelter -> survive -> signal -> rescue
```

The environment, deterministic core, compact actions and observations, procedural generator,
achievement scorer, frozen seed splits, PPO adapters, public-observation scripted oracle, and
Replay 2.3 path are implemented. The seed-0 fixed-island 250K trainability gate has passed,
establishing early-economy separation from legal random. The benchmark is not scientifically
validated until the multi-seed procedural baseline and release gates described below pass.

Stage 7A and 7B remain supported. `VoyagerCivilization-v1` and
`VoyagerCivilization-v2` are advanced compatibility sandboxes, not the official benchmark.
The failed Stage 7C experiments on the broader ten-agent interface are preserved in
`technical_report_progress_notes.md` and the historical Civilization plan.

## Frozen Public Contract

| Contract | Version |
|---|---|
| Environment | `VoyagerIsland-v1` / `voyager_island_v1` |
| Scenario | `voyager_island_benchmark_v1` |
| Observation | `voyager_island_observation_v1` |
| Action | `voyager_island_action_v1` |
| Reward | `voyager_island_achievement_reward_v1` |
| Official PPO training reward | `voyager_island_progression_reward_v4` |
| Achievements | `voyager_island_achievements_v1` |
| Generator | `voyager_island_generator_v1` |
| Replay | `stage7_replay_2.3.0` |

The canonical interface is a PettingZoo `ParallelEnv`. An optional centralized Gymnasium
wrapper emits two actions but does not change simulation semantics. Agent submission order
must never change the state, events, rewards, conservation ledger, or replay state hash.

### Population, world, and horizon

- Two symmetric agents with a shared-policy-compatible two-value identity indicator.
- No productivity roles or private role bonuses.
- One 48x48 island, central camp, and fixed relative construction sites.
- 1,200 world ticks and four 300-tick day/night cycles.
- Night occupies ticks 200-299 of every cycle.
- One initial wreck ration per agent.
- Finite resources with no respawn or renewable economy.
- Episodes truncate at tick 1,200 and terminate early when both agents die or both are
  rescued.

Every generated island contains at least 35 wood, 20 stone, 20 berry units, two deer, and
reachable stalker spawn candidates. Food lies 4-6 path steps from camp, wood 5-8, stone 6-10,
and initial deer 8-12. Generator validation rejects unreachable or under-resourced islands.

### Nineteen actions

The one-dimensional action registry is:

1. `NOOP`
2. `MOVE_NORTH`
3. `MOVE_EAST`
4. `MOVE_SOUTH`
5. `MOVE_WEST`
6. `INTERACT`
7. `ATTACK`
8. `EAT`
9. `REST`
10. `DEPOSIT_ALL`
11. `WITHDRAW_FOOD`
12. `CRAFT_AXE`
13. `CRAFT_SPEAR`
14. `WORK_WORKBENCH`
15. `WORK_CAMPFIRE`
16. `WORK_SHELTER`
17. `WORK_BEACON`
18. `USE_CAMPFIRE`
19. `USE_SHELTER`

The environment exposes an authoritative nineteen-value legal mask. Invalid submissions
become no-op and receive an individual `-0.05` reward. Contested movement and over-subscribed
resource claims fail symmetrically. Swaps and valid movement cycles succeed.

### Observation

The flattened actor input has 373 normalized values:

- 7x7x7 egocentric tiles with terrain, resources, structures, creatures, agents, and light.
- Six self values: health, hunger, energy, inventory usage, sheltered, active.
- Five inventory values: food, wood, stone, raw meat, cooked meat.
- Axe and spear ownership.
- Two-value identity.
- Signed camp x/y bearing and Manhattan distance.
- Twelve-value public board: camp stock, four structure progress values, active fraction,
  normalized time, and night state.

It does not expose hidden entity IDs, food-lot IDs, remote locations, future spawns, or the
privileged simulator state. Structured observations are the official modality; pixels remain
an optional later extension.

## Reduced Economy

| Capability | Materials | Labor/effect |
|---|---:|---|
| Workbench | 3 wood, 1 stone | 20 labor |
| Axe | 2 wood, 1 stone | Passive double wood yield |
| Spear | 2 wood, 1 stone | Two attack damage instead of one |
| Campfire | 2 wood, 1 stone | 20 labor; cooks raw meat |
| Shelter | 4 wood, 2 stone | 40 labor; protects both occupants |
| Beacon | 4 wood, 2 stone | 60 labor; requires the other structures |

Each valid work action contributes ten labor. Materials reserve atomically from camp when
construction starts. Construction and crafting masks expose one coherent progression branch:

```text
workbench -> one team axe and spear -> campfire -> hunt and cook one meal
          -> shelter -> beacon -> 100-tick extraction -> rescue
```

This is a technology dependency graph, not a calendar script. Agents may gather, eat, move,
rest, and deposit whenever those actions are legal, but they cannot spend the finite economy
on later public structures before satisfying the preceding capability. A deer can only be
hunted with a spear. Deer have two health and create a two-unit raw-meat ground pile. Raw
meat is not edible; the campfire converts it to cooked meat. Food does not spoil and the
campfire does not require fuel in the official profile.

One or two seeded stalkers spawn every night at validated distant positions. They have three
health, move or attack on alternating ticks, and deal 25 damage per hit. Sheltered agents
cannot be targeted. Structure damage, repair,
downing, revival, tool transfer, equipment, pickaxes, torches, packs, fishing, markets,
communication, and governance remain disabled for v1.

Completing the beacon begins a 100-tick extraction window. `rescue_both` is awarded when the
window has elapsed, both agents remain alive, and the first night has been survived. A strong
policy can therefore finish early; slower policies retain the full 1,200-tick limit. This
makes time-to-rescue meaningful without allowing a beacon rush to skip night survival.

## Reward-Independent Evaluation

The fifteen achievements are:

1. Collect food.
2. Collect wood.
3. Collect stone.
4. Deposit wood.
5. Deposit stone.
6. Build the workbench.
7. Craft an axe.
8. Craft a spear.
9. Hunt a deer.
10. Build the campfire.
11. Cook meat.
12. Build the shelter.
13. Keep both agents alive through the first night.
14. Build the beacon.
15. Rescue both agents.

Every success rate is reported. The headline score is their smoothed geometric mean, with
separate gathering, delivery, production, survival, and rescue group scores. Evaluation also
reports joint survival, achievement timing, invalid actions, returns, remaining resources,
and ledger-derived contributions.

The public environment's v1 reward is bounded and deliberately simpler than the score:

- `+1.0` shared for the first completion of each achievement.
- `+0.1` individual on a tick with health restoration.
- `-0.1` individual on a tick with health loss.
- `-1.0` shared for every agent death.
- `-0.05` individual for an invalid submission or symmetric conflict.

There is no repeated gathering, navigation, per-step survival, deposit, or construction
shaping reward.

The first fixed-island PPO gate showed that this fully shared reward did not provide enough
causal credit. PPO learned food, wood, and partial first-night survival, but one agent made
all gathering interactions while the other free-rode on shared unlocks. It achieved zero
stone collection, deposits, or workbench completions and scored `0.014`, below legal
random's `0.023`.

The first remediation used the separately versioned
`voyager_island_trainability_reward_v2`. It preserved every v1 component and added only:

- `+0.50` to each recorded causal contributor when an achievement first unlocks. Causal
  actors come from deterministic gather, deposit, structure-completion, craft, hunt, and
  cook events; survival and rescue remain purely shared.
- `+0.20` to an individual material carrier the first time it reaches camp-distance 6, 3,
  and 1, for a maximum of `+0.60` per agent per episode.
- A deterministic seed-based swap between the two spawn-slot assignments during training,
  preventing permanent agent identity from becoming a fixed spatial shortcut.

These signals are training-only, bounded, nonrepeatable, and omitted from public scoring.
The map, mechanics, observation, action registry, achievements, evaluator, and gate thresholds
remain unchanged. The v1 reward remains directly constructible for checkpoint reproduction.

The first 250K run under v2 improved the stochastic score from `0.014` to `0.025`, slightly
above legal random's `0.023`. It also produced occasional stone collection and wood delivery,
but still missed the predeclared delivery, construction, and score-margin thresholds. Training
logs revealed a separate protocol problem: policy entropy fell to approximately `0.013` by the
end, and the final deterministic policy selected almost entirely no-op actions. Because only
the final checkpoint had been evaluated, that run could not establish whether an earlier,
more exploratory checkpoint had learned more useful behavior.

The remediation does not change the world, reward, policy architecture, or 250K budget. It
saves checkpoints at 50K, 100K, 150K, 200K, and 250K agent transitions. Each is evaluated
under both seeded-stochastic and deterministic inference on frozen development seeds. The
checkpoint with the highest stochastic public achievement score is selected, with lower
invalid-action rate as the sole tie-breaker. Only after selection is locked is that checkpoint
evaluated against legal random on held-out test seeds. The final checkpoint is also reported
as a diagnostic, but test results and deterministic results cannot select the model.

That checkpointed run selected the 150K policy and confirmed real end-to-end learning: on
twenty evaluation episodes it gathered all three resources in 100%, deposited wood in 30%
and stone in 20%, and completed the workbench once. Its score was `0.048` versus random's
`0.019`, but it still failed the predeclared delivery and workbench thresholds. The 250K
policy again collapsed primarily to no-op. Raising the entropy floor from `0.001` to `0.005`
then selected a substantially better 150K checkpoint. On twenty held-out episodes it gathered
food and wood in 100%, stone in 95%, deposited wood in 95%, deposited stone in 40%, completed
the workbench in 15%, and survived the first night in 100%. Its score was `0.106` versus
random's `0.019`. It narrowly missed the predeclared 50% stone-deposit and 20% workbench
thresholds. The evidence localized the remaining problem to unreliable stage composition
rather than basic exploration.

The setup remediation began with `voyager_island_progression_reward_v3`. It kept the same
48x48 map, two agents, 373-value observation, nineteen actions, finite resources, threats,
achievements, public scorer, checkpoint protocol, and 250K budget. It changes only the
dependency balance and training credit needed to make the intended chain legible:

- Structure costs and labor are reduced to the table above, while the complete chain still
  consumes 17 wood and 8 stone before mistakes.
- Later construction masks remain closed until the current technology prerequisite is met.
- At most one team axe and one team spear may be crafted, and the spear is required to hunt.
- A material depositor receives `+0.10` per unit that fills the current stage's unmet bundle.
  Excess deposits cannot regenerate reward.
- Applied construction labor receives `+0.10` per ten labor, split symmetrically between
  simultaneous contributors and bounded by finite required labor.
- Surviving agents receive shared `+0.25` milestones at 25, 50, and 75 extraction ticks.

These additions are training-only. Public evaluation continues to use only the fifteen
outcomes, so an algorithm cannot obtain a high benchmark score by farming shaping reward.
The v1, v2, and v3 rewards remain constructible for reproduction of the failed probes.

The v3 250K probe failed because its new progression components were too small to be salient:
stage deposits and labor contributed only about 13 reward units across training, compared
with roughly 1,831 negative units from health loss and deaths. Its selected held-out policy
scored `0.053` and completed the workbench in only 5% of episodes. Re-evaluating the earlier
v2 checkpoint on the tightened world retained its delivery behavior, ruling out the smaller
recipes and stage masks as the cause.

`voyager_island_progression_reward_v4` therefore changes only two bounded magnitudes:
currently useful deposits receive `+0.50` per unit and applied construction receives `+0.25`
per ten labor. It does not add a repeatable reward source. The selected 200K checkpoint from
the 250K v4 run passed the untouched held-out gate: score `0.122` versus random `0.019`,
food/wood/stone gathering `100%/100%/95%`, wood/stone deposits `70%/75%`, workbench completion
`50%`, first tool achievements `15-25%`, campfire completion `5%`, first-night achievement
`90%`, and invalid actions `0.29%`. This establishes reliable early-economy trainability; it
does not claim that feed-forward PPO solves the complete island.

## Splits, Baselines, And Gates

Frozen generator seeds:

- Training: 0-999.
- Development/checkpoint selection: 10000-10049.
- Held-out test: 20000-20099.

Test seeds must never select checkpoints or tune the world. Official learned baselines use
three independent training seeds and one million agent transitions each. Seeded-stochastic
inference is primary; deterministic argmax is a secondary collapse diagnostic.

Required comparators:

- Legal random.
- Public-observation scripted oracle.
- Shared feed-forward PPO.
- Shared recurrent PPO.

MAPPO is optional and is not a release dependency.

Before procedural training, run the 250K fixed-island gate with the v4 training reward.
Feed-forward PPO must gather each
basic resource in at least 80 percent of evaluation episodes, deposit wood and stone in at
least 50 percent, complete the workbench in at least 20 percent, keep invalid actions below
five percent, and exceed random's achievement score by at least 0.02.

This gate applies to the development-selected checkpoint, not automatically to the last
training update. Selection milestones and metrics are fixed before training, development and
test seeds remain disjoint, and the test split is never used for tuning or checkpoint choice.

The final release gate requires:

- The oracle solves at least 90 percent of held-out islands with a reconciled ledger.
- Legal random never rescues and rarely completes the workbench.
- At least one learned baseline exceeds random with a paired bootstrap 95 percent confidence
  interval excluding zero.
- At least one learned baseline completes a middle-tree capability in at least 25 percent of
  held-out episodes.
- At least one strong learned baseline achieves `rescue_both` on at least 25 percent of
  held-out islands. The benchmark is not beaten merely because a policy survives.
- All public versions, seed manifests, experiment configurations, curves, checkpoints, and
  raw episode summaries are published.

The public-observation oracle currently passes its gate on the frozen 100-island test split:
every achievement succeeds on at least 96%, `rescue_both` succeeds on 96%, joint survival is
100%, invalid actions are 0%, and the achievement geometric mean is `0.986`. These numbers
establish mechanical solvability and procedural coverage only; they are not evidence that
PPO learns the task.

If a gate fails, adjust only resource distance, difficulty, compact observation context,
reward magnitude, or action legality. Do not add content, silently increase the budget, tune
on test seeds, or substitute a more complex algorithm before diagnosing the public task.

## Replays And Release

Replay 2.3 records compact actions, achievement unlocks, beacon progress, rescue, creatures,
ground piles, agent tools, complete resource provenance, contribution accounting, state
hashes, and conservation totals. Older Replay 2.0-2.2 artifacts remain loadable.

The canonical oracle replay uses procedural seed 3, completes all fifteen achievements with
zero invalid actions, terminates after successful rescue, and reconstructs every recorded
state through the existing loader. The web viewer renders the beacon and exposes the
achievement timeline; it never simulates outcomes in the browser.

Stage 9 remains the public-package milestone: installable environment, API examples,
baseline commands, frozen checkpoints, benchmark tables, environment card, technical report,
and curated replays. Stage 8 and Stage 10 are optional extensions rather than blockers.
