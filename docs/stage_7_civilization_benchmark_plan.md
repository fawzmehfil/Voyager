# Stage 7: Voyager Cooperative Planning Benchmark

## Decision And Status

Stage 7 turns the existing Voyager project into an auditable cooperative multi-agent
reinforcement-learning benchmark. It expands the current simulator, environment APIs,
training stack, replay format, viewer, tests, and visual identity in place. It does not
create a second game or retire the compact Stage 5/6 benchmark.

The research-facing position is:

> Voyager is an auditable cooperative MARL benchmark for learning decentralized planning,
> resource allocation, public investment, and recovery in a bounded stranded-island
> economy.

The central experiment is:

> Under the same interaction budget, which RL algorithm can most reliably convert finite,
> spatially distributed resources into population survival and rescue on unseen islands?

The lightweight Git tag stage6-complete remains the historical bookmark before the
expansion. The compact 300-step environment, checkpoints, benchmark results, saved
replays, and viewer compatibility remain supported through the shared codebase.

Current implementation status as of 2026-08-04:

| Phase | Status | Result |
|---|---|---|
| 7A | Implemented and committed | Handcrafted 48x48 island, ten agents, two days, workbench, campfire, shelter, hunting, cooking, stalkers, Replay 2.1 |
| 7B | Implemented and committed | Deterministic intent resolution, owned tools, transfers, food lots, spoilage, piles, damage, repair, downing, revival, ledger, Replay 2.2 |
| 7C trainability slice | Achievement scorer and recurrent PPO implemented; calibration runs pending | Failed probes preserved, primitive capabilities verified, fifteen-achievement handcrafted scorer, feed-forward re-score path, and decentralized shared-GRU PPO |
| 7C procedural substrate-7G | Blocked on baseline separation | Procedural islands, rescue economy, scaled MARL baselines, frozen benchmark, final viewer |

Stage 7B is released with a canonical Replay 2.2 demonstration that completes revival
through public actions. The first Stage 7C run completed 2,000,009 transitions and the v2
and v3 remediations each completed roughly 250,000 transitions. All proved the runtime fast enough but failed
their learnability gates. The learning ladder then demonstrated that gathering, returning,
construction, and basic survival are individually learnable. Procedural generation remains
blocked while the fixed achievement scorer tests whether feed-forward or recurrent PPO can
compose those capabilities on the complete current island.

## Brutal Usefulness Verdict

This design can produce a genuinely useful niche benchmark. It is not yet a validated
benchmark, and no document should claim that researchers will adopt it or that current
policies expose the intended capability gap.

The design is credible because:

- Cooperative credit assignment, partial observability, task allocation, memory, and
  procedural generalization are active MARL problems.
- Stage 7A and 7B already provide real inter-agent dependencies, deterministic resolution,
  and unusually exact resource and contribution accounting.
- The final task can compare feed-forward decentralized learning, recurrent decentralized
  learning, and centralized-training/decentralized-execution methods under one fixed
  protocol.
- The bounded island economy connects individual production, resource transfers, public
  infrastructure, repeated shocks, recovery, and a terminal population outcome.

The design fails as a benchmark if any of the following remain true at Stage 7F:

- Learned policies cannot progress meaningfully beyond random or greedy baselines within
  the maximum interaction budget.
- Disabling transfers or public infrastructure barely changes the relevant outcomes.
- A timestep-only policy achieves substantial success without reacting to observations.
- Procedural held-out performance collapses because policies memorized coordinates.
- Learned methods are statistically indistinguishable from random or greedy baselines.
- Reproducing one baseline requires unreasonable compute or undocumented intervention.
- Recurrent or centralized-critic policies do not materially improve interdependent
  capabilities over feed-forward or independent policies on unseen islands.

If a validity gate fails, simplify or rebalance the environment. Do not respond by adding
more content, extending training without limit, or weakening the scientific claim after
seeing final results.

## Scope Of The Claim

Voyager v1 evaluates cooperative planning and capability acquisition in a fully cooperative
Dec-POMDP with shared reward and decentralized execution.

It can support research on:

- Cooperative credit assignment.
- Partial observability and memory.
- Multi-agent exploration and spatial task allocation.
- Parameter sharing and role-conditioned policies.
- Centralized critics with decentralized actors.
- Hierarchical and model-based MARL.
- Resource efficiency and reward-hacking audits.
- Offline trajectories and contribution analysis.

Voyager v1 does not claim to measure:

- General social intelligence.
- Communication, negotiation, honesty, or human compatibility.
- Self-interested public-goods dilemmas or economic institutions.
- Markets, governance, or emergent civilization.
- Open-ended Minecraft-scale competence.
- Long-running societies or persistent worlds.
- LLM-agent behavior.

The environment identifier may retain Civilization for compatibility, but civilization is
not the scientific claim.

## One Project And One Simulation

Voyager continues to have one authoritative Python simulation.

    compact Stage 5/6 scenario
        |
        v
    shared map, agents, resources, survival, camp, and replay systems
        |
        v
    Stage 7 scenario configuration and deterministic v2/v3 resolution

Do not create a copied legacy project or a parallel simulator that reimplements movement,
survival, gathering, structures, rewards, or recording. New interfaces are versioned over
the same engine when compatibility requires them.

Stage 7 may extract focused modules for deterministic resolution, procedural generation,
benchmark scoring, policies, and replay adapters. Those modules must remain parts of the
shared runtime rather than independent games.

## Official V1 Task

### Population And World

- Ten agents.
- One 48x48 procedural island.
- Shared-policy parameterization with visible soft roles.
- Simultaneous actions with symmetric deterministic conflict resolution.
- Structured local observations as the official modality.
- Privileged fixed-size state available only to centralized critics and diagnostics.
- No explicit communication channel in v1.
- One shared cooperative reward contract.

Ten agents are retained because the benchmark concerns population-level allocation and
because Stage 7A/7B already implement and display this population. Parameter sharing keeps
the learned system tractable. Population variation is not part of the official v1
distribution.

### Horizon

The official design target is eight complete day/night cycles:

- 300 world ticks per cycle.
- 200 daytime ticks.
- 100 nighttime ticks.
- 2,400 world ticks per episode.
- Up to 24,000 agent transitions per complete ten-agent episode.

The current 600-tick Stage 7A/7B scenario is a two-day development demonstration, not the
official benchmark horizon.

Eight cycles are sufficient for the intentionally compact progression graph: they provide
eight recurring threat and recovery periods, hundreds of ticks between investment and
payoff, and a sustained final rescue window. The benchmark does not need the 10,000-step
depth of a much larger technology tree or the tens of thousands of steps of a multi-floor
dungeon.

The 2,400-tick target is changed only if the predeclared Stage 7E calibration gates show
that the scripted oracle cannot solve a balanced world, that learned policies receive too
few useful learning opportunities, or that the task saturates. Costs and timing should be
rebalanced before extending the horizon.

The horizon creates a real sample-efficiency risk. Two, five, and ten million agent
transitions equal only about 83, 208, and 417 complete ten-agent episode equivalents.
Intermediate achievements therefore must provide learnable signals before the final
rescue outcome. Those episode counts are not evidence that PPO will learn; Stage 7C and 7E
must measure it.

### Expected Phases Are Not Scripts

A successful population will often show three broad phases:

| Period | Pressure | Likely competent response |
|---|---|---|
| Days 1-2: Bootstrap | Limited wreck food and no infrastructure | Gather, complete the workbench, make basic tools, establish fire |
| Days 3-5: Development | Spatially separated resources, spoilage, and repeated nights | Specialize, transfer tools and food, hunt, cook, build shelter, invest in fishing |
| Days 6-8: Rescue | Deadline, continued consumption, attacks, and signal maintenance | Complete, fuel, defend, and repair the beacon while preserving the population |

These phases describe an intended strategic arc. The environment never commands agents to
hunt on a particular day or locks ordinary actions behind the calendar.

Agents choose every action. They may build shelter early, delay tools, skip fishing, rush
the signal, hoard resources, or fail. Ordering should arise from recipes, geography,
scarcity, labor requirements, delayed returns, night pressure, and the final deadline.

Only external events use fixed time:

- Day and night.
- Seeded creature spawning.
- Food expiry.
- The final rescue window.

A timestep-only policy must fail on procedural islands and stochastic threat placements.

## Difficulty Target

The benchmark must be easy to begin and difficult to complete.

- Days 1-2 are forgiving enough for exploration and early production.
- The middle period becomes difficult without tools, cooked food, shelter, transfers, and
  sensible public investment.
- The final period remains survivable, but maintaining the beacon while handling food,
  attacks, damage, and repairs requires population-level allocation.

The intended calibration is:

| Policy | Intended result |
|---|---|
| Random legal actions | Occasionally unlocks early outcomes, usually collapses, essentially never rescues |
| Greedy individual survival | Keeps some agents alive for several days, underinvests in shared infrastructure, rarely rescues |
| Feed-forward parameter-sharing PPO/IPPO | Learns early production and sometimes reaches later infrastructure |
| Recurrent PPO or MAPPO | Has meaningful headroom through memory, allocation, and centralized credit estimation |
| Cooperative scripted oracle | Rescues a majority on at least 90 percent of standard held-out islands |

Failure should follow understandable decisions: waste, food insecurity, inadequate shelter,
bad tool allocation, insufficient labor, an undefended beacon, or failed recovery. The
generator and balance must not create unavoidable deaths.

The standard distribution should contain a modest resource buffer. A few mistakes remain
recoverable through repair, redistribution, and revival; repeated poor decisions compound
into failure.

The desired shape is:

> Survival is learnable. Rescue is difficult. Perfect rescue is exceptional.

## Compact Progression

    wreck rations and berries
              |
         wood and stone
              |
          workbench
        /      |      \
      axe   pickaxe   spear
        \      |      /
        efficient production
              |
       hunting and cooking
              |
       campfire and shelter
              |
      fishing-net investment
              |
    night defense and recovery
              |
         rescue beacon
              |
            rescue

The graph remains deliberately small. Depth comes from spatial logistics, private
ownership, public investment, partial observation, recurring survival pressure, and delayed
effects rather than a large recipe catalog.

### Resources

The bounded economy contains:

- Finite wreck rations.
- Finite or very slowly regenerating berries.
- Finite trees and stone.
- A bounded animal population.
- Perishable raw and cooked food.
- Consumable fire and beacon fuel.
- Material and labor costs for construction and repair.
- Spoilage and damage as irreversible sinks.
- One renewable food flow created by a fishing-net investment.

Bounded does not mean that every flow must be permanently finite. The fishing net converts
finite construction materials and delayed labor into a limited renewable food stream. That
single investment creates the intended immediate-consumption-versus-future-capacity
decision without adding a full ecology simulation.

### Private Tools

Retain the Stage 7B tools:

- Axe.
- Pickaxe.
- Spear.
- Torch.
- Carrying pack.

Tools remain individually owned, transferable to adjacent agents, and storable at camp.
Roles modify efficiency rather than imposing absolute capability locks.

### Public Infrastructure

The official v1 infrastructure is:

- Workbench.
- Campfire.
- Shelter.
- Fishing net.
- Rescue beacon.

The camp already provides shared storage. Do not add a separate storage building. Cooking
remains a campfire action. Do not add a cooking rack or garden in v1.

Public structures use shared materials, joint labor, condition, damage, and repair. Private
tools and public projects must compete for overlapping wood and stone so allocation choices
have consequences.

## Threat And Recovery Design

Retain one monster type: the night stalker.

- Each night selects and spawns one or two stalkers through seeded randomness.
- Spawn positions and targets vary enough to require observation-conditioned responses.
- The current 25-damage attack remains the initial balance target.
- Fire, shelter, torches, spears, positioning, and joint defense mitigate the threat.
- Exposed agents, shelter, and the rescue beacon can suffer consequential damage.
- Stalkers do not gain tiers, equipment, or a separate combat progression.

The eight nights create repeated preparation, attack, repair, and replenishment cycles.
Difficulty should not escalate by simply increasing monster counts.

Downing, revival, structure condition, and repair remain important mechanics and recorded
diagnostics. They must not be unconditional headline achievements because rewarding
revival or repair can incentivize policies to allow preventable harm.

Report conditional outcomes instead:

- Revival success given that an agent was downed.
- Repair success given that a structure was damaged.
- Time to recovery.
- Damage prevented.
- Resources lost to attacks.

## Rescue

Rescue is the terminal population objective, not an automatic bailout.

- Agents know the task, current day, and remaining time.
- A costly public beacon is constructed at the camp after its material and infrastructure
  prerequisites are met.
- Agents may begin construction whenever prerequisites are satisfied; there is no Day 6
  action lock.
- The beacon consumes resources, labor, and fuel and can be damaged.
- During the final 300-tick window, it must be operational for the required maintenance
  fraction and operational at the final dawn.
- After Night 8, a ship rescues active surviving agents if the beacon condition is met.
- Without a qualifying beacon, surviving until the deadline is incomplete success.

The initial maintenance target is at least 240 of the final 300 ticks plus an operational
beacon at final dawn. Stage 7D balances the exact material, labor, fuel, condition, and
maintenance values before the benchmark contract is frozen.

Knowing the deadline is intentional. It creates a finite-horizon allocation problem:
populations must decide when to stop expanding productive capacity and start investing in
the final public project. Procedural geography, stochastic threat placement, local
observations, and resource state prevent a valid policy from succeeding through clock-only
actions.

Evaluation prevents one-survivor rushes from defining success by separately reporting
majority rescue, perfect rescue, survivor count, population health, and resource loss.

## Environment Interfaces

Retain PettingZoo ParallelEnv as the canonical multi-agent API.

The official decentralized actor receives:

- A local semantic tile window.
- Personal health, hunger, energy, inventory, life state, and role.
- Owned and equipped tools.
- Visible local entities and valid target slots.
- Local interaction eligibility.
- Day, phase, and normalized remaining time.
- Camp and public-project state only when within the defined camp-information radius.
- One flattened legal-action mask equivalent to the structured action contract.

It must not receive:

- Hidden entity IDs.
- Exact remote resource locations.
- Remote camp contents outside the information radius.
- Food-lot IDs.
- The global map.
- Other agents' private inventories outside observation range.
- Ledger entries.

For centralized critics and diagnostics, add a separately versioned fixed-size global-state
tensor. Do not pass the replay-oriented global dictionary or an ever-growing ledger into a
critic.

The final environment is expected to require a versioned interface after v2 because the
fishing net and beacon change registries and state:

| Contract | Planned identifier |
|---|---|
| Environment | VoyagerCivilization-v3 |
| Scenario | voyager_bounded_rescue_v1 |
| Action | civilization_structured_action_v3 and flattened_action_v3 |
| Observation | civilization_local_observation_v3 |
| Central state | civilization_central_state_v1 |
| Achievement | voyager_collective_achievements_18_v1 |
| Reward | voyager_shared_achievement_reward_v1 |
| Benchmark | VoyagerCollective-v1 |

Stage 7A v1, Stage 7B v2, compact environments, and Replay 2.0-2.2 remain loadable.

## Reward And Evaluation

### Training Reward

The official learned baselines use one frozen shared reward:

- A shared first-time reward for unlocking a benchmark-relevant achievement.
- Minimal transparent health/survival feedback.
- Explicit invalid-action penalties.

Dense development rewards may remain available for debugging and ablations, but they are
not the official learned-baseline reward unless Stage 7E pre-registers and freezes them
before final training.

Evaluation never uses accumulated training reward as the public score.

### Eighteen Population Achievements

Use exactly eighteen outcome-oriented achievements, with three in each capability family:

1. Survival.
2. Exploration.
3. Production.
4. Interdependence.
5. Public infrastructure.
6. Rescue.

The achievement concepts are:

| Family | Outcome concepts |
|---|---|
| Survival | Majority survives first night; full population reaches midpoint; majority reaches rescue window |
| Exploration | Forest discovered; quarry discovered; shoreline or signal site discovered |
| Production | Workbench completed; three distinct tool types produced; hunted meat cooked |
| Interdependence | Productive tool handoff; cross-agent food chain; multiple roles complete required public work |
| Public infrastructure | Campfire maintained through a night; shelter protects the population through a night; fishing investment produces food |
| Rescue | Beacon completed; majority rescued; full population rescued |

Stage 7D must define exact, testable event predicates and audit every achievement for
reward-hacking or perverse incentives before freezing IDs.

Do not use arbitrary action counters such as depositing ten times. Transfers only count
when provenance proves that the receiving agent productively used the item or tool.

### Voyager Score

On held-out evaluation islands:

1. Compute the success rate of each achievement across episodes.
2. Aggregate the eighteen rates with the smoothed geometric mean:
   exp(mean(log(1 + success_rate))) - 1.
3. Report the full achievement vector beside the aggregate.

Always report:

- Voyager Score.
- Majority-rescue rate.
- Perfect-rescue rate.
- Mean and distribution of rescued agents.
- Technology depth.
- Invalid-action rate.
- Resources produced, consumed, spoiled, damaged, and remaining.
- Private-tool versus public-infrastructure investment.
- Conditional repair and revival success.
- Per-agent and per-role ledger contributions.
- Environment and training throughput.

The aggregate describes breadth of learned outcomes. It does not by itself prove
cooperation. Provenance-based achievements and causal ablations establish that the
population dependencies mattered.

## Validity Gates

### Learnability

- A parameter-sharing feed-forward PPO/IPPO probe must learn early production above random
  on the handcrafted Stage 7B task within two million agent transitions.
- If it does not, repair observation encoding, action masking, reward, or task balance
  before procedural generation or new mechanics.

### Solvability

- The procedural scripted oracle must achieve majority rescue on at least 90 percent of
  standard held-out islands.
- Invalid or unreachable islands are rejected by generation validation.
- Random rescue must remain below 5 percent.

### Cooperation Dependence

- A greedy/no-sharing population must perform substantially worse than the cooperative
  oracle.
- Disabling transfers must materially reduce the interdependence capability family.
- Removing public infrastructure must prevent qualifying rescue.
- Productive-handoff achievements must reconcile through the ledger.

### Closed-Loop Control And Generalization

- A timestep-only policy must fail.
- Training and evaluation seeds are disjoint and published.
- Resource zones, travel routes, and threat positions vary enough to prevent coordinate
  memorization.
- Full observability is an ablation, not the official actor setting.

### Headroom

- Random performs poorly.
- Feed-forward PPO learns meaningful early and middle outcomes.
- Recurrent PPO or MAPPO has a plausible opportunity to improve memory-dependent or
  credit-assignment-dependent outcomes.
- At least one recurrent or centralized-critic baseline materially improves the
  interdependence or rescue capability families over the feed-forward/independent
  baseline under the same budget, and the gain persists on unseen islands.
- The oracle remains substantially better than learned policies.
- If every method saturates or every method fails, rebalance before freezing.

### Compute Feasibility

The current Stage 7B no-op measurement on the 2023 M2 MacBook Air is approximately:

- 95 joint world steps per second.
- 950 agent transitions per second.

This is a diagnostic, not final training throughput. Stage 7C profiles active trajectories,
policy inference, learner updates, multiprocessing, and long-episode ledger costs.

A five-million-transition feed-forward pilot should finish overnight on the M2. The
official per-seed budget cannot exceed ten million agent transitions for v1. If the full
pipeline exceeds that practical target, optimize the simulator or simplify the benchmark
before final runs.

## Baselines And Training Protocol

Minimum scripted baselines:

1. Legal-action random.
2. Greedy individual survival with no population plan.
3. Cooperative procedural scripted oracle.

Minimum learned baselines:

1. Parameter-sharing feed-forward PPO/IPPO.
2. Parameter-sharing recurrent PPO.
3. Recurrent MAPPO with a centralized critic and decentralized actors.

Stage 7E runs pilot learning curves at:

- Two million agent transitions.
- Five million agent transitions.
- Ten million agent transitions as a hard maximum.

The final budget is the smallest pre-registered budget that produces stable learning and
meaningful algorithm separation. Longer episodes do not linearly increase compute under a
fixed transition budget, but they reduce the number of episode endings and increase
temporal credit-assignment difficulty.

Final Stage 7F results use:

- Three independent training seeds per learned method.
- One hundred held-out evaluation islands per trained checkpoint.
- Deterministic official inference, with stochastic inference reported separately if useful.
- Episode-level exports and confidence intervals across both training and evaluation seeds.
- Exact hardware, software, wall-clock, world-step, and agent-transition reporting.

Do not add QMIX, model-based MARL, pixels, population generalization, or LLM policies to the
minimum v1 release.

## Procedural Island

Stage 7C replaces the handcrafted map only for the official distribution. The handcrafted
7A/7B island remains a development, regression, and replay artifact.

The generator creates:

- A safe wreck and camp region.
- Early food within a validated travel band.
- Separated forest, quarry, hunting, shoreline, and beacon-relevant regions.
- Traversable routes that require exploration without creating unavoidable bottlenecks.
- Seeded deer and nightly stalker placement.

Every generated seed must satisfy:

- Required resources and sites exist.
- Required sites are reachable.
- The progression is solvable under the 2,400-tick horizon.
- Early unavoidable lethal spawns are impossible.
- Resource quantities fall within the standard balance envelope.
- Generation is deterministic under the complete versioned configuration.

Do not implement animal reproduction, a garden, multiple biome ecologies, storms, or shared
discovery memory in v1. Local observations plus recurrent policies provide the official
memory challenge.

## Ledger And Auditability

Stage 7B's append-only ledger is a benchmark feature, not actor information.

It supports:

- Exact resource conservation.
- Provenance-aware collective achievements.
- Per-agent and per-role contribution analysis.
- Reward-hacking investigations.
- Public-versus-private investment measurement.
- Replay reconstruction.
- Offline trajectory datasets.

Training must not repeatedly scan an ever-growing ledger when an incremental balance or
debug-only reconciliation can preserve correctness more efficiently. Evaluation and tests
must still prove exact reconciliation.

## Replay And Viewer

The Python simulation remains authoritative. The browser renders recorded facts and never
simulates, infers, or corrects behavior.

Preserve:

- Replay 2.0 compact runs.
- Replay 2.1 Stage 7A vertical slice.
- Replay 2.2 Stage 7B deterministic-core run.

The final replay extension records:

- Procedural map and seed metadata.
- Fishing-net construction and production.
- Beacon construction, fuel, condition, damage, maintenance, and rescue qualification.
- Population achievements and exact provenance.
- Policy and checkpoint identity.
- Conditional repair and revival metrics.
- Terminal rescue outcome and ledger reconciliation.

Do not record every training episode. Record selected evaluation trajectories after
training.

The final eight-day replay contains 2,400 ticks. The viewer provides:

- Full manual playback and exact seeking.
- Faster overview playback so an eight-day run can be watched in roughly five minutes.
- Automatic slowdowns or bookmarks for discoveries, construction, transfers, attacks,
  repair, downing, revival, beacon activation, and rescue.
- Seed-matched random, greedy, oracle, feed-forward PPO, recurrent PPO, and MAPPO runs.
- Achievement, resource-flow, contribution, and private-versus-public investment panels.

Stage 7G adds presentation quality, not gameplay mechanics.

## Implementation Phases

### Stage 7B Closeout

- Make the canonical Replay 2.2 demonstration complete a revival through public actions.
- Run the complete Python, Ruff, mypy, replay, frontend, TypeScript, production-build, and
  browser acceptance suite.
- Record active and no-op throughput.
- Preserve all current Stage 7B mechanics and compatibility paths.

Exit criterion: Stage 7B satisfies its written acceptance plan and is ready for an
intentional release commit.

### Stage 7C: Trainability And Achievement Calibration

1. Adapt the Stage 5 trainer to the v2 flattened action mask and structured observation.
2. Preserve the failed v1-v3 full-environment probes and the successful component tests as
   diagnostic evidence rather than repeatedly changing the public world.
3. Freeze a reward-independent achievement spectrum on the unchanged 600-tick handcrafted
   island and re-score legal random plus the existing feed-forward PPO checkpoint.
4. Train parameter-sharing recurrent PPO directly on the complete environment, with one
   private recurrent state per decentralized agent and no diagnostic curriculum.
5. Add MAPPO only if recurrent PPO and a minimal public team-state/action-factorization
   remediation still fail to separate learned behavior from random.
6. Profile and optimize simulation and training throughput throughout these pilots.
7. Only after baseline separation, implement deterministic 48x48 procedural generation,
   validation, and disjoint development/held-out manifests. Add centralized critic state
   only when a MAPPO experiment actually requires it.

Exit criterion:

- At least one learned baseline exceeds legal random on the frozen achievement score and on
  meaningful delivery, construction, or tool achievements.
- The desired diagnostic ordering is legal random below feed-forward PPO below recurrent PPO
  or, only if necessary, MAPPO.
- A large seed sweep is deterministic, reachable, and valid.
- A five-million-transition run is projected to complete overnight on the M2.

#### Probe v1 Result

`civilization_trainability_probe_v1` is an immutable failed experiment. On fifty evaluation
seeds over the handcrafted map:

- PPO composite was 0.084 versus 0.404 for legal random.
- The paired difference was -0.320 with 95 percent interval [-0.356, -0.284].
- PPO completed no gathering, deposit, workbench, or tool capability under the v1 metric.
- PPO retained a majority through tick 300 on 42 percent of episodes versus 86 percent for
  random.
- The selected checkpoint's invalid-action rate was 76.1 percent versus 3.7 percent for
  random.
- Two million transitions ran at about 842 agent transitions per second. The measured
  five-million-transition projection was 1.65 training hours before evaluation overhead.

The v1 result is not discarded or relabeled. Its local artifacts remain under
`results/stage7c/ppo_seed0`. The result triggered the predeclared instruction to fix
observations, rewards, action representation, or difficulty before adding content.

#### Probe v2 Remediation

The remediation makes four bounded changes:

1. `WORK` is masked unless the public structure has reserved materials or the complete
   reservation is currently available at camp. `REPAIR` receives the equivalent material
   precondition. Simultaneous competition can still fail symmetrically because no local
   mask can know the other agents' choices.
2. The training adapter appends a ten-way one-hot agent identity. It exposes no map,
   inventory, creature, or remote-agent information; it only lets a shared deterministic
   policy assign different behavior to otherwise observationally symmetric agents. The
   public Stage 7B observation and Replay 2.2 contracts remain unchanged. The actor encoder
   is versioned from 591 to 601 values.
3. `civilization_trainability_probe_v2` rewards finite gathered units, the first delivery
   credit for each gathered wood/stone unit, first-time team tile visits, workbench material
   reservation, applied workbench labor, completion, first production of each tool type,
   and survival. Delivery credit can never exceed gathered production, so
   withdraw/redeposit loops cannot create reward. Entropy decays from 0.02 to 0.005 rather
   than collapsing to 0.001.
4. Episode artifacts include selected verbs/actions, precondition versus symmetric
   rejections, gathered/deposited quantities, peak camp stocks, time to gather the
   workbench bundle, workbench completion time, first-tool time, invalid rate, survival,
   and every capability predicate.

The v2 capability vector is independent of reward:

- Gather at least six wood and two stone by tick 100.
- Have at least six wood and two stone simultaneously available at camp by tick 300.
- Complete the workbench by tick 600.
- Craft at least one tool by tick 600.
- Retain at least six active agents at tick 300.

The stricter gathering deadline was calibrated against legal random: in ten fixed-map
development runs, random gathered the full bundle by tick 100 on two runs, rather than
saturating the metric on every run by tick 300. The other thresholds and the paired
composite gate remain unchanged.

The v2 250K continuation pilot used:

```bash
.venv-train/bin/python examples/run_stage7c_trainability_probe.py \
  --total-agent-transitions 250000 \
  --seed 0 \
  --dev-episodes 10 \
  --test-episodes 20 \
  --evaluation-milestones 250000 \
  --output-dir results/stage7c/ppo_probe_v2_250k_seed0
```

Its predeclared debugging rule required the checkpoint to beat random on composite, gather
the timed bundle on at least half the evaluation runs, reach camp-material or later progress
at least once, and hold invalid actions below ten percent.

Only a successful current pilot may proceed to two million transitions with ten development
and fifty evaluation seeds. The full gate still requires a composite advantage of at least
0.15 over seed-matched random with a paired 95 percent bootstrap interval excluding zero,
plus every predeclared per-capability threshold.

The implementation profiles actor-observation encoding, mask stacking, actor/value
inference, environment stepping, PPO updates, v2 observation construction, v2 legal-mask
generation, entity-slot generation, and conservation-ledger reconciliation. Workflow smoke
runs are not evidence of learnability. Procedural generation, official island manifests,
and the centralized MAPPO state still wait for the full trainability result.

#### Probe v2 Result

The v2 250K pilot also failed its predeclared continuation gate. On twenty held-out seeds,
deterministic PPO scored 0.11 versus 0.23 for legal random, with a paired difference of
-0.12 and 95 percent interval [-0.18, -0.06]. Its invalid-action rate was 65.0 percent
versus 2.9 percent for random. It gathered about 28.6 stone per episode but no wood,
delivered no materials, and completed no workbench or tool milestone. Seeded-stochastic
inference reduced invalid actions to 23.9 percent but gathered about 90.9 stone and still
completed no progression milestone.

The failure was a genuine reward and credit-assignment problem rather than only an argmax
artifact. Per-unit resource rewards remained economically dominant because the handcrafted
island contains hundreds of finite units. Fully shared invalid penalties also gave PPO weak
information about which selected action caused a simultaneous conflict. The artifacts remain
under `results/stage7c/ppo_probe_v2_250k_seed0`. Throughput passed at roughly 884 agent
transitions per second, projecting five million transitions at 1.57 training hours before
evaluation overhead.

#### Probe v3 Remediation

`civilization_trainability_probe_v3` preserves the public Stage 7B world, action registry,
resources, Replay 2.2, and both earlier reward contracts. It makes four focused changes:

1. The actor receives agent identity plus signed horizontal and vertical camp displacement
   and normalized camp distance. This versioned 604-value training observation exposes a
   known home location without revealing remote terrain, resources, creatures, or agents.
2. Individual gathering credit is globally capped at the six wood and two stone required by
   the workbench. Individual delivery credit uses the same caps and irreversible camp
   high-water marks. Same-tick capped credit is divided proportionally among contributors,
   so agent ID never controls allocation.
3. Valid gather/delivery credit and the -0.02 invalid-action penalty apply only to the
   responsible agent. Survival, resource thresholds, camp thresholds, workbench reservation
   and progress, completion, first tools, downing, and death remain shared. Exploration and
   unlimited per-unit rewards are removed.
4. Every development and held-out checkpoint is evaluated with deterministic argmax and
   independently sampled, seed-reproducible stochastic inference. Both are recorded, but
   deterministic outcomes select checkpoints and control continuation.

Run the v3 250K continuation pilot before another full experiment:

```bash
.venv-train/bin/python examples/run_stage7c_trainability_probe.py \
  --total-agent-transitions 250000 \
  --seed 0 \
  --dev-episodes 10 \
  --test-episodes 20 \
  --evaluation-milestones 250000 \
  --output-dir results/stage7c/ppo_probe_v3_250k_seed0
```

The deterministic checkpoint must beat random, gather the timed bundle on at least half of
held-out runs, reach camp-material or later progress at least once, and remain below ten
percent invalid actions. A stochastic-only success is reported as deterministic coordination
collapse and cannot authorize longer training.

#### Probe v3 Result And Learning Ladder

The v3 250K pilot failed. On twenty held-out seeds, deterministic PPO scored 0.02 versus
0.23 for legal random, selected no-op for most actions, gathered nothing, and had a 12.5
percent invalid-action rate. Seeded-stochastic inference improved substantially over v2:
its invalid-action rate fell to 1.5 percent, and it gathered an average 8.8 wood and 35.65
stone. It still scored only 0.15, 0.08 below random, deposited no stone, and never
reserved materials or began the workbench. The training curve plateaued rather than showing
evidence that a longer feed-forward PPO run would solve the task. Throughput remained healthy
at roughly 876 agent transitions per second.

Stage 7C therefore stops reward-only remediation and adds three controlled training-only
presets over the same `VoyagerCivilization-v2` mechanics, 604-value actor observation, and
270-action registry:

1. `delivery` runs for 150 ticks and exposes only movement, interaction, rest, and wood/stone
   deposit actions. It tests exploration, camp navigation, and delayed delivery credit.
2. `construction` runs for 60 ticks with exactly one workbench bundle added to camp and
   exposes movement, rest, and workbench labor. It tests whether PPO can learn the primitive
   public-work action efficiently.
3. `survival` begins at tick 180, supplies a completed six-person shelter, and runs through
   tick 300 with only direct survival and shelter actions. It tests first-night threat
   response without requiring the production chain first.

Each task trains a fresh policy. Seeded-stochastic evaluation is primary because it measures
the policy distribution PPO actually optimized; deterministic argmax remains recorded to
detect synchronization collapse. Legal random uses the identical restricted mask. These
results are diagnostic and cannot be published as the official benchmark result.

Run all three tests:

```bash
.venv-train/bin/python examples/run_stage7c_learning_ladder.py \
  --tasks all \
  --seed 0 \
  --eval-episodes 20 \
  --delivery-transitions 100000 \
  --construction-transitions 50000 \
  --survival-transitions 100000 \
  --output-dir results/stage7c/learning_ladder_v1_seed0
```

Interpret the first failed task as follows: construction failure indicates a primitive
action, optimizer, or mask problem; construction success plus delivery failure indicates
navigation, exploration, or delayed-credit failure; survival-only failure indicates poor
threat response; all three succeeding while the full probe fails identifies task composition,
memory, or multi-agent credit assignment as the remaining benchmark difficulty.

#### First Learning-Ladder Result And Delivery Decomposition

The 250K learning ladder produced a useful but limited result. Construction passed on all
twenty evaluation episodes and completed the workbench at task step 5; legal random never
completed it and averaged only 0.427 progress. This demonstrates that the flattened action
registry, masks, PPO update, public-work mechanic, and direct bounded reward can support
learning. Survival met the intentionally weak gate at 85 percent majority survival and a
0.705 active fraction, but random achieved 95 percent and 0.735. It is therefore not evidence
of improved survival intelligence. Delivery failed at zero success and zero camp progress.

A traced sampled delivery episode gathered thirteen wood and four food but no stone and made
no deposit. Across evaluation, sampled PPO selected no-op 59 percent of the time and never
returned a carried resource to camp. The next diagnostic decomposes that sequence further:

1. `gather_wood` allows movement, interaction, rest, and no-op for 100 ticks. Success requires
   the population to acquire six wood.
2. `gather_stone` uses the same interface and requires two stone.
3. `return_to_camp` starts all ten agents six tiles from camp carrying two wood or stone,
   exposes movement, rest, no-op, and relevant deposits for 60 ticks, and requires the
   six-wood/two-stone camp bundle.

The public environment remains unchanged. Added resources and start positions exist only in
the versioned diagnostic reset, are recorded as ledger sources, and reconcile exactly. Random
calibration over twenty seeds succeeds 25 percent on wood acquisition, 80 percent on stone
acquisition, and zero percent on complete return-to-camp, so the tests are neither impossible
nor automatic.

Run the three delivery components:

```bash
.venv-train/bin/python examples/run_stage7c_learning_ladder.py \
  --tasks delivery_diagnostics \
  --seed 0 \
  --eval-episodes 20 \
  --gather-wood-transitions 75000 \
  --gather-stone-transitions 75000 \
  --return-to-camp-transitions 75000 \
  --output-dir results/stage7c/delivery_components_v1_seed0
```

If an acquisition task fails, fix resource search or acquisition before touching delivery.
If return-to-camp fails, fix homeward navigation or deposit credit. If all three pass, stop
changing primitive rewards and introduce a curriculum or public team-need signal to compose
the learned skills; do not rerun the unchanged combined task and hope that more transitions
will solve it.

The real component run demonstrated partial or complete learning in all three primitives:

- Wood acquisition reached 55 percent success and 0.833 mean score versus random's 25
  percent and 0.567. Its training reward was still increasing at the end of 75K transitions.
- Stone acquisition reached 75 percent sampled success versus random's 80 percent, but
  completed successful episodes at task step 34 rather than 53.9. Deterministic inference
  reached 100 percent at task step 13, showing a learned directed route despite synchronized
  collision failures.
- Return-to-camp reached 100 percent success in both inference modes versus random's zero,
  completing at task step 22.15 sampled and 8 deterministic. Every recorded invalid action
  was a symmetric movement conflict; there were no illegal deposits or precondition errors.

The original automatic report incorrectly labeled return-to-camp as failed because it used
the strict invalid-action threshold as both a capability and efficiency test. The corrected
report separates `capability_learned` from the stricter task gate. The evidence supports
`component_skills_trainable_combination_or_team_state_failure`: gathering, returning,
construction, and basic survival are individually learnable, while a fresh feed-forward PPO
policy does not compose them reliably in the full island. Diagnostic presets remain optional
debugging tools and do not become the official training distribution.

#### Handcrafted Achievement Calibration

The earlier five-condition composite is retired as a Stage 7C continuation gate. It hid
partial learning whenever a policy failed one late dependency and over-weighted an argmax
deployment mode that makes observationally similar agents synchronize. The immutable probe
artifacts remain valid historical experiments; they are not retroactively re-scored as
passes.

`civilization_achievement_benchmark_v1` evaluates the unchanged ten-agent, 600-tick,
handcrafted island independently of the reward used for training. It freezes fifteen binary
population achievements in four groups:

- Gathering: food, wood, stone, and the cumulative six-wood/two-stone workbench bundle.
- Delivery: deposit food, wood, and stone, plus hold the complete bundle simultaneously at
  camp.
- Progression: start and complete the workbench, craft a tool, and transfer a tool to another
  agent or public camp storage.
- Survival: retain a majority and then the full population through tick 300, and retain a
  majority at tick 600.

Every per-achievement success rate is reported. The aggregate is the smoothed geometric mean
used for sparse achievement spectra:

```text
score = (exp(mean(log(1 + 100 * success_rate_i))) - 1) / 100
```

The formula prevents one zero-rate late achievement from collapsing all visible early
learning to zero, while still penalizing a policy that succeeds only on easy achievements.
Group scores, mean unlock ticks on successful episodes, invalid-action rate, survival, and
episode-level records remain visible beside the aggregate.

All policies use identical seeds. Seeded-stochastic inference is the primary PPO result
because PPO trained a categorical policy distribution. Deterministic argmax remains a fully
recorded diagnostic for synchronization or coordination collapse; it no longer alone decides
whether the environment is learnable.

Re-score legal random and the existing feed-forward checkpoint without training:

```bash
.venv-train/bin/python examples/evaluate_stage7c_achievements.py \
  --feed-forward-checkpoint results/stage7c/ppo_probe_v3_250k_seed0/checkpoints/best \
  --episodes 20 \
  --seed-start 40000 \
  --output-dir results/stage7c/achievement_rescore_v1
```

The recurrent baseline is a shared GRU actor-critic trained directly on the complete v3
environment and reward. Each agent owns an independent hidden state; truncated PPO batches
preserve per-agent temporal order and split at episode boundaries. The actor and critic see
only the same local 604-value input used by feed-forward PPO. No diagnostic reset, restricted
action set, curriculum, or global state enters this run.

```bash
.venv-train/bin/python examples/run_stage7c_recurrent_ppo.py \
  --total-agent-transitions 250000 \
  --seed 0 \
  --dev-episodes 10 \
  --test-episodes 20 \
  --evaluation-milestones 250000 \
  --output-dir results/stage7c/recurrent_ppo_250k_seed0
```

The desired calibration is legal random below feed-forward PPO below recurrent PPO or,
only if necessary, MAPPO. A learned baseline separates only when its aggregate score exceeds
random and it improves at least one meaningful delivery, construction, or tool achievement
by ten percentage points. Strict recurrent headroom additionally requires recurrent PPO to
beat feed-forward PPO by those same criteria.

If feed-forward PPO already separates, the recurrent result measures memory headroom. If
feed-forward fails and recurrent separates, Voyager still has a useful learned baseline but
the strict algorithm ordering is not yet demonstrated. If neither separates, add one minimal
public camp-needs vector or factorize the action output and repeat the short comparison.
MAPPO and its centralized training-only state are conditional after that remediation, not an
automatic feature. Procedural islands, fishing, rescue, and longer episodes remain blocked
until at least one learned baseline establishes meaningful separation on this current world.

### Stage 7D: Final Economy And Rescue

- Add the fishing net as the only renewable investment.
- Add the camp rescue beacon and fixed final window.
- Balance the official eight-day task.
- Define and audit the eighteen achievement predicates.
- Add conditional repair and revival metrics.
- Implement a procedural cooperative oracle.
- Extend replay and functional viewer support for the new facts.

Exit criterion:

- The oracle reaches the solvability threshold.
- Random and greedy baselines fail for understandable reasons.
- Rescue depends on public infrastructure and population allocation.
- No headline achievement rewards preventable failure.

### Stage 7E: MARL Baselines And Calibration

- Carry the calibrated Stage 7C feed-forward and recurrent PPO implementations onto the
  frozen procedural task.
- Add recurrent MAPPO with centralized training and decentralized execution only if the
  Stage 7C evidence or procedural task demonstrates a credit-assignment limitation that
  decentralized recurrent PPO cannot address.
- Run two-, five-, and at most ten-million-transition learning curves.
- Validate the 2,400-tick horizon and choose the fixed interaction budget using the
  predeclared gates.
- Freeze the training reward and baseline hyperparameters before final seeds.

Exit criterion:

- Learned policies exceed random on meaningful capabilities.
- Recurrent or centralized-critic learning materially improves at least one
  interdependence or rescue capability family over the feed-forward/independent baseline
  under the same budget and on unseen islands.
- The final budget remains within the ten-million-transition cap.

### Stage 7F: Freeze And Official Benchmark

- Freeze environment, scenario, action, observation, state, reward, achievement, replay,
  and score versions.
- Freeze training and evaluation manifests.
- Train three independent seeds per learned baseline.
- Evaluate each checkpoint on one hundred held-out islands.
- Run the open-loop, transfer, infrastructure, observability, and generalization tests.
- Export episode data, aggregates, confidence intervals, checkpoints, configs, checksums,
  throughput, and representative replays.

Exit criterion: a third party can reproduce one official benchmark row from a clean
installation, and the evidence supports the stated cooperative-planning claim.

### Stage 7G: Viewer And Benchmark Presentation

- Finish sprites and animations for fishing, beacon operation, damage, maintenance, and
  rescue.
- Add capability, achievement, contribution, provenance, and investment panels.
- Curate seed-matched baseline comparisons.
- Provide overview playback, event bookmarks, and a five-minute presentation path through
  a complete eight-day run.
- Verify every visual statement against recorded simulation data.

Exit criterion: a viewer unfamiliar with RL can explain what the population attempted, why
it succeeded or failed, and how the compared policies differed.

## Stages After Stage 7

Stage 7 must finish a complete RL benchmark. Later stages are not required to make it valid.

### Stage 8: Optional Research Extensions

Stage 8 becomes optional and may include:

- Pixel or hybrid observations.
- Learned communication.
- Mixed individual and team incentives.
- Variable population sizes.
- Partner generalization.
- QMIX, model-based, or hierarchical baselines.
- A JAX or other accelerated implementation.

None of these blocks the v1 release.

### Stage 9: Public Research Showcase

Stage 9 packages the completed benchmark:

- Public documentation and installation path.
- Benchmark tables and capability profiles.
- Research-style report.
- Notable and seed-matched runs.
- Polished website and replay demo.
- Limitations and reproduction instructions.

Voyager may be called complete after Stage 9.

### Stage 10: Optional LLM Interface

An LLM action adapter remains a separate optional extension. It must not alter the RL
environment, score, or baseline claim. Do not pursue it without a concrete later use case.

## Test Plan

### Compatibility

- Compact Stage 1-6 environments and benchmark results retain regression coverage.
- Stage 6 Replay 2.0 and Stage 7 Replay 2.1/2.2 remain loadable.
- Existing environment IDs retain their declared action and observation meanings.
- New contracts use versioned registrations rather than silent mutation.

### Determinism And Conservation

- Identical seed and joint actions reproduce byte-identical events and state hashes.
- Action-submission order never changes outcomes.
- Symmetric movement and resource conflicts remain exact.
- Resources reconcile across agents, camp, world, piles, transformations, and sinks.
- Training optimizations do not change transition outcomes.

### Mechanics

- Tool effects, ownership, transfer, and storage remain exact.
- Food provenance, expiry, cooking, consumption, and spoilage remain exact.
- Structure construction, damage, degradation, repair, and restoration remain exact.
- Night spawn count, placement, targeting, mitigation, and damage remain seeded.
- Downing, revival, terminal death, and conditional metrics remain exact.
- Fishing production is delayed, bounded, and auditable.
- Beacon construction, fuel, damage, maintenance, qualification, and rescue are exact.

### Environment And Training

- PettingZoo parallel API compliance.
- Stable spaces within each declared version.
- Structured and flattened action equivalence.
- Correct one-dimensional legal-action masks.
- No privileged or remote information in actor observations.
- Fixed-size centralized state for MAPPO.
- Feed-forward and recurrent hidden-state resets are correct.
- Evaluation does not update weights.

### Benchmark

- Disjoint seed manifests.
- Generator solvability and rejection tests.
- Stable achievement predicates and score formula.
- Perverse-incentive audit.
- Open-loop baseline failure.
- Cooperation-dependence ablations.
- Multiple training seeds and confidence intervals.
- Episode exports reconcile with aggregates and replay terminals.
- Checkpoints, configs, artifacts, and environment versions are checksummed.

### Viewer

- Replay reconstruction matches recorded hashes and terminal metrics.
- No browser-side mechanics.
- Exact seeking and high-speed playback remain deterministic.
- Every overlay uses recorded facts.
- A complete 2,400-tick replay remains usable in manual and presentation modes.

## Completion Criteria

Stage 7 is complete only when:

- VoyagerCollective-v1 is a frozen eight-day cooperative MARL benchmark.
- The compact Stage 5/6 benchmark and all declared replay versions remain reproducible.
- One shared simulation powers compact and Stage 7 scenarios.
- The official world is a validated procedural 48x48 island with ten decentralized agents.
- The workbench-to-rescue graph is complete without the cut feature set.
- Survival is learnable, rescue is difficult, and the oracle reliably succeeds.
- Inter-agent provenance and ablations demonstrate that cooperation affects outcomes.
- Feed-forward PPO/IPPO, recurrent PPO, and MAPPO are trained under one fixed budget.
- Three training seeds and one hundred held-out evaluation islands support each official
  learned result.
- The score, capability vector, rescue outcomes, conditional recovery, resource flows, and
  contributions are published.
- A clean third-party installation can reproduce one benchmark row.
- Selected runs replay exactly and are understandable in the final viewer.
- Documentation states limitations honestly and does not claim civilization, social
  intelligence, or guaranteed external adoption.

## Explicit Non-Goals For V1

- Thirty-day campaigns.
- A separate storage building.
- Cooking rack, garden, or multiple renewable systems.
- Animal reproduction or detailed ecology.
- New storms or multiple crisis systems.
- Multiple monster species, weapons, armor, or combat tiers.
- Shared discovered-map communication.
- Explicit agent communication or natural language.
- Population-size generalization.
- Pixel-policy training.
- Currency, markets, governance, reputation, or mixed motives.
- LLM-dependent environment behavior.
- A JAX, CUDA, Rust, or C++ rewrite before profiling proves it necessary.
- Claiming cooperation from shared rewards, role labels, or attractive replays alone.

## Immediate Order Of Work

1. Re-score legal random and the existing feed-forward checkpoint with
   `civilization_achievement_benchmark_v1` on fixed seeds.
2. Train the 250K recurrent PPO pilot directly on the complete island and evaluate it with
   the same achievement spectrum.
3. Check for legal random below feed-forward PPO below recurrent PPO, while requiring at
   least one learned policy to beat random on meaningful progression achievements.
4. If no learned policy separates, add a minimal public camp-needs vector or factorized
   action output and repeat the short comparison; add MAPPO only if still justified.
5. Proceed to procedural generation and new content only after useful baseline separation is
   demonstrated.

This order prevents further content work from accumulating on top of an untrainable
interface.
