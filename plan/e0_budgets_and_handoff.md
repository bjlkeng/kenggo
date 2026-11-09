# E0 budgets and handoff

Decision date: 8 November 2026. Bead: `kenggo-n5n.3`; E0: `kenggo-n5n`.

Start with a **1,000-candidate-game audit**, a separate **0.5 GPU-hour throughput probe**, a **24 GPU-hour supervised-learning (SL) pilot** and a **48 GPU-hour reinforcement-learning (RL) pilot**. The pilot caps cover all arms and all GPU work charged to each pilot. These are project limits, not measured costs or promises of a successful model. No corpus, GPU availability, model run or human recruitment is established by this note.

Use the [scope decision](e0_scope_decision.md) for the candidate KGS population, support limits and fallback choices. The [evaluation protocol](e0_evaluation_protocol.md) owns metric weights, opponent panels, comparison rules, provisional gain/regression thresholds and calibration counts. This note completes [E0](go_project_plan_v0.5.md#e0-fix-scope-and-evaluation) by assigning costs and unresolved decisions. It does not change the selected stack or qualify any profile.

## Audit and throughput probe

The audit in `kenggo-6yr.2` samples **1,000 candidate game IDs**, not 1,000 surviving clean games. Follow the scope note's frozen discovery frame and sampling method; retain failed retrievals, exclusions and retries in the accounting. Initially plan CPU-only acquisition, inspection and replay, with no audit GPU allocation. Before acquisition, record permitted discovery/retry limits, a finite CPU/wall-time and storage budget, and artifact locations using E1's access and workspace evidence. Stop and record a shortfall if those limits prevent completing the sample. Record actual CPU-hours, elapsed time, bytes and human effort separately from both pilots. Any proposed GPU audit work needs its own explicit budget first.

After the E1 core readiness gate, E2 may run one named throughput probe with a **combined 30-minute GPU allocation limit**, including warmup, all configurations, retries and failures. Plan around one RTX 3090; `kenggo-6yr.1` verifies actual availability. Probe representative data loading, forward/backward steps and reference inference using training/debug inputs. Record hardware, code/dependency/configuration versions, input sizes, completed units, elapsed time, memory and errors. Repeat and alternate cheap measurements at least five times when affordable; retain the spread and distinguish warmup from steady-state time while charging both.

The probe is for implementation feasibility and cost estimation. It does not select models from held-out scores or supply a trained checkpoint to the pilot. A step-rate measurement does not establish complete-game throughput. Measure complete-game and opponent costs in the relevant pilot before projecting them. Only this bounded probe is outside the pilot caps; later pilot tuning, benchmarks and failed training remain pilot costs. A further separate probe needs a recorded purpose and finite budget before it starts.

## Pilot allocations

These initial reservations provide room for evaluation and failures. They can be revised from measured costs **before freezing the comparison**, while retaining every cost already incurred and the same total cap. They are not evidence that the listed work will fit. Freeze matched training units, evaluation counts and stop rules before comparing arms; reserve setup costs already spent when calculating what remains.

| SL pilot activity | GPU-hours reserved |
| --- | ---: |
| Training and tuning, including failed attempts: initially 6 hours each for conditioned and unconditioned arms | 12 |
| KataGo/reference inference and held-out move/value evaluation | 4 |
| Complete-game checks and validation matches | 4 |
| Unassigned reserve within this pilot | 4 |
| **Total SL cap** | **24** |

This is an exploratory, single-training-seed BASE comparison. Use measured throughput to freeze common training data/units that fit both learned arms; equal time reservations alone do not establish equal training. Include raw and any separately declared corrected reference costs. If the protocol's evaluation schedule cannot fit, reduce the exploratory scope before freezing it or defer the comparison; do not quietly omit the expensive control or report a promotion.

| RL pilot activity | GPU-hours reserved |
| --- | ---: |
| Return/boundary checks, small learning task and the lambda 0.95 versus complete-game lambda 1 pilot comparison | 6 |
| A/B rollout inference and B training/tuning: initially 8 hours each for single, pool and curriculum arms | 24 |
| Independent evaluation, including initial B, floor setup, scheduled checkpoints and failures | 12 |
| Unassigned reserve within this pilot | 6 |
| **Total RL cap** | **48** |

Use one saved random initial B checkpoint for the three exploratory arms, with no A weight transfer. E6 still waits for E5a's qualified frozen raw A pool. Freeze a common learner-transition budget that fits the slowest arm after accounting for rollout and evaluation cost; do not equate eight hours with a fixed number of transitions. Include A's inference cost and every lambda/setup trial. If the lambda comparison is inconclusive, record that outcome and any retained default; expanding it into another set of full training arms requires a new cost decision. A 9 x 9 debug learning result remains separate from full-board evidence.

No unused SL allowance transfers to RL, or vice versa. The reserves are part of the caps. Each pilot must report all attempted arms, tuning, evaluations, reference inference and failed/retried runs, including work that yields no checkpoint. These limits do not fund every E5 qualification, three-seed confirmation, extension, final evaluation or release task. Such work requires its own bounded budget when it is not already explicitly charged within a pilot.

## Charge resources and stop at the limit

A GPU-hour is one allocated device for one elapsed hour, including time waiting on CPU work or I/O while that device remains reserved. Sum allocated device time for multiple devices. Keep disjoint intervals per device so shared jobs are charged once in the total; record fixed cost shares for the arms using them. Preserve the original charge when a cached reference result is reused, and report its provenance and reuse in each comparison. Zero marginal compute from reuse does not erase its acquisition cost.

Record a resource ledger alongside run artifacts: budget/comparison/arm/run IDs, code/configuration/data/split/profile versions, device, UTC allocation start/stop, charged GPU-seconds, purpose, outcome or failure reason, shared-cost attribution and balance remaining. Also record CPU/wall time, memory, storage and W&B references when available. Keep reproducibility configurations, manifests and decision reports in version control; raw data, checkpoints and run output remain ignored. Beads owns task status and budget decisions, not a second Markdown task list.

Before launch, reserve an upper cost estimate plus measured checkpoint/shutdown allowance from the remaining balance. Set a deadline that leaves time to stop before the cap. If a run cannot fit, shorten the proposed experiment before it is frozen or defer it. An unexpected overrun is a recorded budget breach and stops further launches; it never resets the ledger. A change to the hard cap requires a new measured-cost decision before more work starts.

Budget exhaustion does not turn a game into a draw or a true termination. Keep scheduled unfinished/unfilled matches as unresolved under the evaluation protocol; preserve the learner's saved-final-observation and bootstrap rules for truncation. If an arm cannot reach its frozen matched training units, report an incomplete comparison. Do not compare unequal completed work as the planned matched experiment or discard costly failures to select a successful checkpoint.

## Budget confirmation and extensions from measurements

Before a promoted comparison, record a separate bounded budget using measured pilot costs and progress. Include **three independent training seeds for every learned arm**, all controls, setup, tuning already spent, reference inference, confirmation evaluation and a finite failure reserve. Keep pilot and confirmation charges identifiable in the cumulative project total; earlier charges are not charged twice. The confirmation sample remains separate from model selection under the evaluation protocol.

Estimate each activity as its declared units times a conservative measured end-to-end cost per unit, then add setup, shutdown and failure allowances. Record the measurement versions, repetitions, variation and assumptions. Include both agents in rollout cost, all candidate/control checkpoints in match cost and the actual profile/context weights and counts. Use comparable hardware and settings; estimate CPU, storage and elapsed time as well as GPU-hours. If a needed workload has not been measured, budget a bounded cost probe before committing to the larger run.

Derive confirmation counts from the protocol's [match and reliability requirements](e0_evaluation_protocol.md#match-protocol-scores-and-intervals), including assertion multiplicity, strata and E1's absolute limits. The 100-color-pair exploratory starting count does not establish a 1% reliability gate. Multiplying one training run's cost by three is not a full confirmation budget. If the required precision, controls or complete-game qualification cannot fit, retain exploratory results and defer promotion.

HIST, PERSONA, TIME, SEARCH, interactions between retained features and a comparison against another B training family each need their own measured-cost decision. Include extra population-training controls, adaptation, search/reference inference and unsuccessful variants where relevant. Record the hypothesis, useful gain, regression limits, maximum spend and keep/defer/drop rule in the owning bead before starting. Keep the final-test panel sealed until selection is complete. A negative pilot or a decision to defer can finish the experiment without establishing playable-release readiness.

## Human calibration is deferred

The [calibration protocol](e0_evaluation_protocol.md#human-rank-calibration-budget) proposes **1,000 games and 250 opponent/profile assignments per claimed label**, or **5,000 games and 1,250 assignments for five labels**. Assignments are not unique people; B needs an additional budget for its own claim. No recruitment route, rates, funding or measured game/service time is confirmed, so this decision schedules no calibration and commits no spending. Until those inputs and the calibration evidence exist, report engine-relative strength and keep human-rank labels unavailable.

Cost participant time, recruitment, administration and inference separately from the SL/RL caps. With N games and M assignments, use these planning quantities:

```text
participant hours = N * hours per game + M * orientation hours per assignment
staff hours = recruitment hours + N * administration hours per game
              + adjudication and rescheduling hours
GPU-hours = sum of allocated inference-device hours, including reserved idle time
cash = participant compensation + staff compensation + recruitment expenses
       + compute and storage expenses
```

For an **illustrative scenario only**, one participant-hour per game, 0.25 hours of orientation per assignment and 0.1 staff-hours of administration per game imply **1,062.5 participant-hours and 100 staff-hours per label**; five labels imply **5,312.5 and 500 hours**, respectively. Recruitment, adjudication, rescheduling and inference are additional. These assumed durations are neither observations nor upper bounds. Volunteer participation can reduce cash cost without removing the time requirement.

Before scheduling, the owning E7 bead must establish access to verified cohorts (including the outer 16k/2d checks), participant terms/rates, staffing and measured game/inference costs, then record finite time, cash and compute limits plus interruption rules. A small feasibility sample also needs a bounded budget and cannot become the frozen calibration sample after its results inform selection. Unfunded or inconclusive calibration remains deferred; it does not relax the rank-claim criterion.

## E1 owns the unresolved contracts

The following are inputs to existing Beads tasks, not completed findings. E1 must record evidence and revise the E0 documents when the audit changes the advertised scope or protocol.

| Existing owner | Decision or evidence required |
| --- | --- |
| `kenggo-6yr.1` — research workspace | Reproducible Python setup and actual CPU smoke commands; separate GPU availability check; artifact locations and ignore rules. |
| `kenggo-6yr.2` — 1,000-game audit | Permitted acquisition and finite audit budget; source/rank/account/clock coverage, exclusions, explicit passes/endings, usable counts and supported/narrowed/deferred capabilities under the scope fallbacks. |
| `kenggo-6yr.3` — rules/endgame/missingness | Exact rules/komi, repetition, scoring/dead stones, draws/forfeits, disputes, cap/resource limits and termination policy; historical-rank, identity, result and timing masks. Define premature-pass judgments, move categories, stress populations, absolute behavior/length/diversity thresholds and judgment coverage before candidate qualification. |
| `kenggo-6yr.4` — shared engine | Demonstrate rules/reference agreement, complete legality history and cloned state, both player perspectives, endgame and truncation behavior across replay/search/RL/serving. |
| `kenggo-6yr.5` — canonical records and splits | Deduplicated played main lines and provenance; chronological whole-game splits, disjoint reserved account cohorts and bridging exclusions; separate development, confirmation and sealed final manifests with usable counts. Keep audit-inspected games out of the reserved final set. |
| `kenggo-6yr.6` — masked examples | Mover-view encoding, pass/action coordinates, legal masks, pre-move inputs and full-history ownership; valid duration extraction and training-only normalizers; verify result/post-move/target-time leakage is absent. |
| `kenggo-6yr.7` — readiness | Accept only usable core population data, splits, game contract and reproducible masked inputs. Keep the gate open for any failed core prerequisite; record optional timing/persona deferrals separately. |

E2 then implements and validates the frozen panels, uncertainty methods, match fixtures and cost ledger before relying on them. Use the linked [gain/regression rules](e0_evaluation_protocol.md#useful-gains-allowed-regressions-and-selection) unchanged unless a revision is recorded before the affected comparison. E0 completion means the initial scope, evaluation choices, budgets and ownership are recorded. It supplies no E1 readiness, model qualification, human-rank or release verdict.
