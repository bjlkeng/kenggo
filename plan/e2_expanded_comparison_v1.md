# Retrain the human-move baseline on the expanded data

Bead: `kenggo-ydo.13`. The owner approved proceeding on 18 November 2026,
with a 24-hour cap. This is an exploratory E2 experiment, not qualification
of an opponent or a human-rank claim.

## Questions and controls

Train four models from the same random initial tensors:

| Training population | Rank input on | Rank input off |
| --- | --- | --- |
| Preserved Japanese 6.5 komi, 1,000,000 moves | baseline-conditioned | baseline-unconditioned |
| Combined Japanese 6.5 and 0.5 komi, 2,000,000 moves | combined-conditioned | combined-unconditioned |

The [data study](../reports/e2_komi_study_2026-11-16.md) supplies the second
population and preserves the first. The addition changes both the game
population and its komi mix; a result cannot isolate komi from those population
differences. It measures the usefulness of this particular addition.

Use the existing eight 128-channel residual blocks, policy cross-entropy and
masked outcome MSE, Adam at 0.001, batch 128, clock dropout 0.1 and seed 20261118.
Keep zero recent-move history and the existing missing-rank fallbacks. The rules
engine retains full legality history. No architecture, history, search,
personalization or learned resignation experiment is included.

All four models receive the same number of updates and 128 examples per update.
Shuffle each training population deterministically, carrying a batch across an
epoch boundary when needed. Within a population, the two rank treatments use
identical data order. Across populations, match update and presentation counts,
not epochs or source identities. Each model uses its population's existing
training-only normalizer; no development data fits normalization.

## Timing and checkpoint selection

After implementation checks and independent prelaunch review, start the one-use
production session. Prepare the evaluation sets, then run four training-only
cost probes, one per arm. Each probe uses 256 updates: six warmup updates and
five consecutive blocks of 50 updates. Charge loading, warmup, durable writes,
checkpoints and shutdown too. Probe checkpoints never initialize main training.

Before main training, freeze a shared update count using the slowest measured
block and the largest measured setup/checkpoint overhead. Reserve 20% of the
remaining per-arm wall allowance, round down to a multiple of 15,625 updates
(one combined-data epoch), and cap at 187,500 updates. If even one such cycle
does not fit, stop with a cost result. This is a conservative planning rule,
not a guarantee that the full run will fit.

Use the final fixed-update checkpoint for every comparison. Save initial,
middle and final checkpoints for evidence. Do not choose checkpoints from
development scores or reuse prior pilot weights. Record initial tensor hashes,
actual update/presentation counts and each population's data-order evidence.

## Evaluation population

Freeze four separate sets: population development and unseen-account
development, each at 6.5 and 0.5 komi. Select whole games using only source,
rank, komi, role and recorded action-count metadata. For each rank from 15k
through 1d, order eligible games by a hash of seed, role, komi, rank and source.
Try to include two games with moves from that rank, unioning duplicates across
rank choices. Each set has at most 32 games and 12,000 positions. Skip a game
that cannot fit whole and retain coverage shortfalls. Selection never consults
model predictions or winner labels. All recorded moves in a selected game enter
evaluation; unsupported or missing ranks remain explicit.

All learned arms use identical raw positions, legal masks, targets and row
order. Apply each model's own training normalizer only at inference. Report
the 6.5-only models' 0.5 results as out-of-training-support diagnostics.
KataGo Human SL remains an external reference on the two 6.5 sets only; its
current profile does not support 0.5. It keeps its documented history and
era/time defaults, with unknown training overlap. No confirmation or final
target is opened.

Report move NLL (lower is better), argmax move agreement, valid-outcome MSE and
calibration, with rank, phase and clock breakdowns. Keep the existing
equal-rank, equal-game-side, equal-position weighting and account-aware
uncertainty checks. Retain 32 temperature-1 samples per position and existing
move-category diagnostics. A small selected development sample can remain
insufficient for a supported interval even when the training set is larger.

The four predeclared contrasts in each set are conditioned minus unconditioned
within each population, and combined minus baseline within each rank treatment.
The main addition comparison is combined-conditioned versus baseline-conditioned
on 6.5 development; report both development roles separately. The main rank
comparison is conditioned versus unconditioned within each training population.
Use the [E0 gain and regression rules](e0_evaluation_protocol.md#useful-gains-allowed-regressions-and-selection):
a useful NLL reduction is 0.02 nats per move, with at most 0.02 NLL or value-MSE
regression in supported groups. Other behavior limits remain unchanged.
Ordinary intervals are descriptive; this single-seed study cannot promote a
profile. A failed or unsupported required check retains an inconclusive result.

## Complete games

Run the existing bounded 6.5-komi panel twice, once for each training population.
Each panel has its conditioned and unconditioned models plus Human SL, with one
color pair against each of uniform legal play and capture-first play: 12 games
per panel. Both panels use the same seed and existing referee limits. The Human
SL repetitions are shared-context controls, not independent extra strength
evidence. The 5k input is a model setting, not a calibrated rank.

Keep raw sampling, premature-pass judgments, unresolved endings, long games and
unfilled slots visible. No pass correction, new referee scope or 0.5-komi match
qualification is introduced. A limited match panel can reveal failures but does
not establish the project's reliability or strength requirements.

## Approved limits and accounting

The 24-hour elapsed clock begins at the one-use production launch, before input
checks, selection or probes. Stop expensive work before the final publication
reserve. Keep all earlier charges in `runs/e2-sl-budget-2026-11-14/resources.json`:
the original cumulative cap stays 86,400 GPU seconds, of which 1,246.177333 were
already charged. Do not create a replacement ledger or reset a failed attempt.

| Activity | Maximum per job |
| --- | ---: |
| Common-set preparation | 1 elapsed hour |
| Four cost probes | 15 elapsed minutes each |
| Four main training jobs | 2.5 elapsed hours each |
| Four learned evaluations | 1 elapsed hour each |
| Human SL recorded-position evaluation | 2.5 elapsed hours |
| Two raw-game panels | 1.5 elapsed hours each |
| Paired analysis | 30 elapsed minutes |

All stages also share a 32-CPU-hour cap and 16-GiB new-output cap; each job has a
smaller explicit CPU/output limit. One production worker runs at a time. GPU
allocation includes CPU/I/O waits while a GPU job owns the device. Tracking and
setup overhead remain charged separately without double-counting worker costs.
Keep at least 15 elapsed minutes for shutdown and publication. The absolute
deadline and remaining ledger balance override any per-job allowance.

These are maxima, not promised durations or an obligation to spend the budget.
Stop on a failed training/evaluation job; retain its costs and partial evidence.
Do not compare unequal completed training as the matched experiment. Match
panels may report explicit unresolved or unfilled slots under their fixed caps.
No new game downloads are included. W&B receives experiment metrics and
checkpoint artifacts through the existing project; raw games stay local.

Implementation, synthetic tests and code review are outside production accounting.
Before launch, freeze code/configuration/input pins and obtain fresh independent
Astra/max review. Afterward, verify saved evidence, publish a self-contained HTML
review and concise report, and obtain a fresh completion review. E2 readiness
(`kenggo-ydo.6`) remains a separate gate.
