# E2 BASE pilot

The fixed 20-epoch recipe did not show a rank-conditioning benefit on either
development population. Conditioned minus unconditioned move NLL was
+0.149797 nats in population development and +0.465038 nats in unseen-account
development; lower is better. Both populations are too small for the frozen
uncertainty gate, so these are descriptive single-seed results. They do not
establish a supported winner, a human rank, or a qualified raw opponent.

## Frozen comparison

The pilot uses the owner-approved filtered E1 population: 106 training games,
22,916 policy examples and 20,888 valid recorded-outcome labels. The two owner
exclusions remain in force. Neither confirmation nor final targets were opened.
The two observed-rank development aggregates were fixed before predictions:

| Role | Games | Accounts | Positions | Aggregate mover ranks |
| --- | ---: | ---: | ---: | --- |
| Population development | 14 | 19 | 3,363 | 11k, 7k, 3k, 2k, 1d |
| Unseen-account development | 10 | 11 | 2,563 | 8k, 4k, 3k, 1k |

Every exact-rank cell lacks the 100-game-side/30-account promotion support.
The all-16-rank aggregate is unavailable. The main mean weights ranks equally,
then game sides within each rank, then eligible moves within each side.
Outcome errors and calibration use only valid recorded-result labels.
Pooled-move and account-balanced means remain diagnostics. Intervals use the
frozen weighted dyadic method and are inconclusive here; moving many positions
through a model does not create independent accounts.

Both models use eight 128-channel residual blocks, Adam at 0.001, batch 128,
clock dropout 0.1 and seed 20261114. They start from identical random tensor
values and complete the same 3,600 updates and 458,320 presentations, including
each epoch's final four-row batch. All 20 data-order hashes match. The final
epoch-20 checkpoints were selected before evaluation; no early stopping or
validation-selected checkpoint was used. Probe weights were not reused.

HumanSL is the pinned KataGo v1.18.1 Eigen reference with the humanv0 checkpoint.
It reports raw humanPolicy probabilities and samples at temperature 1, with
full legality history. Its fixed KGS-era/time defaults differ from some source
contexts, and training overlap is unknown. The compact models have zero board
history planes while the shared game state retains complete legality history.
No pass-corrected or searched policy is included in this comparison.

## Recorded-position comparison

All 5,926 HumanSL position queries completed without failures. The same common
positions and label masks were used for every arm; the hidden-rank diagnostic
adds a second pass through the conditioned model, not additional data.

| Development role | Arm | NLL (nats/move) | Argmax match | Recorded-outcome MSE |
| --- | --- | ---: | ---: | ---: |
| Population | Conditioned | 5.953220 | 19.14% | 1.291261 |
| Population | Unconditioned | 5.803423 | 18.23% | 1.416339 |
| Population | HumanSL | 1.574140 | 54.77% | 0.906648 |
| Unseen accounts | Conditioned | 6.003836 | 17.93% | 1.148356 |
| Unseen accounts | Unconditioned | 5.538799 | 19.15% | 1.004303 |
| Unseen accounts | HumanSL | 1.589261 | 55.39% | 0.617442 |

HumanSL has substantially lower descriptive NLL and higher argmax agreement on
both observed development populations. These imitation scores are not match
strength or human-rank calibration. The compact model is not required to match
the external reference to complete a baseline experiment.

Value MSE compares mover-perspective values with valid recorded human outcomes.
For HumanSL, this is `2 * humanWinrate - 1`, converted from White's perspective to
the mover. The page also shows the ten frozen value-calibration bins; MSE alone
does not establish calibration. The per-rank, move-number and nominal-main-time
breakdowns remain in the reports and page, including unsupported cells.

The page keeps argmax agreement separate from 32 seeded temperature-1 samples
per position. Sampling reports contain distinct actions, pairwise disagreement,
pass mass and the six shared move-category frequencies versus recorded moves.
A single human action per position does not identify human conditional entropy.
No searched or pass-corrected policy is pooled with these raw results.

## Conditioning investigation

| Role | Conditioned NLL | Unconditioned NLL | Conditioned, both ranks hidden |
| --- | ---: | ---: | ---: |
| Population development | 5.953220 | 5.803423 | 6.175477 |
| Unseen-account development | 6.003836 | 5.538799 | 6.054492 |

The predeclared hidden-rank diagnostic worsens the conditioned model's mean NLL
by 0.222257 and 0.050656 nats. The input pathway affects predictions, but this
counterfactual does not establish a useful or calibrated interpretation of rank.
Individual ranks have mixed responses, and the separately trained unconditioned
control has lower NLL on both populations. No interval supports promotion.

A post hoc inspection of the fixed initial/final checkpoint tensors confirms
that every observed rank embedding changed in the conditioned model. The 8k
embedding stayed at initialization because training contained no 8k game sides
or visible 8k inputs. Training had only one game side and one mover account each
at 5k, 13k and 14k. The unconditioned model left every non-unknown rank embedding
unchanged. Both models started with identical embeddings; their unknown-rank
parameters changed. There were 2,367 hidden mover-rank rows and 4,693 hidden
opponent-rank rows among 22,916 training positions, plus only 97 recorded passes.
This verifies the learned input paths while exposing sparse coverage; it does
not isolate a causal explanation for the held-out score gap. No new model
query, parameter update or GPU allocation was needed for this inspection.

Training policy loss declined from 4.4840 to 0.9909 for the conditioned arm and
from 4.3122 to 1.0903 for the unconditioned arm. Final development NLL remains
between 5.54 and 6.00. The gap is consistent with poor generalization on this
small corpus; it does not isolate the cause. Training loss is an online epoch
average with training masks/dropout, whereas development uses the frozen
rank/game-side weighting. The values are not a matched train-versus-validation
experiment. More epochs on the same 106 games are not justified by these
results. Additional architecture, history, persona and search comparisons remain
deferred while data support and complete-game behavior are assessed.

## Raw games and endings

All twelve scheduled games were recorded and independently replayed, including
saved referee evidence. There were no missing or unverified slots, runtime
failures or administrative action caps. Each arm played one color pair against
uniform legal play and one against capture-first play. The conditioned model
and HumanSL used a 5k input label; the unconditioned model used no rank. These
are untimed raw policies against unranked opponents, not a calibrated 5k claim.

| Arm | Wins | Draws | Losses | Unresolved | Scheduled-score bounds |
| --- | ---: | ---: | ---: | ---: | --- |
| Conditioned | 1 | 0 | 0 | 3 | [0.25, 1.00] |
| Unconditioned | 2 | 0 | 0 | 2 | [0.50, 1.00] |
| HumanSL | 4 | 0 | 0 | 0 | [1.00, 1.00] |

The bounds treat each unresolved slot as a possible loss or win; they are not
confidence intervals. All five unresolved endings exhausted the fixed referee
cleanup move allowance. They remain in the denominator. The seven wins were
stable across the pinned empirical cleanup budgets and retained independent
native replay evidence; they are not exact minimax certificates. Candidate
moves, referee cleanup and human source results remain separate.

| Arm | Candidate passes | Acceptable | Premature | Unjudgeable |
| --- | ---: | ---: | ---: | ---: |
| Conditioned | 53 | 27 | 4 | 22 |
| Unconditioned | 27 | 11 | 5 | 11 |
| HumanSL | 4 | 1 | 0 | 3 |

Known premature passes occurred in two conditioned games and one unconditioned
game. The other two conditioned games and three unconditioned games lacked a
complete pass judgment. Three HumanSL games also lacked a complete pass judgment;
zero detected premature passes is not zero unknown risk. Passes within a game
are not independent qualification trials. The two resolved unconditioned games
had 608 and 617 candidate/opponent actions, exceeding the frozen 600-action tail
threshold. Unresolved lengths remain censored.

Per-candidate-decision median / p95 latency was 125.8 / 204.6 ms conditioned,
141.4 / 277.8 ms unconditioned and 368.8 / 488.0 ms HumanSL. These describe this
small panel's actual contexts and process boundary. There were 878, 1,110 and
1,049 candidate decisions respectively. Learned network-call counts are known;
HumanSL's native neural-forward count is unknown. Referee calls are recorded
separately and do not become candidate actions or SL labels.

This panel does not qualify any raw opponent. The learned policies have observed
ending and premature-pass failures, and HumanSL still lacks sufficient pass
judgment and qualification coverage. Four games per arm cannot establish broad
strength or an advertised difficulty range.

## Resource interpretation

Both training jobs completed on their first attempts. Their charged GPU
allocations total 1,246.177333 seconds (0.346160 GPU-hours), including process
startup, loading, checkpoint work, durable progress writes and idle time.
The shared SL cap remains 86,400 GPU seconds across all attempts and arms.
The separate E0 cost probe consumed 23.489078 GPU seconds of its own allowance.

Each full training job took about 622 seconds, while the probe's approximately
159-second estimate covered update work. The GPU job performs durable
progress recording around every update and saves checkpoints. The parent then
uploads checkpoints after the GPU worker exits. These boundaries differ; the
numbers are not a measured slowdown ratio.

A separately recorded, single-worker cached-source audit ran on CPU during part
of the raw-play panel. It used no GPU and performed no model inference. These
wall-time and latency observations describe that execution environment; they
are not an isolated performance comparison. The separate throughput probe
finished before this audit began.


## Evidence and next budget decision

The saved-artifact checks independently recomputed 9,600 metric and calibration
values across both development roles and all ordinary/hidden-rank outputs.
Every value matched. The retained evidence index covers 12,604 files totaling
420,437,666 bytes, excluding W&B internal telemetry/cache and writer locks.
The training-only probe already passed exact CUDA checkpoint continuation for
both arms; the pilot also matched both initial tensors and all epoch-order hashes.
This remains a single-seed exploratory experiment, not three-seed confirmation.

The completed pilot charged 8,072.248946 CPU seconds and 8,743.181659 summed
elapsed seconds in its ledger, with 1,246.177333 GPU seconds. The three subsequent
saved-artifact check jobs added 7.976802 CPU / 6.677691 elapsed seconds and no GPU
allocation to that same ledger. Display export and the separate cached-source
audit have their own cost records. Summed job/overhead elapsed time is an
accounting quantity, not a claim that the jobs ran continuously at one boundary.

No further training on the same 106 games is justified now. Architecture,
history, persona and search extensions remain deferred. The
[cached-source audit](e2_cached_coverage_2026-11-14.md) found 295 additional
source-admitted training candidates in its partial prefix, but its observed
unseen-account development subset has only six handles at 4k. It does not
establish the proposed rank range or ready-to-use training examples.

The review page proposes finishing the fixed cached list and building a larger
legally replayed corpus before another SL recipe, with at most six CPU-hours,
eight elapsed hours, zero downloads and zero GPU work. This is a concrete
proposal pending the owner's direction, not an executed build or budget charge.
The alternatives are to prioritize a broader data/cohort proposal or defer
further E2 data/training. Existing cohorts, final reservations, owner exclusions,
full legality history and missing-data masks must survive any next build.
The pass-only baseline-readiness bead `kenggo-ydo.6` remains open while the next
selected scope is unresolved; no raw-opponent qualification or human rank follows
from this pilot.

The [evidence manifest](../manifests/e2_base_pilot_v1.json) pins the complete run,
postchecks, checkpoints and offline review export. Run and artifact outputs stay
outside Git; the recipe, code, manifests and report are versioned. The owning
bead `kenggo-ydo.5` records validation commands and independent review outcomes.


## Reproduction and tracking

The recorded pilot command, run once after freezing the source and inputs, was:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" UV_OFFLINE=1 \
  uv run --locked python -m kenggo.base_pilot configs/e2_base_plan_v1.json \
  runs/e2-base-pilot-2026-11-14 \
  --config-sha256 32cd35356fcea92c1f6e3c081ad47f6f9bd6ebca641e1dd9e593a30f99fe1f75
```

The existing output is immutable. An explicitly justified repeat must use an
unused output directory, retain the shared SL ledger and all previous charges,
and record its source snapshot. This report does not initiate that repeat.

| Stage | W&B run |
| --- | --- |
| analysis | [a61538b5f90940efa6100e0e5fb82387](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/a61538b5f90940efa6100e0e5fb82387) |
| conditioned | [23618932703d4b06ba887b52430e1f9b](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/23618932703d4b06ba887b52430e1f9b) |
| evaluation-conditioned | [42eea184749e449d9af63c89e5f2c9fa](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/42eea184749e449d9af63c89e5f2c9fa) |
| evaluation-human-reference | [e2e649990e3b4217b02fb785f97b8bbc](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/e2e649990e3b4217b02fb785f97b8bbc) |
| evaluation-unconditioned | [cc0c1b8834cf449cac0fc547608f14fb](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/cc0c1b8834cf449cac0fc547608f14fb) |
| matches | [76e60dc6b15841658eb20dcfa091357d](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/76e60dc6b15841658eb20dcfa091357d) |
| preparation | [4ee6fe2a39dd426f9d89177b4f9f8122](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/4ee6fe2a39dd426f9d89177b4f9f8122) |
| unconditioned | [c75046b0f984438497ad003bcf4d51ab](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/c75046b0f984438497ad003bcf4d51ab) |


The full CPU suite passed 440 tests in 187.387 seconds with the pinned native
rules reference enabled and zero skipped comparisons. Lock/sync checks, the
CPU smoke check, package build and UI interaction tests passed. Local Chromium
140.0.7339.16 checked the actual page, replay/pass navigation, persisted notes,
exact SGF and JSON downloads, and mobile overflow with zero page errors or HTTP
requests. Screenshot review found and corrected an overly tall calibration-table
coverage column; its regression failed before the fix and passed after it.
The built-in preview service timed out even on a minimal page, so the local
browser supplied the rendering and interaction evidence. No server was started.
