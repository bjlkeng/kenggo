# Policy-only continuation improved human-move prediction

Bead: `kenggo-ydo.19`.

The four-trial study finished under its original three-day allowance. The trial
starting at learning rate **0.001**, without weight decay, had the lowest stopping
validation loss. Its selected checkpoint predicts recorded human moves better
than the common parent on all four fixed development samples. Every trial stopped
under the frozen plateau rules, with time left. Further training after each
selected checkpoint did not produce a lower stopping-validation loss.

Open the [sealed review page](../runs/e2-policy-recovery-2026-11-26/publication/worker/review.html)
for every trial, training and validation curves, earlier runs, and paired
comparisons. The [compact charts](../runs/e2-policy-completion-2026-11-26/charts.html)
show the selected trial and its comparisons with the parent. The
[manifest](../manifests/e2_policy_tuning_v1.json) pins the saved evidence.

This is exploratory move prediction. No new complete-game panel, playing-strength
measurement, human-rank calibration or E2 readiness decision occurred. The value
head remains unqualified for search. No owner feedback or promotion is assumed.

## Training and checkpoint selection

The [frozen protocol](../plan/e2_policy_tuning_v1.md) continued the same saved
4M-data parent, which already had 125,000 updates and 16 million move
presentations. It retained **3,999,997 fitting moves / 18,126 whole games**, the
training-only normalizer, rank inputs, eight residual blocks, 128 channels and
batch size 128. There was no new data collection or transfer of A's weights to B.
Full legality history remains independent of the model's zero move-history window.

All trials reset AdamW once, shared parent weights and sampling/dropout seeds,
then resumed optimizer, sampler and random state between intervals. Only legal
human-move cross-entropy contributed gradients; the value-head parameters were
frozen. Shared representation changes can still alter value predictions. Value
metrics could neither select a checkpoint nor veto a policy result.

| Initial learning rate | Weight decay | Completed intervals | New updates | Move presentations | Selected interval | Best stopping NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0 | 10 | 78,130 | 10,000,640 | 4 | 2.290686 |
| 0.0003 | 0 | 8 | 62,504 | 8,000,512 | 1 | 2.318607 |
| 0.0001 | 0 | 8 | 62,504 | 8,000,512 | 1 | 2.320233 |
| 0.0003 | 0.0001 | 8 | 62,504 | 8,000,512 | 1 | 2.318243 |

NLL is negative log likelihood: lower means more probability assigned to the
recorded human move. The raw minimum selected each checkpoint; the unchanged
0.002 improvement rule controlled learning-rate reductions and patience.
All four trials reached `validation_plateau`, with no uncertain updates or
interrupted intervals. The study retained 38 checkpoint measurements, including
four epoch-zero evaluations that added no training.

The selected 0.001 trial contains **31,252 new updates / 4,000,256 presentations**,
at lifetime step **156,252**. Its initial learning rate had fallen to 0.0003 by
interval 4. Each other selected checkpoint contains 7,813 new updates at lifetime
step 132,813. Selected checkpoints therefore have unequal training exposure.
The [selected run's saved tracking receipt](https://forge.coreweave.com/wandb/bjlkeng/kenggo/runs/7f748d7e340341ed862ec4f2f6089272)
and all other run receipts are pinned in the manifest; no new remote query or
upload was needed during completion.

For the selected trial, fixed training-monitor NLL fell from 2.1185 to 1.9934 at
selection and 1.9086 at the final interval. Stopping-validation NLL went from
2.3559 to 2.2907, then rose to 2.3646. This divergence is consistent with further
fitting that did not generalize to this stopping sample. It does not establish
why it occurred or justify changing the already frozen stopping rule.

## Samples and fixed comparisons

Stopping validation contains **577 whole games / 125,998 positions**, including
**124,851 positions within 12k–1d** and 578 accounts. It was selected from metadata,
before model evaluation, excluding fitting, fixed-assessment, prior-stopping and
protected memberships and duplicate identities. Its exact source digest matches
the frozen plan. Three protected trajectory fingerprints remain unavailable;
their existing SGF/action-sequence fingerprints were checked without opening
protected targets.

Coverage is uneven: the 6.5-komi groups have only five 11k games, 23 at 7k and
19 at 1d; the 12k/0.5 group has six mover accounts. The separate training monitor
uses 11,147 in-scope positions out of 11,273. Both curves weight ranks equally,
then game sides within rank, then valid moves within side. Online minibatch loss
uses changing weights, dropout and different averaging; it is separately labeled.

The selected trial and every trial's best checkpoint were sealed **before** the
fixed development assessment. All five models, including the parent, completed
four reports: **602,250 predictions** across **539 games / 120,450 positions** per
model, with **111,591 positions in 12k–1d** and 8,859 weaker-rank positions excluded.
These games are disjoint from stopping validation; accounts may overlap. This
development sample and parent were used earlier, so this is not an independent
confirmation. Protected final and confirmation targets remained closed.

| Fixed development scope | In-scope moves / games | Parent NLL | Selected NLL | Paired NLL reduction, 95% interval | Parent / selected agreement |
| --- | ---: | ---: | ---: | --- | --- |
| Population, 0.5 komi | 27,558 / 131 | 2.3929 | 2.3144 | 0.0785 [0.0587, 0.0983] | 39.60% / 40.53% |
| Population, 6.5 komi | 29,119 / 130 | 2.3834 | 2.3217 | 0.0617 [0.0499, 0.0734] | 39.02% / 39.86% |
| Unseen accounts, 0.5 komi | 27,646 / 134 | 2.3790 | 2.3010 | 0.0779 [0.0564, 0.0995] | 38.62% / 40.52% |
| Unseen accounts, 6.5 komi | 27,268 / 125 | 2.3948 | 2.3107 | 0.0841 [0.0617, 0.1065] | 39.31% / 40.46% |

Game counts above include games with eligible in-scope moves; the full sample
also retains games whose rows are all outside scope. Intervals account for
shared games/accounts using the saved weighted dyadic method. They do not cover
training-seed variation or adjust for multiple comparisons. The main unseen-account
6.5-komi agreement gain is 1.16 percentage points [0.46, 1.85]. Selection was not
changed after viewing these comparisons.

All new trials used policy-only loss. There is no equally trained joint-loss
control, so improvement over the parent combines extra training, loss and
optimizer changes; it cannot be attributed solely to removing value loss.
The page retains diagnostic value errors and their masks, but they do not
qualify a value model or reverse the policy result. Earlier runs keep their own
sample definitions; cross-sample history is context, not a matched gain.

## Recovery, costs and verification

The original launch was **26 November 2026, 01:35:51 UTC**. A setup error omitted
required tracking pins, so preparation and publication failed before new model
updates. The [reviewed one-use recovery](../plan/e2_policy_recovery_v1.md) reused
readiness and the original ledger. It launched at 02:03:06 UTC after a clean fresh
GPT-6-Astra/max review. Original failures, snapshots and all five original ledger
entries remain unchanged; no allowance was reset or whole-study retry launched.

Production finished at **17:04:56 UTC**, **15.49 elapsed hours** after the original
launch, before the unchanged **29 November, 01:35:51 UTC** deadline. The settled
98-entry ledger records **6.61 CPU hours** and **14.93 allocated GPU hours**, with
no reservations or budget breach. Its output snapshot is **10.69 GiB**. The
original limits remain 72 wall hours, 108 CPU hours, 72 GPU hours and 96 GiB.

The sealed page explicitly reports the pre-publication snapshot. The final
production summary adds **8.688910 CPU seconds**, **11.434212 elapsed seconds**
and **1,847,364 output bytes**; allocated GPU time is unchanged. Its CPU total
includes the final supervisor charge. Completion preserves both snapshots and
the settled production ledger.

A separate [accounting record](../runs/e2-policy-completion-2026-11-26/completion-accounting-v2.json)
charges a conservative **60 CPU-hour bound** for all local recovery, review,
health checks and completion work since launch. It covers the documented recovery
session, seven health-check windows and completion through 20:40 UTC on
26 November. Charging all 12 possible host CPUs throughout those windows gives
51.83 CPU hours; the 60-hour charge adds slack and deliberately double counts
production CPU during overlaps. Actual timed checks are included in this bound.
This is conservative accounting, not measured usage or a new allowance.

The first completion review caught that a provisional five-hour charge covered
completion only and omitted earlier local work. That record is retained and
superseded. The corrected total is **at most 66.61 CPU hours** under the original
108-hour cap, with no further GPU allocation. Systemd's combined service CPU
record is below the production ledger charge, corroborating its coverage of
tracking helpers. The final handoff checks the completion cutoff, actual timed
checks, elapsed time and output size; it does not rewrite the sealed ledger.

Source remained clean at `76fc5801f7399a355d586ff30ec40f6465e23626` until all
production and saved-evidence verification workers stopped. Completion checked
314 artifact hashes, every selected payload hash, the 38 checkpoint measurements
and 20 assessment reports. All 32 paired comparisons recomputed exactly from
saved predictions. The page's embedded data matches its immutable publication
input and includes the shared learning-review component plus four earlier runs.
No new inference, full fitting-data replay or protected-target decoding occurred.

Offline Chromium checks passed at 1000 and 390 pixels: 160 selector combinations
per width, separate training/validation curves, selected markers, history metrics,
local evidence links and notes download. No network requests, script errors or
page overflow were found. The shared preview tools timed out or rejected local
file URLs; the local checks and screenshots are retained.

Repository lock, sync, CPU smoke, build and diff checks passed. The full suite
ran 820 tests: 812 passed and eight native-reference checks were skipped; this
study changed no rules code. Check results and the fresh independent
completion-review verdict are recorded in the manifest and bead.
The completion source change consists
of this report and its evidence manifest; production implementation spans
`fcad1aa` and `76fc580`, reviewed against the bead's original base `efb2f4a`.

## Owner review

1. **For .19, inspect the selected checkpoint and curves on the review page.**
   The later training-monitor gains did not lower stopping loss; interval 4,
   rather than the last checkpoint, supplies the selected model.
2. **For .19, inspect the paired comparisons and sparse groups.** The intervals
   support better move prediction on these development positions, with one seed,
   unequal selected exposure and no matched joint-loss control.
3. **For E2, record a next-step preference only after that review.** Complete-game
   passing/ending problems and value qualification still need separate work.
   No further training or owner decision is required to complete this study.

E2 and readiness `kenggo-ydo.6` remain open. This study's acceptance is separate
from readiness and does not authorize another experiment.
