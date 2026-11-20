# E2 rerun: complete comparisons, continuing game problems

Bead: `kenggo-ydo.14`. The owner allowed up to one week so the unfinished
reference checks and game slots could be reviewed. Production completed in
**26 hours 19 minutes**. All four learned evaluations, both Human SL reports,
16 paired comparisons and 24 game slots finished. No new training ran.

The larger training dataset improved move prediction on the selected
**12k–1d development positions**. The rank-on model's move agreement on
unseen-account 6.5-komi positions rose from **35.83% to 37.17%**. Its NLL
improved by **0.0941**, with an ordinary 95% interval of **0.0664–0.1219**
for the reduction. This exceeds the declared useful reduction of 0.02 for
this exploratory comparison. It is still one training seed and does not
qualify a model or human rank.

Game behavior remains a problem. **All 16 learned-model games had a flagged
premature pass.** Across all candidates, nine games resolved, 13 ended
unresolved and two reached the move limit. E2 readiness (`kenggo-ydo.6`)
remains open. Completing this rerun supplies the missing review evidence;
it does not establish reliable play.

Open the [owner review page](../runs/e2-week-owner-review-2026-11-20/worker/review.html).
The [manifest](../manifests/e2_review_rerun_v1.json) pins the source, results,
verification correction, resource records, page and browser evidence.

## What ran

The [frozen plan](../plan/e2_review_rerun_v1.md) reused the four fixed final
models from the [earlier expanded comparison](e2_expanded_comparison_2026-11-19.md).
They have an eight-block, 128-channel board network, metadata input,
362 move probabilities and a scalar outcome prediction. They have no
recent-move input window; the rules adapter retains full legality history.
Each model had finished 31,250 updates and 4,000,000 training presentations.
The 6.5-only population has 1,000,000 training moves; the combined population
has 2,000,000 moves from 6.5- and 0.5-komi games. Rank-on and rank-off models
share initialization and data order within each population. The rerun used
each model's own training normalizer and changed no weights.

Selection used the frozen metadata hash ordering, up to ten whole games
per rank. Every selected move was evaluated. The smaller rank scope affects
reporting only, after checking all outputs.

| Development set | Games | All positions | 12k–1d positions | Excluded weaker-rank positions |
| --- | ---: | ---: | ---: | ---: |
| Population, 6.5 komi | 136 | 31,233 | 29,119 | 2,114 |
| Population, 0.5 komi | 142 | 31,784 | 27,558 | 4,226 |
| Unseen accounts, 6.5 komi | 127 | 27,949 | 27,268 | 681 |
| Unseen accounts, 0.5 komi | 134 | 29,484 | 27,646 | 1,838 |
| Total | 539 | 120,450 | 111,591 | 8,859 |

All **481,800 learned predictions** and **59,182 Human SL positions** finished.
Human SL remains limited to 6.5 komi. Its history/context defaults and unknown
training overlap differ from the learned models, so it is an external
reference rather than a matched architecture comparison.

The full 15k–1d reports remain available as diagnostics. Population/6.5 and
unseen-account/0.5 have no 15k games; their declared all-rank means remain
unavailable. The other two full-rank sets have concentrated account weights,
so their intervals remain inconclusive. The 12k–1d scope was chosen from
metadata before new predictions; it does not replace the original study or
repair weak-rank coverage. Per-rank support is still below qualification
requirements. Confirmation and final targets stayed closed.

## Prediction results

The following numbers use the **12k–1d, unseen-account, 6.5-komi** scope.
NLL is the loss on the recorded human move; lower is better. Agreement counts
the model's top move matching the human move. Outcome MSE is squared error
against valid recorded winners; lower is better. Unknown, timeout and
forfeit outcomes remain masked for that metric.

| Model | NLL | Move agreement | Outcome MSE |
| --- | ---: | ---: | ---: |
| 6.5-only, rank on | 2.5865 | 35.83% | 1.4096 |
| 6.5-only, rank off | 2.6008 | 35.80% | 1.4270 |
| Combined, rank on | 2.4924 | 37.17% | 1.1722 |
| Combined, rank off | 2.5163 | 36.59% | 1.2387 |
| Human SL reference | 1.5915 | 55.04% | 0.7656 |

Paired NLL changes below compare combined rank-on minus 6.5-only rank-on on
the same positions. Negative means the combined model assigned more
probability to the human moves. Intervals use the frozen game/account
dependence calculation, with equal rank and game-side weighting.

| Development set | NLL difference | Ordinary 95% interval |
| --- | ---: | --- |
| Population, 6.5 | -0.1149 | [-0.1422, -0.0875] |
| Population, 0.5 | -0.1108 | [-0.1425, -0.0791] |
| Unseen accounts, 6.5 | -0.0941 | [-0.1219, -0.0664] |
| Unseen accounts, 0.5 | -0.1467 | [-0.1752, -0.1181] |

The rank-off combined model also improves NLL over its 6.5-only counterpart
in all four scopes; those intervals also lie below -0.02. Adding data changes
both population and komi mix, so this does not isolate the effect of komi.
Evaluation of 6.5-only models at 0.5 komi is outside their training support.

Rank input has a smaller, mixed effect. Within the combined population, its
NLL differences are -0.0422, -0.0264, -0.0239 and -0.0283 in the table's order.
All four intervals exclude zero, but only population/6.5 lies wholly beyond
the useful -0.02 threshold. Baseline rank-on versus rank-off NLL intervals
cross zero in every set.

There is also a value regression: on unseen-account **0.5-komi** positions,
combined rank-on MSE is **1.1900**, versus **1.0385** for rank-off. The paired
increase is **0.1515 [0.0284, 0.2746]**, beyond the allowed 0.02 regression.
This needs investigation before selecting a rank-conditioned policy/value
profile. The page retains rank, phase, clock, diversity and calibration detail.
Single-seed development intervals are descriptive; three-seed confirmation
and qualification gates remain unmet.

## Complete-game behavior

Both panels filled and replay-verified all 12 scheduled slots. They use raw
temperature-1 sampling, both colors, uniform-legal and capture-first opponents,
the same seeds and referee settings as before. Rank-on and Human SL receive
5k as an input setting; rank-off models correctly receive no rank input.
That setting is not a calibrated playing rank. All games use 6.5 komi.

| Candidate | Resolved | Unresolved | Move-limit stops | Games with premature-pass flags |
| --- | ---: | ---: | ---: | ---: |
| 6.5-only, rank on | 0 | 4 | 0 | 4/4 |
| 6.5-only, rank off | 0 | 4 | 0 | 4/4 |
| Combined, rank on | 1 | 3 | 0 | 4/4 |
| Combined, rank off | 2 | 2 | 0 | 4/4 |
| Human SL, both panels | 6 | 0 | 2 | Unjudgeable |

The nine resolved outcomes are candidate wins against these weak opponents.
Twelve unresolved games exhausted the referee's 64-move cleanup allowance;
one exhausted a per-judgment CPU allowance. The two Human SL truncations
reached the action cap. There were no unfilled slots or invalid game records.
The small panels and unresolved outcomes do not establish playing strength.
Human SL repeats the same contexts in the two panels, so those controls are
not independent extra strength evidence.

Across 893 candidate passes, the referee labeled **208 premature, 327
acceptable and 358 unjudgeable**. All 44 Human SL passes were unjudgeable;
zero flagged premature passes there does not prove correct passing. These
judgments need the owner's board review, especially before a pass correction
or referee change. More training alone is not an established remedy.

## Verification correction and costs

The first saved-evidence helper passed 1,789 checks and failed eight because
it incorrectly expected `5k` from rank-off models. Production correctly
stores `null`. The one-use finalizer therefore remains recorded as failed.
Its files and charges were not overwritten or retried.

The root assessment reproduced exactly those eight false positives, checked
all 24 profiles against the intended rank treatment, and rejected synthetic
wrong-rank variants. All **79 assessment checks** passed. The other 1,789
passed checks remain tied to their original pinned evidence. This was a
saved-evidence assessment, with no model queries, native replay or dataset
decoding. It used the original unused publication allowance: at most 300 CPU
seconds, 900 elapsed seconds and 128 MiB, plus a 60-CPU-second closing/wait
reserve. The original November 26 deadline did not change.

The owner page adds a plain results summary and a link explaining the
correction. Its embedded results and interactive script are byte-identical
to the first exported page. The first page passed 4,579 browser checks; the
final owner page passed **4,582**, at 1,000 and 390 pixels. They cover both
rank scopes, all sets, game selectors, flagged passes, boards, SGFs and notes.
There were no script errors, external requests or horizontal overflow.
Automated notes were cleared. These checks use saved worker replay evidence;
they do not independently recompute legality, scoring or every statistic.

| Resource | Final recorded charge or measurement | Limit |
| --- | ---: | ---: |
| New GPU | 0 seconds | No new allocation |
| Cumulative SL GPU | 22,270.127059 seconds | 86,400 seconds |
| Production CPU | 88,138.346346 seconds | Shared cap below |
| Verification, failed attempt, assessment and reserves | 843.014522 CPU seconds | 1,800 seconds |
| Total new CPU | 88,981.360868 seconds (24.72 hours) | 518,400 seconds |
| Production elapsed | 94,761.425721 seconds (26.32 hours) | Shared clock below |
| Elapsed through final page checks | 95,318.738447 seconds (26.48 hours) | 604,800 seconds |
| Output including closing reserves | 5,621,353,399 bytes (5.24 GiB) | 32 GiB |
| Verification/publication output, included above | 73,492,383 bytes | 512 MiB |

The final ledger is settled. CPU charges retain the original 600-second
wrapper reserve and the assessment's 60-second reserve; the latter includes
0.445012 seconds of passive-wait CPU and its failed setup. No historical
training is charged again. The producer's outer timing was 94,763.34 elapsed
and 83,906.32 user-plus-system seconds; conservative process-tree and closing
charges explain why ledger CPU is higher. Source documentation, development
tests and independent review remain separate, as declared in the plan.

Before launch, 647 tests passed with the native rules reference enabled.
Targeted repair tests and a fresh Astra/max launch review passed. Completion
review covers the full bead from its recorded base, the reports, verification
helper and assessment. The final verdict and delivery commit are in Beads.

## Owner review

1. For **the rerun (`kenggo-ydo.14`)**, open the
   [review page](../runs/e2-week-owner-review-2026-11-20/worker/review.html), keep
   the 12k–1d scope, and select unseen-account 6.5-komi data. Compare move
   prediction and uncertainty; then inspect the 0.5-komi value regression.
   This separates the data benefit from the rank/value tradeoff.
2. For **E2 readiness (`kenggo-ydo.6`)**, inspect a few learned games in each
   population. Jump to premature-pass flags and inspect the endings. Note
   whether the judgments match the board and where the failure begins.
   This is the clearest remaining obstacle to reliable play.
3. Download and share the page's notes. The recommended next investigation
   is the causes of premature passing and unresolved endings, followed by
   the rank/value tradeoff, before another long training run. No new model,
   pass correction, referee change or training budget is selected here.

The rerun task can close after independent completion review. E2 and the
readiness gate remain open; no owner feedback is assumed.
