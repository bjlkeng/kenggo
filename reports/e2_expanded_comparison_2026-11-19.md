# Expanded-data Go models: promising prediction, unreliable endings

Bead: `kenggo-ydo.13`, under E2. Run dates: 18–19 November 2026.

The larger dataset gave better human-move prediction estimates in this
matched four-model experiment. It did **not** establish a reliable playing
opponent: every one of the 14 recorded games from our models had at least
one pass flagged as too early. Only three of those games received a resolved
outcome. E2 readiness (`kenggo-ydo.6`) stays open.

All four models finished training, all 16 learned-model evaluation reports
finished, and all 16 planned paired comparisons were saved. The experiment
remains **incomplete**: the external Human SL evaluation hit its CPU limit,
and the two game panels left four slots unfilled. These limits and partial
results are retained. The [evidence manifest](../manifests/e2_expanded_comparison_v1.json)
pins the plans, source, inputs, outputs, costs and validation.

## What was trained

The owner approved the [comparison](../plan/e2_expanded_comparison_v1.md)
with a 24-hour cap. The two frozen training populations were:

| Population | Complete training games | Training moves | Rank input |
| --- | ---: | ---: | --- |
| Japanese 6.5 komi baseline | 4,561 | 1,000,000 | One model on, one off |
| Combined 6.5 and 0.5 komi | 9,135 | 2,000,000 | One model on, one off |

Komi is White's point bonus. The addition changes both the source population
and its komi mix, so the comparison does not isolate a komi effect.

This is supervised learning. Inputs describe the board from the mover's view,
legal actions, rule state and metadata such as rank, komi and available clocks.
The network uses eight residual blocks with 128 channels, a 362-action move
output including pass, and an outcome output in [-1, 1]. The loss adds move
cross-entropy and masked outcome mean squared error (MSE). An earlier position
uses the eventual recorded winner from that mover's view; unknown, timeout and
forfeit outcomes are masked. The network has no recent-move-history input;
the rules engine still retains full legality history.

All four models began with identical random tensors and finished **31,250
updates of 128 examples: 4,000,000 presentations each**. This is four passes
through the baseline data and two through the combined data. Within each
population, rank-on and rank-off models saw the same order. Each used its own
training-only normalizer. The final checkpoint was fixed before evaluation.
Adam used a learning rate of 0.001; the seed was 20261118. There was no
checkpoint selection from development scores.

## Prediction results

Each model evaluated the same four frozen sets of whole games:

| Development set | Games | Positions | Overall estimate |
| --- | ---: | ---: | --- |
| Population, 6.5 komi | 30 | 7,209 | Unavailable: 15k missing |
| Population, 0.5 komi | 31 | 7,155 | Available; interval inconclusive |
| Unseen accounts, 6.5 komi | 30 | 5,813 | Available; interval inconclusive |
| Unseen accounts, 0.5 komi | 29 | 6,195 | Unavailable: 15k missing |

These are small development samples, not final tests. They produced 105,488
learned-model predictions in total. Final and confirmation targets stayed
closed. No missing rank was filled after seeing results.

On **unseen-account 6.5-komi development**, using the declared equal-rank,
equal-game-side weighting:

| Model | Move NLL ↓ | Top-move agreement ↑ | Outcome MSE ↓ |
| --- | ---: | ---: | ---: |
| Baseline, rank on | 2.6085 | 36.11% | 1.5034 |
| Baseline, rank off | 2.5987 | 36.26% | 1.4692 |
| Combined, rank on | 2.4759 | 38.07% | 1.1172 |
| Combined, rank off | 2.4920 | 37.54% | 1.6070 |

NLL measures how much probability the network gives the recorded human move;
lower is better. Agreement measures whether its highest-probability move
matches. Outcome MSE measures error against valid recorded game results.

Adding data reduced rank-on NLL by **0.132609** on this set. The corresponding
population 0.5-komi difference was **-0.090000**. Both are exploratory point
estimates: their uncertainty intervals are inconclusive because an account
can carry more than 10% of the comparison weight. The one-seed study cannot
promote a model. Pooled diagnostics do not replace the declared estimates.

Rank input helped NLL by only 0.016064 within the combined population on the
unseen-account 6.5 set, below the predeclared useful reduction of 0.02. It
worsened baseline NLL by 0.009801 there. These results do not establish useful
rank control. The outcome head also remains a concern: the combined rank-off
model's MSE rose to 1.6070 from the baseline rank-off model's 1.4692.
The page retains rank, game-phase, clock and calibration details.

Human SL completed the full 7,209-position population reference report. Its
unseen-account report remains unavailable. The last saved progress was
3,995 of 5,813 unseen-account positions, or 11,204 of 13,022 overall; these
are not final native shutdown counts. The job was not retried. Human SL
supports only the 6.5-komi reference scope here, and its context and possible
training overlap differ from our models.

## Game behavior

Each population scheduled 12 raw-sampling games: rank on, rank off and Human
SL, each playing both colors against uniform legal play and capture-first
play. The 5k input was a model setting, not a calibrated rank. The fixed
CPU limits allowed 20 recorded games; all 20 passed worker replay verification.

| Recorded status | Baseline panel | Combined panel | Meaning |
| --- | ---: | ---: | --- |
| Resolved | 2 | 5 | Referee cleanup gave a stable winner |
| Unresolved | 5 | 4 | Cleanup exhausted its move allowance |
| Truncated | 1 | 1 | Reached the 800-action limit |
| Invalid | 1 | 1 | Interrupted by the experiment budget |
| Unfilled slot | 3 | 1 | No game record |

The earlier status message grouped all 13 recorded games without a resolved
outcome as "unresolved". The table gives their exact saved categories.
Budget endings are not draws or evidence of illegal play.

Our models supplied 14 records: six baseline and eight combined. All 14
contained a referee-flagged premature pass. The combined rank-on model had
one resolved win and rank-off had two; the baseline learned games had none.
The six Human SL records supplied four resolved wins. The Human SL panels
repeat the same contexts and are not independent extra strength evidence.

Across all recorded candidates, 729 passes were judged: **165 premature,
261 acceptable and 303 unjudgeable**. All 42 Human SL passes were unjudgeable;
zero flagged premature Human SL passes therefore does not establish correct
passing. The 0.5-komi population was not tested in complete-game play.
These small, capped panels do not establish playing strength or reliability.

## Costs and validation

The original deadline stayed **19 November 2026, 14:11:48.993557 UTC**.
After the reference failure, a separately reviewed
[amendment](../plan/e2_expanded_finish_v1.md) ran only the saved paired analysis
and the two original game panels. It kept the trained weights, positions,
seeds, original failure files and cumulative budget. No additional training
or data collection ran.

| Resource | Recorded charge or measurement | Limit |
| --- | ---: | ---: |
| GPU charged by this experiment | 21,023.949726 seconds | Shared cap below |
| GPU including earlier SL work | 22,270.127059 seconds (6.19 hours) | 86,400 seconds |
| Production CPU ledger | 33,877.262594 seconds | Shared cap below |
| Final verification and passive wait | 59.825605 CPU seconds | 180-second verification cap |
| Parent report/hash/display reserve | 300 CPU seconds | Shared cap below |
| CPU including verification and reserve | 34,237.088198 seconds (9.51 hours) | 115,200 seconds |
| Elapsed through production completion | 47,772.271095 seconds (13.27 hours) | Original 24-hour clock |
| Elapsed at verification handoff | 48,152.421880 seconds (13.38 hours) | Original 24-hour clock |
| Observed output plus closing reserve | 1,848,377,329 bytes (1.72 GiB) | 16 GiB |

CPU totals use the supervisor ledger plus the separately recorded final
verification and publication reserve. Match workers also report conservative
local stopping counters; those overlap the supervisor measurements and are
not added again. The first verifier and the earlier 120-CPU-second preparation
reserve are already in the production ledger. Development tests and independent
code review are separate, as declared in the plan. There are no open GPU
reservations or ledger breaches.

Saved-output verification passed **711 evidence checks and 5 accounting
checks**. It checked all frozen prediction reports, contrasts, stage and
checkpoint references, original ledger history and game evidence. It did not
rerun models, engines, referees or raw dataset decoding. The original workers
had already replay-verified all 20 recorded games.

Browser QA passed **969 checks** at 1,000 and 390 pixels, covering all 24
scheduled slots at each width, selectors, boards, pass flags, controls, SGF
downloads and notes. There were no JavaScript errors, external requests or
horizontal overflow. Automated notes were cleared. An initial source-list
comparison helper and a Chromium launch failed; their retained attempts
were corrected without rebuilding or retrying production. Verification stayed
within its 600-elapsed-second and 180-CPU-second limits.

The original prelaunch suite passed 614 tests. After the continuation change,
the 627-test run had 12 setup errors because new files had not yet been staged;
the other 615 passed. All 14 affected cached-audit tests passed after staging.
No code fix was required for that guard. The fresh continuation reviewer also
passed 29 focused tests and found no significant issue. Environment checks,
CPU smoke, package build, input pins, links and diff checks passed. The fresh
completion review and delivery outcome are recorded in Beads.

## Owner review and next decision

Open the [single-page review](../runs/e2-expanded-completion-review-2026-11-19/review.html)
locally; no server is needed.

1. For **the expanded-data experiment (`kenggo-ydo.13`)**, select
   "Unseen-account development · 6.5 komi" and compare all four models.
   Check the missing-rank and uncertainty notices in the other sets too.
   This separates promising estimates from evidence strong enough to select
   or promote a model.
2. For **E2 readiness (`kenggo-ydo.6`)**, inspect a few learned games in both
   populations. Use the pass list to jump to flagged moves and inspect the
   end of each game. Note whether the premature-pass judgments make sense.
   This is the clearest remaining obstacle to a playable baseline.
3. Download the page's notes and send any concerns. The recommended next
   investigation is premature passing and unresolved endings, together with
   the cost of the checking process, before another long training run.
   A changed model, pass correction or larger confirmation study needs its
   own frozen comparison and budget. No owner feedback is assumed here.

This experiment task can close as a documented, bounded incomplete result
after independent completion review. E2 and its readiness gate remain open;
no opponent profile or human rank is qualified.
