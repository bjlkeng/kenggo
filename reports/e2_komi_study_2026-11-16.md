# More training data from cached 0.5-komi games

Bead: `kenggo-ydo.12`, under E2. Study date: 16 November 2026.

The builder produced **2,000,000 training moves from 9,135 complete games**.
It added 1,000,000 moves from 4,574 Japanese 0.5-komi games while preserving
the existing 6.5-komi dataset. The exported coverage reaches the target of
200 training games at **all 16 ranks**, including the four previous gaps.
Komi is White's point bonus. This study measures
data coverage; no model was trained and no playing-strength gain is claimed.

The owner approved the [separate cached-data study](../plan/e2_komi_study_v1.md)
with a 20-CPU-hour, 24-elapsed-hour budget and no downloads or GPU work.
The build, all-row verification and browser checks passed. The
[evidence manifest](../manifests/e2_komi_study_v1.json) pins the inputs, outputs,
costs and checks. The completion review and delivery are recorded in the bead.

## Prepared data

| Split | Preserved 6.5-komi games / moves | Added 0.5-komi games / moves | Combined games / moves |
| --- | ---: | ---: | ---: |
| Training | 4,561 / 1,000,000 | 4,574 / 1,000,000 | 9,135 / 2,000,000 |
| Population development | 1,117 / 249,987 | 1,128 / 249,958 | 2,245 / 499,945 |
| Unseen-account development | 1,138 / 249,993 | 1,116 / 249,998 | 2,254 / 499,991 |

The run added **6,818 complete games and 1,499,956 moves**. It stopped at
the planned move limits (`budget_stopped / example_row_limit`). No game was
cut into a partial example set. Development totals fall just below their
limits because the next complete games did not fit.

All 13,047 frozen candidates were processed. Duplicate checks excluded
107 records; the whole-game move limits excluded another 6,122. Duplicate
reasons overlap: 104 records belonged to ambiguous duplicate groups, 90 were
duplicate copies, 44 matched the frozen corpus and 96 matched prior rejected
records. These overlapping counts must not be added together.

The original source population remains incomplete: one monthly ZIP was
unavailable in the previous collection, leaving 1,360 development records
unavailable. This cached study made no new requests or substitutions.

## Training-rank coverage

| Rank | Preserved 6.5-komi games | Added 0.5-komi games | Combined games | Combined mover accounts |
| --- | ---: | ---: | ---: | ---: |
| 15k | 66 | 138 | 204 | 50 |
| 14k | 41 | 189 | 230 | 71 |
| 13k | 90 | 454 | 544 | 112 |
| 12k | 350 | 794 | 1,144 | 132 |
| 11k | 182 | 820 | 1,002 | 145 |
| 10k | 245 | 658 | 903 | 157 |
| 9k | 349 | 653 | 1,002 | 186 |
| 8k | 292 | 759 | 1,051 | 220 |
| 7k | 352 | 772 | 1,124 | 260 |
| 6k | 431 | 661 | 1,092 | 304 |
| 5k | 441 | 619 | 1,060 | 296 |
| 4k | 384 | 465 | 849 | 279 |
| 3k | 290 | 634 | 924 | 303 |
| 2k | 309 | 737 | 1,046 | 289 |
| 1k | 407 | 555 | 962 | 284 |
| 1d | 515 | 231 | 746 | 199 |

A game counts once at a rank only if that player made a recorded move.
Different-rank games count in both rows. Accounts are not verified distinct
people. The 15k and 14k groups remain the smallest, so meeting the 200-game
target should not be read as equally strong evidence at every rank.
The added 0.5-komi 15k games have only **29 mover accounts**, below E0's
30-account threshold even before separating move-number and clock groups.

The combined training data cover **2,498 mover accounts**. The addition
supplies 1,779 and the preserved baseline has 1,805; these sets overlap.
The review page separates each komi setting and the finer move-number/clock
groups required by E0.

| E0 groups meeting the count check | Preserved 6.5 komi | Added 0.5 komi | Combined |
| --- | ---: | ---: | ---: |
| Training | 79 / 96 | 81 / 96 | 89 / 96 |
| Population development | 19 / 96 | 15 / 96 | 49 / 96 |
| Unseen-account development | 21 / 96 | 18 / 96 | 50 / 96 |

Each group combines rank, move-number range and clock range, requiring
100 game sides and 30 mover accounts. One game side means one player's
moves in a game. Seven training groups still fall short; this finer check
and later model qualification remain separate from the 200-game target.

## Labels and implementation

| Split | Combined pass labels | Valid outcome labels | Masked outcome labels |
| --- | ---: | ---: | ---: |
| Training | 12,700 | 1,905,030 | 94,970 |
| Population development | 3,268 | 474,997 | 24,948 |
| Unseen-account development | 3,065 | 478,991 | 21,000 |

The move label is the human's next recorded move, including recorded passes.
The outcome label is the eventual recorded winner from the current mover's
view, including at earlier positions. Unknown, timeout and forfeit outcomes
remain masked. Move durations and dynamic clocks remain masked too.

Komi now belongs to the shared game state, so replay, saved states, encoding
and scoring use the same value. The separate study profile admits 0.5 komi;
existing 6.5-komi profiles keep their old behavior and saved bytes. Old model
and referee profiles explicitly reject unsupported 0.5-komi play.

The combined dataset references the old shards unchanged and has a new
normalizer fitted only on training rows. It keeps the original account groups,
date boundaries, owner exclusions, rejection history and reserved targets.
The network architecture and zero recent-move-history setting are unchanged.

## Costs and validation

The metadata preflight used 16.37 CPU seconds and 16.73 elapsed seconds.
It retained a conservative 300-CPU-second charge inside the approved budget.
It checked 882 cached ZIPs and 33,817 root headers before freezing the exact
13,047 candidates and their order.

The builder's outer measurement was **66,579.21 CPU seconds and 70,185.08
elapsed seconds**. Its internal accounting, including the conservative preflight
charge, was 66,878.93 CPU seconds. The retained interruption reserve raised its
budget charge to 66,938.92 CPU seconds. Elapsed time from the recorded production
launch through the build was 70,247.63 seconds.

Through final local verification, measured CPU use was **18.54 hours** and the
conservative charge was **18.68 hours**. Elapsed time was **19.76 hours**.
An additional 300 CPU seconds covers the parent's report, hash checks and
display handoff, bringing the conservative total to **18.77 CPU hours** within
the 20-hour cap. Development tests and independent code review are separate,
as specified in the approved plan. Charged output was **1.64 GiB**, including
the final evidence and handoff reserve, within the 32-GiB limit.

The exporter completed once, using 112.45 measured CPU seconds and 111.27
elapsed seconds. No production build or export failed. Two small local helper
attempts failed: one used a Python version without the required hashing helper;
the other could not parse that failed attempt's timing log. Both were corrected,
and their costs remain included. The shared preview could not open a local file,
so the actual page was checked with installed Chromium in offline mode. An
optional inline chart preview timed out; this did not affect the page checks.

The final prelaunch suite passed **555 tests**, including the native rules
reference, with zero skips. Environment, smoke and package checks passed.
Fresh independent Astra/max prelaunch review passed on its third pass.
Review fixes bind each move label to its recorded game, retain interrupted
CPU charges, prevent rebuilding after verification starts, and allow final
publication to resume without extending replay or encoding limits.

The readout checks each allowed move and mover against its hash-bound
canonical trajectory, whose complete legal replay was checked during data
preparation. It does not independently recompute every stored board or legal
mask. All **2,999,936 training/development rows** passed those checks, including
komi signs, value masks and training-only normalization. It also checked
**11,032 old memberships and 18,226 old example files**. Confirmation and
final targets remained unopened.

Browser checks passed at widths of **1,000 and 390 pixels**. At each width they
covered all nine split/komi selector combinations, seven sample games and
24 positions, including board stones, labels, controls and notes download.
There were no JavaScript errors, external requests or horizontal overflow.
Automated notes were cleared; no owner feedback is assumed.

The [final verification evidence](../runs/e2-komi-inputs-2026-11-16/verification-final.json)
and [verification seal](../runs/e2-komi-inputs-2026-11-16/verification-final-seal.json)
contain the detailed checks, timings, limitations and artifact hashes.

## Owner review

Open the [single-page review](../runs/e2-komi-expansion-review-2026-11-16/review.html)
locally; no server is needed.

1. For **the cached-data study (`kenggo-ydo.12`)**, compare 6.5 komi, 0.5 komi
   and combined coverage, especially 15k and 14k. Note any concern about the
   small account counts. This shows where more games still provide limited
   variety.
2. In the same study, inspect a few positions from both komi settings. Check
   the next-move marker, mover, komi and recorded-outcome label. This is a human
   spot check of what the model would learn.
3. Download your notes and send any concerns for **E2 readiness
   (`kenggo-ydo.6`)**. These checks inform the next model-comparison plan;
   accepting this dataset does not establish better play or fund training.

E2 readiness (`kenggo-ydo.6`) remains open. A later model comparison needs
matched 6.5-only and combined-data training runs, a frozen compute budget,
and development and complete-game checks. Data coverage alone cannot select
an opponent profile or establish a human rank.
