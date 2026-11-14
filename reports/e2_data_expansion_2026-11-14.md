# E2 network and larger dataset

Bead: `kenggo-ydo.8`. Started 14 November 2026.

The usable training dataset now has **167,851 recorded moves in 757 games**,
up from 22,916 moves in 106 games: **7.3 times as much training data**.
The requested 100,000-move minimum is met. New archive downloads stopped at
a sign-in requirement, so this expansion uses the available cache. No model
has been trained on the larger dataset.

## Why the model has two losses

Both outputs use supervised learning. **Policy** predicts the human's next
recorded move. **Value** predicts the recorded winner from the current mover's
view: +1 for a win and −1 for a loss. Unknown, timeout and forfeit outcomes do
not supply value labels. Their moves can still supply policy labels.

The loss is the mean policy cross-entropy plus the mean squared error over
valid value labels, with coefficient one on each term. Cross-entropy rewards
giving the recorded move more probability. Squared error rewards predicting
the recorded result. There is no reinforcement learning or search-generated
training label in this stage.

## Model used in the pilot

The [implemented model](../src/kenggo/learner.py) has **2,410,515 parameters**.
The comparison without rank inputs uses the same architecture and parameter
count; it hides both ranks and their validity flags.

| Part | Actual implementation |
| --- | --- |
| Board input | Four 19 × 19 planes: mover stones, opponent stones, ko and legal placements. |
| Other input | Sixteen state/time fields and sixteen validity flags; two 16-number learned rank embeddings and two rank-validity flags. This gives 66 metadata values. |
| First layers | A 3 × 3 board convolution with 128 channels. A 66 → 128 → 128 metadata network is added across the board, followed by group normalization and ReLU. |
| Shared layers | Eight residual blocks. Each has two 3 × 3 convolutions with 128 channels, group normalization, ReLU and a skip connection. |
| Policy output | A 1 × 1 convolution gives 361 board scores. A pooled board vector gives the pass score. Illegal actions are masked before probabilities are computed. |
| Value output | Average pooling, a 128 → 128 → 1 network, and tanh produce a value between −1 and +1. |
| Training | Adam, learning rate 0.001, batch size 128, 20 epochs. |

The sixteen fields are mover color, mover-relative komi, prisoners for each
side, consecutive passes, move number, pending stop, truncation and cleanup
flags, main-time seconds, overtime-period seconds, number of periods, and
remaining time and overtime status for each side. Dynamic clock values are
unavailable and stay masked. The model has no previous-move planes; the rules
engine retains the full game history.
The full input and training contract is in [the learner guide](../docs/learner.md).

## Data used by that model

| Role | Games | Recorded moves |
| --- | ---: | ---: |
| Training | 106 | 22,916 |
| Population development | 14 | 3,363 |
| Unseen-account development | 10 | 2,563 |
| Confirmation, not used for model selection | 50 | 10,462 |
| Reserved final test, still sealed | 70 | 15,223 |

Only **20,888 training moves** have valid value labels. The two development
groups supply 5,926 evaluation moves. Twenty epochs presented the same
22,916 training moves repeatedly; they did not create new examples.

The [pilot comparison](e2_base_pilot_2026-11-14.md) did not show a benefit from
rank conditioning. This was a small-data result, not evidence that conditioning
cannot work with more data. The new data must be tested in a separate fixed
training comparison before claiming an improvement.

## Expansion decision and access

The owner authorized up to 24 elapsed hours for collection and preparation,
with at least 100,000 training moves desired and preferably more. The
[frozen plan](../plan/e2_data_expansion_v1.md) aims for one million usable moves
within that limit. It preserves the original population, split dates, reserved
accounts, masks and two owner exclusions. No new model training is part of
this data run.

The public archive request returned a login page. The collector retained that
failure and did not attempt the remaining 285 new account-month pages. It
made one request and received 1,291 bytes. The existing cache supplies 4,269
accessible candidates; 13 additional candidates lack cached ZIP files. These
are candidate counts, not usable training games. New downloads need KGS
sign-in details through the secure input flow.

## Dataset result

All 4,269 available candidates were processed. Source screening retained
883 games, and all 883 passed replay, duplicate and split checks and were
added as complete games. No candidate was replaced after a failure.

| Role | Old games / moves | New total games / moves | Valid value labels | Recorded passes |
| --- | ---: | ---: | ---: | ---: |
| Training | 106 / 22,916 | 757 / 167,851 | 158,505 | 971 |
| Population development | 14 / 3,363 | 206 / 44,204 | 41,661 | 283 |
| Unseen-account development | 10 / 2,563 | 50 / 11,800 | 10,675 | 79 |

The two development groups now supply **56,004 moves**. Confirmation remains
50 games / 10,462 moves, and the usable final test remains 70 games / 15,223
moves. Their targets were not decoded or used for this expansion.

Of the 3,386 source exclusions, 2,350 involved rules or komi, 1,919 involved
time controls, 184 involved unverified variations, 16 involved mismatched
results and five involved missing moves or turn order. These reasons overlap.
The existing admission rules and the owner's two exclusions were preserved.

Training contains 399 distinct mover accounts and 1,513 game sides. A game
side counts a player's recorded moves in one game; a player with no recorded
move does not add a side. Population development has 145 accounts / 412
sides; unseen-account development has 31 accounts / 100 sides.

Coverage remains uneven. The 4k rank supplies 76,070 training moves, about
45% of the total. Training still has only one game side each at 13k and 14k,
and four at 8k. Of the 96 rank/move-number/main-time cells, **11 training,
four population-development and zero unseen-account cells** meet the E0
minimum of 100 game sides and 30 mover accounts. All exact-rank profile
promotions remain deferred. More data does not establish model quality or
playing strength.

The page includes all ending counts. In training, 324 games have scored
results, 381 end by resignation, 26 by timeout, 19 by forfeit and seven have
missing or incomplete results. There are 362 two-pass endings; these can
overlap result types and do not by themselves prove a score. No pass was
invented at a file ending.

## Review and reproducibility

Open the [single-page review](../runs/e2-data-expansion-review-2026-11-14/review.html)
locally, or run this from the repository root:

```bash
xdg-open runs/e2-data-expansion-review-2026-11-14/review.html
```

1. Read the policy/value explanation for E2. It explains the two supervised
   targets and the model used in the pilot.
2. Check the new counts and rank coverage for `kenggo-ydo.8`. This separates
   the larger dataset from the difficulty settings that still lack evidence.
3. Step through a few sample positions and save any concerns in the page.
   This gives the next E2 comparison concrete data feedback. The open
   readiness bead, `kenggo-ydo.6`, still needs a selected and evaluated model.

The [evidence manifest](../manifests/e2_data_expansion_v1.json) pins the source
request, build request, runtime patch, dataset files, checks and review page.
Raw games and generated files remain in ignored local `runs/` directories.
The dataset is under `runs/e2-data-expansion-2026-11-14/dataset/`; keep the
referenced cache and run files to inspect or reproduce it.

The full build process used **8,541.81 CPU seconds and 9,598.83 elapsed
seconds**: about 2 hours 22 minutes of CPU and 2 hours 40 minutes elapsed.
The separate result-check/export process used 6.04 CPU seconds and 6.04
elapsed seconds. It used no GPU and made no network requests. Collection,
input-freezing attempts and their costs are recorded separately in the
manifest. These measurements exclude development tests and code review;
the shared 24-hour collection/preparation deadline was respected.

## Validation

The source collector passed a fresh independent review after a fix for
truncated HTTP responses. A failed download now retains its byte count and
uses the bounded retry policy.

The first builder review found that rejected copies could escape duplicate
checks when they used another pass spelling or had conflicting rank headers.
The fix uses the same canonical move fingerprint for permitted records,
including rejected copies. Rank conflicts remain excluded; their fingerprints
are available only after identities, dates and the final quarantine pass.
Tests also verify that reserved or wrong-date actions are never inspected.

The separate result checker verified all 223,855 training/development rows,
all 1,200 old memberships and all 446 old example files. It checked complete
games, legal target masks, recorded winners and mover perspectives, unavailable
labels, reserved accounts and independently recomputed training statistics.
Neither confirmation nor final targets were decoded.

CPU behavior tests, the pinned native rules comparison, dependency checks,
smoke check and package build passed. The new readout tests also cover a
consistent reversal of every value label: that corruption must fail even
though labels remain internally consistent within each game. Desktop and
phone browser checks cover displayed totals, controls and saved/downloaded
notes. Independent review and final delivery are recorded in the bead.
