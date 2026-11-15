# More human-game data across ranks

Bead: `kenggo-ydo.10`, under E2. Collection date: 15 November 2026.

The prepared dataset now has **1,000,000 training moves from 4,561 games and
1,805 mover accounts**. It has about six times the previous training data.
The target of 200 distinct training games is met at **12 of 16 ranks**.
15k, 14k, 13k and 11k remain below target. No new model was trained, so these
results establish data coverage, not better prediction or playing strength.

The owner asked for hundreds of games at each rank and authorized up to
24 hours for collection and processing. This run retrieved 1,200 archive
pages and made 33,817 selected game records available for processing.
The [frozen plan](../plan/e2_rank_collection_v1.md) keeps the existing
population, data splits and game rules. No model training is part of this run.

## Prepared data

| Split | Previous games | Current games | Previous moves | Current moves |
| --- | ---: | ---: | ---: | ---: |
| Training | 757 | 4,561 | 167,851 | 1,000,000 |
| Population development | 206 | 1,117 | 44,204 | 249,987 |
| Unseen-account development | 50 | 1,138 | 11,800 | 249,993 |

This run added 5,803 complete games: 3,804 training games, 911 population
development games and 1,088 unseen-account development games. It stopped at
the fixed move limits, with no game cut into a partial example set. The
machine status is `budget_stopped / example_row_limit`; this is the planned
limit, not a failed build. All available records had already been screened.

| Training rank | Distinct games | Mover accounts |
| --- | ---: | ---: |
| 15k | **66** | 38 |
| 14k | **41** | 30 |
| 13k | **90** | 51 |
| 12k | 350 | 88 |
| 11k | **182** | 91 |
| 10k | 245 | 111 |
| 9k | 349 | 121 |
| 8k | 292 | 146 |
| 7k | 352 | 182 |
| 6k | 431 | 225 |
| 5k | 441 | 242 |
| 4k | 384 | 146 |
| 3k | 290 | 163 |
| 2k | 309 | 187 |
| 1k | 407 | 211 |
| 1d | 515 | 179 |

Bold counts are below the 200-game target. A game counts once at a rank
only if that player made a recorded move; games between different ranks
count in both rows. These are account counts, not verified distinct people.
The survey is a selected sample and does not represent every KGS player.

The more detailed E0 count check passes in 79 of 96 training groups,
19 population-development groups and 21 unseen-account-development groups.
Each group combines rank, move-number range and clock range, and requires
100 game sides and 30 mover accounts. This check differs from the 200-game
target and does not qualify a model or establish a human rank.

| Split | Pass labels | Valid value labels | Masked value labels |
| --- | ---: | ---: | ---: |
| Training | 6,251 | 947,128 | 52,872 |
| Population development | 1,574 | 237,239 | 12,748 |
| Unseen-account development | 1,452 | 238,932 | 11,061 |

The 50 confirmation games and 70 final games remain unchanged. The final
set still has 15,223 moves. Their target arrays were not decoded during
this expansion or its verification.

## What the value label means

Both model outputs use supervised learning. The policy predicts the human's
recorded next move. The value predicts the eventual recorded outcome from
the current mover's view.

**Earlier positions receive value labels too.** If White eventually wins,
a Black-to-move position has target −1 and a White-to-move position has target
+1, including positions in the opening. Unknown, timeout and forfeit results
do not supply value labels; their move labels can still be usable.

The loss is mean policy cross-entropy plus mean squared value error over
positions with valid outcomes, with weight one on each term. The extra
outcome task may help the shared features and could support later search.
Neither benefit has been established here. The current raw move chooser uses
only the policy output. Value is not an evaluation of the next move, optimal
play or a calibrated human win probability.

The [earlier model report](e2_data_expansion_2026-11-14.md) gives the full
architecture: 2,410,515 parameters, four board planes, state and rank inputs,
eight 128-channel residual blocks, 362 move scores and one outcome value.
The review page also includes the layers, inputs, losses and pilot settings.

## Move history

The current E2 network has **zero recent-move history**. The rules engine
retains full history for legality, but that does not expose the move sequence
to the network. The source sequences are retained for later history inputs.

E3's open history decision, `kenggo-kv3.1`, compares 0, 8, 32 and 64 previous
individual moves by both players, including passes. It starts with a simple
recent-move encoding. This will test the owner's concern that recent play
helps predict weaker players' responses. This collection does not change the
network or establish a history benefit.

## Collection result

The first sample requested 1,000 account-month pages and selected 35,000
candidate games: 25,000 training and 5,000 in each development group. All
pages were retrieved. Of 816 required monthly ZIPs, 815 succeeded. One ZIP
exhausted three attempts, leaving 1,360 selected records unavailable:
1,351 population-development records and nine unseen-account-development
records. All training candidates remained available.

Before opening new SGF move sequences, a second frozen sample targeted
15k–13k players. Its 200 pages covered 169 accounts, including 130 not
queried before. All pages and all 67 required ZIPs succeeded. It added
177 candidates, below its 5,000-candidate cap. The combined selection has
35,177 frozen IDs and 33,817 available records. Earlier missing IDs and
failed-request costs were retained; failed candidates were not replaced.

Credentials remain in an owner-only local file outside the repository.
Cookies stay in memory. Neither credentials nor login request bodies are
stored in project artifacts or uploaded to GitHub.

## Why downloaded records were excluded

All 33,817 available records completed screening. **8,949 passed the source
checks and replayed legally; 24,868 failed the source checks.** Duplicate
checks excluded 50 of the passing records, and whole-game move limits
excluded another 3,096. The remaining 5,803 supplied the new examples.

| Source check that failed | Games |
| --- | ---: |
| Rules or komi outside the chosen scope | 20,386 |
| Clock missing or outside the chosen scope | 9,620 |
| Played path through a branching record unconfirmed | 1,773 |
| Moves missing or turn order unclear | 466 |
| Archive and game file disagree on the result | 206 |

The counts overlap because one record can fail several checks. Most failures
concern the chosen rules and clock population. A valid Go game can still be
outside that population. The other checks require unambiguous training labels
and consistent source records.

A separate audit read only six header fields from every selected new game
file. It reproduced every saved rules/komi and clock decision. No rules,
komi or rank fields were missing. Of the 20,386 rules/komi exclusions,
**19,638 (96.3%) explicitly use Japanese rules with 0.5 komi**. Another 742
specify other rules and six specify another Japanese komi value.

The 9,620 clock exclusions also mostly have known settings: 8,330 specify
Japanese overtime outside the approved ranges, 748 specify Canadian overtime,
and 542 have missing clock fields. A few records have more than one clock
problem, but these three groups are mutually exclusive.

Komi is White's point bonus. The current dataset fixes it at 6.5; 0.5 is a
different game setting. The network receives komi as an input, so a separate
comparison can test whether admitting that group is useful.

Across 25,177 training candidates, **9,525 Japanese 0.5-komi games meet the
current clocks and have no other recorded source exclusion**:

| Rank | New candidates | Passed current source checks | Additional 0.5-komi candidates | Additional candidates from changing clocks only |
| --- | ---: | ---: | ---: | ---: |
| 15k | 260 | 65 | 138 | 7 |
| 14k | 316 | 41 | 191 | 8 |
| 13k | 766 | 91 | 457 | 27 |
| 11k | 1,740 | 193 | 830 | 107 |

These counts use the ranks in each record's header, before final selection
and confirmation that each player recorded moves. Different-rank games
overlap between rows. The additional candidates still need full replay,
duplicate and split checks under any new data scope. They are not extra
usable games in this run.

## Why the weak-rank follow-up added few games

The 200 pages contained 9,290 row sightings and **9,204 distinct games**.
Twenty-two pages explicitly reported no games. The following exclusions
are mutually exclusive, applied in the listed order:

| Archive check | Games excluded | Games remaining |
| --- | ---: | ---: |
| Distinct discovered games | — | 9,204 |
| Already selected in round one | 365 | 8,839 |
| Older completed records | 7 | 8,832 |
| Not ranked | 2,519 | 6,313 |
| Not even 19×19 | 3,185 | 3,128 |
| Either rank unknown or outside 15k–1d | 805 | 2,323 |
| Known or suspected bot involvement | 882 | 1,441 |
| Reserved validation account | 166 | 1,275 |
| Reserved test account | 78 | 1,197 |
| Neither player at 15k, 14k or 13k | 1,020 | **177** |

The underlying reasons overlap. Across all 9,204 distinct games, 4,147 were
19×19 handicap games and 4,020 triggered the bot filter. Among 5,264 games
involving at least one focus rank, 2,995 triggered that filter and 2,731
were handicap games.
These records can be valid Go games while falling outside this baseline's
chosen population. SGF rules, komi, clock and replay checks come later.

| Rank | First-round candidates | Added candidates | Combined candidates | Accounts observed at that rank |
| --- | ---: | ---: | ---: | ---: |
| 15k | 233 | 27 | 260 | 57 |
| 14k | 265 | 51 | 316 | 86 |
| 13k | 632 | 134 | 766 | 135 |

These are candidate counts, before SGF admission and confirmation that each
player recorded moves. A game between two focus ranks appears in both rows.
The first sample selected every metadata-eligible 15k–13k training game it
found; its 25,000-game cap did not strand weaker-rank candidates.

A read-only simulation of another 100 pages reached 96 accounts but only
eight new accounts. Only 22 pages had a directly observed rank in the chosen
month; the other 78 used neighboring months. Collection stopped after the
second round so the remaining time could prepare and check the available
records. This is a bounded coverage result, not a claim that further
collection cannot help.

## Recommended data scope

Keep ordinary ranked, even 19×19 games with Japanese rules, 6.5 komi and
the approved 10–30-minute time controls for this baseline. Preserve legal
replay, duplicate checks and separate account groups. Timeout and forfeit
games can still provide move labels; mask their outcome labels.

For the next data decision, prioritize a separately frozen comparison that
adds **Japanese 0.5-komi games with the same clock and quality checks**. Keep
the existing 6.5-komi dataset as the baseline and report each setting
separately. This has much more potential at the weak ranks than changing
clocks alone. The owner has not yet chosen that broader scope; the choice is
tracked in `kenggo-ydo.11`.

The current restrictions make games comparable, but they remove many valid
weak-rank games. Other rules, clocks, handicap or human-versus-bot data need
their own data decision and comparison. More data alone does not establish
better move prediction, complete-game behavior or playing strength.

## Costs and checks

Both collection rounds, including the probe, login, redirects and retries,
used **2,094 requests and 269,935,482 response bytes**. They used 75.107 measured
collector CPU seconds plus the conservative 120-second probe charge. Elapsed
time from the first collector's start through the second collector's final
accounting was about 3 hours 42 minutes, including the review gap. All shared source
limits were met. The smallest request-start interval was 5.000063 seconds.

The build used 57,089.68 CPU seconds and 64,249.14 elapsed seconds: about
15 hours 52 minutes of CPU work and 17 hours 51 minutes elapsed. It finished
around 22:02 UTC, before the 23:05:23 preparation cutoff. Output occupied
1,043,486,424 bytes, below the 16 GiB limit. There were no build-time network
requests or GPU use.

The original dataset and both one-use collection markers remain intact.
Source code and configurations were pinned before execution. The metadata
filter analysis independently reproduced all 177 additional candidate IDs.

Independent export verification checked **all 1,499,980 training and
development rows**. Complete games, legal labels, recorded-winner signs,
missing-data masks and training-only normalization passed. All 2,083 old
memberships and 2,800 old example files were preserved. The canonical file
matched its early compression check exactly: 11,032 records and 56,854,920
bytes, below the 64 MiB limit.

Export verification took 41.51 CPU seconds and 40.84 elapsed seconds.
Browser checks passed at 1000 and 390 pixels wide, covering four games and
20 sample positions per width, stones, moves, passes, value labels, controls
and note downloads. There were no script errors, network requests or page
overflow. The masked-value display used a browser-only test fixture; it did
not change the page's stored data or count as owner feedback.

The full code suite passed **503 tests**, with the pinned native rules
reference enabled and zero skips. After adding the optional scope-audit
readout, all four affected tests passed. Lockfile, environment, CPU smoke, package
build and whitespace checks passed. Fresh independent GPT-6-Astra reviews
checked the collector and continuation before execution. Review found and
fixed redirect provenance handling; the failing-before regression now passes.
See `kenggo-ydo.10` for the completion-review and delivery records.

Production costs exclude development tests and independent code review.
The separate header audit took 10.81 CPU seconds, and the early compression
check took 2.68 CPU seconds. Full timings, source hashes and local artifact
references are in the [evidence manifest](../manifests/e2_rank_collection_v1.json).
The owner allowance ends at **16 November 2026, 00:05:23 UTC**; collection and
preparation use the earlier **15 November, 23:05:23 UTC** cutoff.

## Review and next decision

Open the self-contained [review page](../runs/e2-rank-expansion-review-2026-11-15/review.html)
locally; no server is needed.

1. **Check rank and player coverage (`kenggo-ydo.10`).** The four shortfalls
   matter when choosing which difficulty settings the next comparison can
   support. More total moves do not fill those gaps by themselves.
2. **Step through the four sample games (`kenggo-ydo.10`).** Check the stones,
   next recorded move and value-label sign. Save any concerns in the page's
   notes; this makes data mistakes easy to report with a concrete example.
3. **Choose the next data scope (`kenggo-ydo.11`).** Study Japanese 0.5-komi
   games with the current clock and quality checks, or keep the 6.5-komi
   scope. The candidate counts favor testing the 0.5-komi group, but they do
   not establish usable yield. A follow-up needs its own frozen scope,
   budget and validation plan.

E2 readiness (`kenggo-ydo.6`) stays open. The earlier small pilot's negative
rank-conditioning result and five unresolved raw-game endings are unchanged.
A new fixed model comparison and complete-game checks are still required.
E3's history decision (`kenggo-kv3.1`) also remains open.
