# Study 0.5-komi games alongside the current baseline

Decision: `kenggo-ydo.11`. Follow-up: `kenggo-ydo.12`. Recorded 16 November 2026.

The owner chose **“Study 0.5-komi alongside the current baseline.”** Keep the
completed 6.5-komi dataset unchanged and test a separate addition drawn from
cached Japanese 0.5-komi games. Komi is White's point bonus.

**The owner approved both the scope and the production budget.** The earlier
allowance ended at 16 November 2026, 00:05:23 UTC. The owner approved the new
20-CPU-hour, 24-elapsed-hour run on 16 November, with no downloads or model
training. Launch only after implementation checks and independent review.
Task status and authorization belong in Beads.

## Evidence and purpose

The [completed collection](../reports/e2_rank_collection_2026-11-15.md) has
1,000,000 training moves from 4,561 games. Twelve of sixteen ranks reach
200 distinct games. A header audit found these additional candidates:

| Rank | Current usable training games | Possible 0.5-komi training games |
| --- | ---: | ---: |
| 15k | 66 | 138 |
| 14k | 41 | 191 |
| 13k | 90 | 457 |
| 11k | 182 | 830 |

The full candidate set has **9,525 training games**, 1,502 population-development
games and 2,020 unseen-account-development games: **13,047 records** in total.
They have Japanese rules, 0.5 komi, current clocks and no other recorded source
exclusion. These are not admitted games. Replay, duplicate, split and example
checks may remove them; games between different ranks appear in both rank rows.

The question for this run is whether this addition fills the four rank gaps
with usable, diverse data. Report distinct active-player games and accounts,
recorded passes, result types, value masks and E0 support groups separately
for 6.5 komi, 0.5 komi and their union. The collection target stays 200 games
per rank; it does not establish a supported difficulty setting or human rank.
A valid shortfall or negative result completes this investigation.

## Fixed inputs and populations

Use the existing [evidence manifest](../manifests/e2_rank_collection_v1.json)
and its cached inputs. Important pins are:

| Input | SHA-256 |
| --- | --- |
| `runs/e2-rank-expansion-2026-11-15/summary.json` | `f3209a278e9ab3ff7074913ca7f4e30377f60148f8b3ebd56c54fd5e53f2b136` |
| `runs/e2-rank-continuation-2026-11-15/sources.json` | `0fe528cd35e00a402aa1a8617243ea58df618bcd7a3c3e9a9b8cd763ac54c8bc` |
| `runs/e2-rank-inputs-2026-11-15/sgf-scope-audit.json` | `126de24cbd290a81b1b39040d047d595200998c3d93de21e12c624a74e990844` |

The control is the complete, unchanged 6.5-komi dataset, including its split
memberships, example bytes, normalizer and exposure records. The experimental
dataset is that control plus newly admitted 0.5-komi examples. Store the new
output in a separate directory with separate manifests and a new normalizer
fitted only on its training rows. Reference old shards without rewriting them.

This is a versioned extension of the source population in the
[E1 contract](e1_game_and_data_contract.md), limited to this study. Keep ranked,
even 19×19, Black-first KGS 2025 games, both ranks 15k–1d, and the existing
10–30-minute main time with 3–5 Japanese overtime periods of 20–60 seconds.
Keep bot screening, source consistency, played-path checks, whole-game legal
replay and the two owner exclusions. Do not admit handicap, setup stones,
other rules, other komi values or newly permitted clocks.

Preserve the original account groups and chronological boundaries: training
before September 2025, development in September–October, and no new
confirmation or final examples. Both players matter for reserved-account
exclusions. Do not decode old confirmation/final targets. The 1,360 unavailable
development records and their failed request remain unavailable; do not retry
them or select replacements.

## Selection and reconsideration of old exclusions

Before any new move replay, use the pinned headers and prior source decisions
to freeze the exact 13,047 candidate IDs, source/member hashes and roles. If
that count or any pin fails to reproduce, stop and investigate. No new archive
queries, candidate substitution or outcome/model-score selection is allowed.

Within training, put records involving 15k, 14k, 13k or 11k first. Preserve
the original frozen source order within that group and within the remaining
training records. Preserve the original order within each development role.
Interleave one record from each role in train/population-development/
unseen-account-development order, skipping exhausted lists. Pin this order
before replay. Apply move limits only to whole games; retain every exclusion
and unprocessed candidate in the result.

The old builder intentionally rejects any previously rejected source. Do not
delete that history or disable its duplicate checks. Create a separate,
versioned reconsideration ledger linking each selected source to its old
decision, old hashes and the new 0.5-komi scope. An exception applies only when
the sole prior exclusion is the old rules/komi filter and the pinned header
explicitly says Japanese/0.5. Any other rejection, owner exclusion or reserved
cohort restriction still blocks admission.

Check duplicate components against all old accepted and rejected records
using their existing fingerprints without decoding sealed targets. A selected
record's own scope-only rejection may be reconsidered; an unselected alias or
another exclusion cannot be silently waived. Re-run duplicate and split checks
for the new batch. Keep the original decisions intact and record the new
decision beside them. A hash mismatch or unexpected prior reason stops the run.

## Required implementation checks

Changing the admission flag alone is insufficient. Current code fixes 6.5
komi in [SGF screening](../src/kenggo/sgf_audit.py),
[strict replay](../src/kenggo/rules_replay.py),
[example admission](../src/kenggo/examples.py),
[input encoding](../src/kenggo/encoding.py) and
[scoring](../src/kenggo/adjudication.py). The current shared game state also
uses a fixed rules version.

Use one explicit game-level komi value, represented as integer half-points:
1 for 0.5 and 13 for 6.5. Replay, cloning, serialization, snapshot hashes,
encoding and scoring must agree on it. Encoding should read it from game
state, not from an unrelated caller override. Preserve the old 6.5 defaults,
serialized evidence and example arrays. Give the new contract/profile an
explicit version; reject incompatible old scoring evidence or checkpoints
rather than silently reinterpreting them.

An encoder-only override would be smaller, but could present 0.5 to the model
while replay or scoring still assumes 6.5. The shared-state design prevents
that disagreement. A new profile must not broaden the default old readers.

Before production execution, tests must cover both colors' komi signs,
state round trips and clones, unchanged legality/full history, known scoring
fixtures at both komi values, rejected mismatches, duplicate aliases and the
narrow reconsideration exception. Run the pinned native rules-reference
checks; a skipped check is not validation. Keep uncertified endings unresolved.
Unsupported 0.5-komi referee/inference paths must reject the setting explicitly.

Value labels still use the recorded eventual winner from the current mover's
view at every eligible pre-move position. Unknown, timeout and forfeit outcomes
stay masked; source result labels never become inputs. Retain all other masks
and actual pass labels. Do not invent passes or change the network architecture,
zero-history setting, learned resignation or existing qualification claims.

## Approved production budget

The owner authorized the following single run. The timer starts at the
recorded launch after implementation, tests and independent prelaunch review.
Freeze the code and input pins before the metadata preflight. That preflight
records the launch and freezes the ordered candidates, profile, request and
absolute deadlines before move replay. Give this study its own one-use marker;
preserve all old markers.

| Resource | Maximum |
| --- | ---: |
| Production CPU, including checks and failed attempts | 20 hours |
| Elapsed production time | 24 hours |
| Replay/encoding cutoff | 19 CPU hours or 22 elapsed hours, whichever comes first |
| Network requests / GPU time / model training | Zero |
| New training moves | 1,000,000 |
| New moves in each development role | 250,000 |
| Candidate records | The fixed 13,047 |
| Selected SGF bytes / one member | 256 MiB / 1 MiB |
| ZIP directory expansion limit | 4 GiB |
| New canonical file / total new output | 128 MiB / 32 GiB |
| Production workers | One |

The move limits are additions to the unchanged base, so the combined training
cap is 2,000,000 moves. Development caps are 499,987 population moves and
499,993 unseen-account moves. This reserves up to one CPU hour and two elapsed
hours for publication, verification and the readout. Stop on any binding limit;
retain partial evidence and costs. A resume uses the same remaining allowance
and absolute deadline. Development tests and code review are recorded separately
and do not reset or extend production limits.

The metadata-only launch preflight gets a conservative 300-CPU-second charge
within those 20 hours and a hard 10-minute elapsed cap. Its one-use marker is
created before reading production inputs. It freezes headers and candidate
order before any new move replay. Record measured preflight cost separately;
an interrupted preflight retains the full charge and its marker. Subtract its
member and ZIP expansion bytes from the builder's remaining byte limits.
Reserve 512 MiB each for preflight and readout, leaving at most 31 GiB for the
build. Verification attempts share the original deadline and retain their
costs or a conservative interrupted-attempt charge; they cannot reset the budget.
Builder checkpoints keep an interruption charge during final processing too.
Once verification starts, the published build cannot be resumed; verification
retries use its own retained costs and the same deadline.

Budget basis: the previous build added 1,276,125 rows in 15.86 CPU hours and
17.85 elapsed hours. Scaling that single observation to 1,500,000 new rows gives
about 18.64 CPU hours and 20.98 elapsed hours. This is a rough planning estimate,
not a benchmark or a guarantee: game lengths, replay work and the new code may
change the cost. The caps allow a smaller valid dataset; do not extend them to
reach a target.

## Comparison and acceptance

This first run compares **data coverage**, not model quality. Report the
before/after counts, all failures and actual costs, with each komi setting
separate. Independently verify every allowed example's legal policy label,
komi input, value sign and masks, complete-game membership and the new
training-only normalizer. Rehash every old membership, shard and pinned input.
Compare every study row's action and mover with its pinned canonical trajectory,
whose complete legal replay was checked during preparation. This correspondence
check does not independently recompute every saved board or legal-mask entry.
Do not count a candidate until it survives the entire pipeline and supplies
recorded moves for the rank being counted.

Return a self-contained review page with rank/account/pass coverage, exclusions,
costs and fixed sample games at both komi values. Obtain a fresh independent
Codex review of code, data checks and claims before completing `kenggo-ydo.12`.

A later model comparison must freeze its own training/evaluation budget and
selection rule under [E0](e0_evaluation_protocol.md) and
[E2](go_project_plan_v0.5.md#e2-build-and-measure-human-baselines). Use matched
model/optimizer settings and training-compute limits for 6.5-only and combined
arms, with paired evaluation on the same fixed development positions. Report
6.5 and 0.5 results separately, alongside rank/phase/clock breakdowns. Do not
confound this test with the E3 history extension. Decide minimum useful NLL gain,
allowed regressions, seeds and complete-game checks before training. This plan
does not fund training or promote an opponent profile.

The [budget review page](../reports/e2_komi_study_review_2026-11-16.html)
preserves the proposal the owner approved. E2 readiness stays open.
