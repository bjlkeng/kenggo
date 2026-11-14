# E2 cached source-coverage audit

The one bounded attempt inspected 2,134 of 4,269 fixed candidates and admitted
415 at the source-screening stage. It stopped at its predeclared time reserve;
2,135 candidates remain unattempted. The admitted subset contains 295 provisional
training games, 97 population-development games and 23 unseen-account development
games. These are additional source IDs, not ready-to-train examples or qualified
rank profiles. Legal replay and trajectory deduplication have not run.

The audit inherits the approved 2025 source population and screening rules. Its
purpose is to measure the cache's usefulness before another model experiment.

## Observed coverage

| Provisional role | Fixed candidates | Attempted | Source-admitted | Distinct handle strings | Explicit passes | Games ending with two explicit passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 3,031 | 1,491 | 295 | 187 | 376 | 142 |
| Population development | 986 | 583 | 97 | 61 | 141 | 50 |
| Unseen-account development | 252 | 60 | 23 | 6 | 40 | 14 |

All 2,134 attempted members were retrieved and parsed; there were no retrieval
failures or header-quarantine failures. Screening reasons overlap: 1,207 had
out-of-scope rules/komi, 941 lacked an in-scope time control, 92 had an unverified
played variation and seven disagreed with the archive's result. Their rejection
counts must not be added as if mutually exclusive.

The fixed URL order is not a random sample. The attempted prefix is concentrated
at 4k: 292 training game sides from 69 handle strings, 165 population-development
sides from 45 handles, and all 46 unseen-account development sides from six
handles. Do not extrapolate this prefix's admission rate or rank mix to the
unattempted candidates. Handle counts are not claims of verified distinct people.

The 415 admitted source records contain 187 scored, 190 resignation, 25 forfeit,
12 timeout and one incomplete/unknown result. Timeout, forfeit and unknown
outcomes still cannot supply value labels. The 557 explicit pass actions remain
recorded actions; no file ending was converted into a synthetic pass or proof
of a legally completed game.

## Fixed scope

Metadata selection was frozen before any extension SGF was inspected. It yielded
4,269 additional URLs: 3,031 provisional training games, 986 population
development games and 252 unseen-account development games. Their sorted compact
JSON list has SHA-256
`d7ef1700e26310b5ce51be884156190eeac814ac45364a8de62f7ed476ded636`.

The selection excludes the original 1,000 candidates, either participant in the
reserved final cohort, documented and bot-named accounts, held-out prefixes,
confirmation dates and uncached bundles. It preserves the two owner exclusions.
Their exact SGF hashes remain excluded even under another URL; broader
trajectory deduplication remains a later build requirement.

The original 885-key handle map reproduces the existing split configuration's
grouping hash. New handles are appended without changing prior group assignments.
Only the final index's quarantine groups are used; its records and referenced
payloads are not traversed. Confirmation and final targets remain sealed.

The historical admission policy still applies to its original 1,000-ID audit.
This separate, pinned implementation decision extends the same pure screening
rules to the fixed additional list under the owner's instruction to continue E2.
It does not assert new owner approval or broaden the original commands.

## Resource and failure policy

The frozen envelope allows zero network requests and GPU allocation, one CPU
worker, 300 CPU seconds, 600 elapsed seconds including cleanup, 64 MiB of selected
SGFs, 1 MiB per SGF, 256 MiB of declared expanded ZIP content and 128 MiB of new
output. The process supervisor reserves time and output for metadata preparation,
final accounting and cleanup. Successful cached page and ZIP request evidence is
verified before selected members are opened. Failed or ambiguous exact members
remain failed candidates; they are not replaced.

Root SGF identities, ranks and dates must match the selected archive row before
full action/result diagnostics. Source admission preserves the ordinary-game,
nominal-clock and presumed-human rules and keeps identity, elapsed-duration and
dynamic-clock masks invalid. The output retains every attempted source, its
retrieval/admission decision and the original provenance pins.

## Limits that expansion cannot remove

Even the pre-SGF candidate frame cannot meet every exact-rank development support
requirement. The largest unseen-account candidate cell at 8k has 22 additional
mover handles; adding the current two game sides cannot reach 30 accounts.
Other cells cannot reach 100 game sides even before SGF filters. The remaining
cached frame reaches only 12k, so it cannot replenish 13k–15k. More training
examples from this cache would not by themselves qualify the proposed rank range.

Any later build must retain the existing cohorts, chronology, owner exclusions,
full legality history and missing-data masks; deduplicate, replay and construct
whole-game splits before writing examples; and refit the normalizer on training
rows alone. The exposure output is an addition to the prior registry, never a
replacement. No new training budget or difficulty label follows from source
coverage counts.

## Actual cost and checks

The supervised attempt consumed 27.332553 charged CPU seconds and 550.551648
elapsed seconds, with zero GPU seconds and zero new network requests. Its
9,245,253 counted output bytes were below 128 MiB. It read 10,042,185 selected SGF
bytes; the required ZIPs declared 37,464,515 expanded bytes. The worker body used about 12 CPU seconds before its 540-second cooperative
reserve stopped the loop. The CPU/elapsed gap is consistent with durable-write
cost, but per-operation timing was not measured. The full accounting adds
imports, verification, wrapper work and recorded preparation, with timer
rounding charged upward. Final accounting-file writes are outside that boundary.

The audit ran on one CPU worker during part of the separate BASE raw-play panel;
it performed no model inference. No automatic retry or budget extension followed
the partial result. The full frozen list remains conservatively exposed, including
unattempted IDs, and the prior exposure registry must be retained.

Independent verification of the saved records passed for all 2,134 rows: they
form the exact frozen prefix, role/admission counts match, hashes and source
membership match, no reserved-final group or owner exclusion was admitted, and
all identity, duration and dynamic-clock masks remain invalid. That JSON-only
check used 0.083199 CPU / 0.083205 elapsed seconds and made no SGF or model call.

The 18 synthetic worker/accounting tests passed. Fresh Astra/max prelaunch
review pass 2 was clean after fixing the first review's missing wrapper-cost
accounting. The regression checks include actual synthetic subprocess failure,
timeout and total-budget-breach cases. The final review scope and outcome are recorded in `kenggo-ydo.8.1`.
