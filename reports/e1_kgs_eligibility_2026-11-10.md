# KGS eligibility and account grouping, version 1

**Decision, 10 November 2026:** the frozen 1,000-game audit supplies **zero
training-eligible games** under `kgs_eligibility_v1`. The 241 metadata-compatible
games remain unverified for historical human control, ordinary game context and
downstream training use. Use conservative handle groups for split controls, and
keep every persona on the default, masked path. E1 readiness remains blocked.

This completes the evidence and policy investigation in `kenggo-6yr.9`; it does
not complete a human corpus. The [machine-readable manifest](../manifests/e1_kgs_eligibility_v1.json)
and [per-game decisions](../manifests/e1_kgs_eligibility_games_2026-11-10.parquet)
preserve this result. The [original audit](e1_kgs_audit_2026-11-10.md) is unchanged.
These are counts within a convenience sample, not estimates of KGS-wide coverage.

## Evidence and admission rules

| Question | Evidence | Version-1 treatment |
| --- | --- | --- |
| Does a rank prove a human player? | KGS permits administrator-approved ranked bots. The documented historical Game Summary player flags are reliable only for rank. | Human status remains unverified. Registration, rank, ordinary authorization level and absence of a bot flag cannot admit a game. |
| Are known bot handles usable? | The discovery revision records a historical operator statement naming GnuGo2 and LibertyBot. | Exclude both whole handle groups, including any separately reviewed aliases. This is a conservative exclusion, not a claim that every later account generation was a bot. |
| Does archive `Type=Ranked` establish ordinary play? | The JSON schema distinguishes ranked, tournament, simul, teaching and other game types. The help also describes a separate tournament marker. The cached HTML frame has only Ranked, Free and Review examples. | The archive HTML mapping has no validated tournament/simul contrast. Keep Ranked unverified. Explicit Free, Review, Teaching, Demo, Simul, Rengo, Tournament, Private and KGS Plus types are excluded. |
| Do absent SGF event fields establish ordinary play? | None of the selected records has `EV` or `RO`. | Absence proves nothing about event context. Present event/round metadata requires classification and is excluded in v1. |
| Do handles identify persistent accounts or people? | Names are case-insensitive. Expired names can be registered by the same or another person while earlier games remain archived. The inspected fields supply no persistent account-generation identifier. | Use whole handle groups across all dates; do not infer generations or people. Personalization remains disabled. |

Sources: [KGS bot FAQ](https://www.gokgs.com/help/faq/bots.html),
[JSON data types](https://www.gokgs.com/json/dataTypes.html),
[game types](https://www.gokgs.com/help/gametypes.html),
[account names](https://www.gokgs.com/help/app/main.html),
[account deletion](https://www.gokgs.com/help/_vi_VN/Accounts.html), and the
[historical operator statement](https://groups.google.com/g/computer-go-archive/c/PuNvbA19DdM).
The deletion page has a localized navigation path but English account-deletion
content. Captures and hashes are listed in the manifest.

[`kenggo.kgs_eligibility`](../src/kenggo/kgs_eligibility.py) enforces these choices.
It preserves the original metadata exclusions and emits either `excluded` or
`unverified`; **v1 has no positive admission path**. Incoming fields such as
`human_status`, `stable_account_ids_verified`, `ordinary_verified`,
`source_reuse_permitted` or `training_ready` cannot grant eligibility. A positive
path requires an evidence-backed, reviewed policy revision. The decision table
is not a canonical dataset or a replacement for replay, masks, deduplication and
splits.

## Handle groups and personalization

Normalize a valid documented KGS name with ASCII lowercase, preserving the
original spelling separately. Its initial group is `kgs-handle:<lowercase name>`.
The same handle stays in one group across the entire corpus, including all
possible deleted/reused generations. KGS describes expiry after six months
without login for regular accounts and two years for accounts with KGS Plus;
game gaps do not measure login activity. Dates, rank changes or inactivity in
the archive therefore cannot justify splitting a handle.

Only explicit alias links supported by separately reviewed evidence may merge
groups. Apply their transitive closure and use the lexicographically smallest
normalized handle as the representative. There are **no accepted alias links**
in v1. Similar names, shared opponents, ranks and playing style never create
links. Opponents in a game remain distinct groups. Reject a game whose two
players resolve to the same group; missing or invalid handles never share an
invented unknown group.

This can keep unrelated people who reused a name together. It cannot detect
undocumented multiple accounts or account sharing. The supported claim is
**disjoint handle groups under the frozen alias registry**, not person-disjoint
cohorts. The canonical builder must still apply both-player exclusions for
reserved validation/test groups, exclude games bridging those cohorts, and
split whole games chronologically after deduplication.

Group keys are split controls, not persona embeddings. Every side has
`identity_valid=false` and `persona="default"`, even when its handle is known.
Do not merge all default personas into one split group. Account-generation
evidence, adequate history, later untouched games and E3's controlled comparisons
remain prerequisites for personalization; nothing here qualifies a persona.

## Final-evaluation exposure

Reserve the union of both participant groups in every inspected selected game,
all 48 original discovery handles (including the two excluded bot handles), and
the preliminary `cu2` probe handle. Propagate through accepted alias links across
all dates. This produces **695 quarantined groups**: 679 observed in the sample,
15 additional original discovery handles, and the probe handle. No final-test
cohort was created or inspected during this work.

Preserve the 1,000 selected source IDs, their exact SGF hashes and sequence
fingerprints, the original discovery account-month slots and the September 2025
probe slot as separate exclusion predicates. Failed or later relabeled records
remain reserved by source ID; copies with new IDs remain reserved by bytes or
sequence. Matching sequences are conservatively quarantined here without
asserting that they prove duplicate source games. Unknown dates or groups also
exclude a record from final evaluation.

A later game between an exposed handle and a new opponent is excluded, but it
does not recursively expose the new opponent. Only actually inspected records
or explicit alias evidence extend the exposure registry. Changing the group
registry invalidates its digest: rebuild exposure and affected splits before
using it. Never silently keep old cohort assignments after an alias merges
reserved groups. `final_evaluation_excluded=false` means only that this function
found no exposure predicate; it does not establish admission or a valid split.

## Continued collection and use

The owner authorized bounded authenticated acquisition with gentle server load.
That supports the recorded acquisition and local audit/engineering work. The
inspected [robots file](https://www.gokgs.com/robots.txt) disallows
`/gameArchives.jsp`; the [service terms](https://www.gokgs.com/tos.jsp) describe
service conduct and privacy but do not establish a blanket research, ML or
redistribution licence. Successful login and owner authorization are not
asserted to be KGS automation approval or a downstream licence. This is an
unresolved evidence boundary, not a finding that the terms forbid all research.

Continue offline analysis of the retained evidence and implementation with
synthetic fixtures. Any further owner-directed audit collection needs a recorded
finite purpose, population and budget under the existing controls: one request
at a time, at least five seconds between starts including redirects, no automatic
retries, and a stop on errors or caps. The old discovery deadline is expired and
does not authorize an open-ended job. Resolve the source conditions for the
proposed collection/use before scaling or adopting a training corpus; do not
infer permission from this policy's existence. No administrator contact or other
external message was sent or authorized in this bead.

This version records `local_audit_only` as the demonstrated use scope and
`training_reuse_basis_unresolved` for every record. No training, raw-corpus
redistribution, additional game download, login or live-server session occurred.
The versioned Parquet contains only derived decisions, handles, source IDs and
hashes; no SGF moves, comments, credentials or cookies.

## Recomputed coverage

Of the 1,000 inspected records, **759 remain excluded** by the inherited audit
filters and **241 are unverified**. Twenty-nine records include a historical bot
handle; the other 971 have no affirmative human classification. All 1,000 have
unverified ordinary-game status. All are reserved from final evaluation.

The 241 metadata-compatible games contain 216 handle groups, 52,900 recorded
actions including 266 explicit passes, and 108 endings with two explicit passes.
Their results are 102 scored, 114 resignations, 16 timeouts, 6 forfeits and 3
incomplete/unknown. These counters do not become usable labels simply because
the source parsed. Separate [rules validation](../docs/rules_engine.md) replayed
the 52,900 actions against pinned KataGo without disagreement, but its automatic
certifier resolved **0 of the 108 endings**. That is a separate readiness blocker.

| Historical rank | Metadata-compatible player-game appearances | Recorded actions | Explicit passes | Two-pass ending appearances | Eligible player-game appearances |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15k | 10 | 897 | 5 | 4 | 0 |
| 14k | 2 | 119 | 1 | 1 | 0 |
| 13k | 3 | 376 | 2 | 2 | 0 |
| 12k | 27 | 2,841 | 29 | 23 | 0 |
| 11k | 29 | 2,964 | 21 | 15 | 0 |
| 10k | 22 | 3,091 | 14 | 13 | 0 |
| 9k | 55 | 6,141 | 34 | 30 | 0 |
| 8k | 11 | 1,138 | 7 | 7 | 0 |
| 7k | 21 | 2,756 | 16 | 13 | 0 |
| 6k | 12 | 1,282 | 7 | 6 | 0 |
| 5k | 5 | 448 | 1 | 1 | 0 |
| 4k | 57 | 6,687 | 43 | 27 | 0 |
| 3k | 48 | 4,712 | 28 | 24 | 0 |
| 2k | 44 | 5,067 | 18 | 16 | 0 |
| 1k | 56 | 6,350 | 27 | 22 | 0 |
| 1d | 80 | 8,031 | 13 | 12 | 0 |
| Total | 482 | 52,900 | 266 | 216 | 0 |

Each game contributes one appearance per player, so same-rank games contribute
twice to that rank. Ending appearances similarly count both sides. All eligible
rank, action, pass, result and ending counts are zero. There is no supported raw
profile, personalized profile or calibrated rank claim.

## Reproduction and cost

With the retained `sample.json` and the [locked CPU workspace](../docs/workspace.md),
run from the repository root into a fresh output directory:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv run --locked python -m kenggo.kgs_eligibility \
  manifests/e1_kgs_audit_games_2026-11-10.parquet \
  runs/e1-kgs-audit-2026-11-10-v4/sample.json \
  manifests/e1_kgs_discovery_2026-11-10_v2.json \
  runs/e1-kgs-eligibility-reproduced
uv run --locked python -m unittest discover -s tests -p test_kgs_eligibility.py -v
```

The command verifies the frozen audit/sample hashes, discovery provenance and
exact 1,000 source IDs, then writes `decisions.parquet`, `decisions.json` and
`summary.json`. It reads no raw SGFs and makes no HTTP requests. Fresh clones
can inspect the committed evidence and run behavior tests; full regeneration
also needs the retained sample. The input ceiling is 64 MiB and the computation
limit is 300 CPU seconds / 600 wall seconds, with one audit worker and no GPU.
The two final runs produced identical decision Parquet bytes and aggregate
results apart from timing. These are reproduction checks, not speed comparisons.

Four new official help-page GETs used one connection with at least five seconds
between starts and no retries: 31,134 response bytes, 15.062 seconds wall time
and 0.019 CPU seconds. Existing terms, robots and schema captures were reused.
The predeclared ceiling was 12 pages, 1 MiB per page / 12 MiB total, 120 CPU
seconds / 600 wall seconds. Three separate web-search calls (five queries)
located documentation; their provider-side request counts and compute were
not measured. The manifest records each local policy run, capture hashes and
measurement scope. Authoring, tests and independent review are outside the
reported acquisition/audit timings.

## Capability decision

The conservative grouping and negative admission policy are usable by the
canonical builder. `kenggo-6yr.11` owns affirmative historical human-control
evidence, an established ordinary-game mapping and a documented training-use
basis, or an explicit reviewed source/population rescope followed by an audit.
Evidence must identify the affected games/account generations, its source and
time coverage; a present-day bot list or unverified boolean is insufficient.
Any accepted revision must recompute coverage and preserve quarantine and masks.

Canonical construction and example encoding (`6yr.5`/`.6`) can proceed with
synthetic behavior checks while refusing the current sample for training.
Usable certified endings (`6yr.10`) remain independently unresolved. E1 readiness
(`6yr.7`), shared-model training and all proposed opponent profiles stay blocked.
