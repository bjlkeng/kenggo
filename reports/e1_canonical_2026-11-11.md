# Canonical KGS records and whole-game splits

This report describes the frozen version-1 artifacts. The later
[populated data build](../docs/e1_population.md) uses the approved admission policy
and a separately sealed final cohort; its new pins preserve this reproduction.

**11 November 2026, UTC — `kenggo-6yr.5`.** The canonical builder retains all
1,000 selected games and their original SGF references. It validates replay,
preserves missingness and provenance, and writes deterministic whole-game
partitions. The current sample supplies **zero admitted training, development,
confirmation, adaptation or final-evaluation games** under
[`kgs_eligibility_v1`](e1_kgs_eligibility_2026-11-10.md).

The [run manifest](../manifests/e1_canonical_v1.json),
[per-game index](../manifests/e1_canonical_index_2026-11-11.parquet) and
[split configuration](../manifests/e1_split_config_v1.json) are versioned evidence.
Full canonical records, including moves and raw clocks, remain in ignored local
run storage. The frozen [original audit](e1_kgs_audit_2026-11-10.md) is unchanged.

## Source evidence and replay

[`canonical_audit`](../src/kenggo/canonical_audit.py) pins the original audit,
sample, discovery provenance and eligibility decisions. It recomputes the
eligibility decisions before opening any SGF. Incoming affirmative flags cannot
replace that evidence. All 103 ZIP hashes are verified, and each selected source
ID must map to one exact date/filename member with its original SGF hash. Changed
or ambiguous input aborts the run; nothing is resampled.

[`canonicalize`](../src/kenggo/canonical_data.py) retains source URLs, ZIP/member
references and hashes, archive metadata, SGF metadata, original results and
explicit actions. It records normalized board/pass actions and their original
SGF encoding. The leftmost line of a branching record is retained only for
diagnosis; an ambiguous played line, setup, illegal move or unsupported action
after two passes excludes the entire game. No valid prefix is salvaged for
training. EOF adds no pass, resignation or terminal event.

Replay uses the [shared rules adapter](../docs/rules_engine.md), preserving the
whole actual action history. The source result and the adapter's end state stay
separate: `RE[B+R]` does not insert a resignation action into the source history.
Certified outcomes, when available, are checked against known source outcomes;
unresolved adjudication does not invent a score. A future encoder must replay
the complete canonical history before taking a limited model-context window.

Each adjudication retains its versioned method, snapshot, proof payload, reason,
score, resource limits and node count in the canonical record. `canonicalize`
returns that deterministic record and the full per-call execution cost
separately. The builder writes `adjudications.json`, linking every measured cost
to its source ID, SGF hash, canonical-record hash and certificate hash. Its hash
and total measured costs are recorded in the run summary. Separating CPU/wall
timings keeps the canonical and split artifacts reproducible without discarding
proofs or per-call budget evidence.

Missing results, timeouts, forfeits, unknown results and unsupported draws have
no value target. Otherwise consistent recorded scores/resignations may supply
`recorded_y_black` of +1 or −1 after the structural checks; this remains a
diagnostic label until source and split gates pass. Two known conflicting
results quarantine the game. Source identity, rank or date conflicts also
quarantine it. At Japanese/6.5, a recorded numeric score must have a positive
odd number of half-points.

All present `BL/WL/OB/OW` values are retained by SGF node, including non-move
nodes, as raw hexadecimal byte lists with parsed values and parsing masks.
An absent property has no entry; invalid or repeated values keep their raw
bytes with a false parsing mask. Dynamic-clock inputs and duration targets
remain masked in v1. These post-move values and recorded outcomes are never
authorized as pre-move model inputs by this artifact.

## Duplicate handling

[`data_splits`](../src/kenggo/data_splits.py) constructs duplicate components
before filtering any record. Edges use exact SGF bytes, the original audit
fingerprint, or the normalized complete action sequence with board/rule/komi
settings. Normalization connects legacy `tt` passes with explicit empty passes.
Zero-move sequence coincidences do not establish duplicate games.

All component members remain traceable. A component with conflicting canonical
participants, dates, ranks, results, clocks or other retained semantic evidence
is quarantined. Any member's invalidity or failed source admission excludes
the component; choosing a cleaner copy cannot hide that evidence. Only coherent
copies select one deterministic source-URL representative, with the other copies
marked `duplicate_copy`. Sequence equality is conservative collision evidence,
not proof that two independently played short games are the same game.

Final-evaluation exposure propagates through the whole component. Exposure alone
does not exclude an otherwise admitted game from nonfinal work. The reviewed
handle-group policy still reserves all 695 inspected/discovery/probe groups
across dates; none can be assigned to an unseen final-test cohort.

## Frozen cohort and date rules

The explicit configuration chooses **2025-09-01** as the first development day
and **2025-11-01** as the first confirmation/final day. These are project choices
for this implementation, not fitted coverage thresholds. Both reserved-group
lists are empty for the current inspected audit. No usable or unseen final
cohort was created.

Dates retain source calendar-day precision and unknown timezone. No UTC midnight
or within-day ordering is invented. URL order makes output deterministic; it
does not prove chronology. Both colors participate in every cohort check.
Overlapping reserved sets are rejected, and validation–test bridging games are
excluded at every date, including prefix dates.

| Participants | Before 1 September | September–October | November–December |
| --- | --- | --- | --- |
| Neither group reserved | Shared training | Population development | Population confirmation |
| Any validation group, no test group | Validation adaptation prefix | Unseen validation development | Unseen validation confirmation |
| Any test group, no validation group | Test adaptation prefix | Embargo, unused | Reserved final |
| Validation–test bridge or excluded record | Excluded | Excluded | Excluded |

Shared training and fitted normalizers may consume only the `train` partition.
Test-group games never enter selection; validation-group games never enter
shared training. Development and confirmation games are disjoint. Both use the
same earlier adaptation prefix; neither can enlarge it after evaluation.

Adaptation permission is per target color and group. A known opponent in a
held-out prefix is not an additional adaptation target. Prefix policy-action
counts include only the permitted target sides, and prefix value-action counts
are zero because personalization is policy-only. Missing identity evidence
preserves prefix membership but grants no adaptation actions. Known-persona
annotations require valid identity evidence and actual earlier shared-training
actions by that side; an account first appearing later remains on the default
path. These annotations describe data availability, not a trained embedding.

The future trainer must freeze shared weights and use a validation-selected
adaptation procedure; only the target embedding may change. Confirmation needs
a frozen candidate, and final evaluation stays unopened until selection is
complete. The manifest records those execution requirements. Hashes prove
content consistency, not that an evaluator followed an unopened-holdout policy.

## Artifacts and caller boundary

Each fresh run writes `canonical.parquet`, a small `index.parquet`, and separate
Parquet membership tables for every role under `splits/`, including explicit
empty schemas. The adjudication execution ledger remains a linked local run
artifact. `splits/manifest.json` binds the canonical input, configuration,
group registry, exposure registry, source evidence, lockfile and implementation
hashes. Existing output directories are never replaced.

The pure split core accepts trusted canonical entries. Synthetic tests exercise
positive admission with fixture-only manifests; the production KGS adapter has
no positive admission route. `read_membership(directory, expected_sha256)` checks
an independently pinned manifest hash and each membership artifact before
returning rows. Each row identifies its canonical-record hash. The example
encoder must verify those hashes against the canonical payload and apply its
role, color and label masks. A manifest's own hash is not an independent grant
of source permission.

## Observed result

| Check | Result |
| --- | ---: |
| Frozen source IDs retained and parsed | 1,000 |
| Structurally supported and legally replayed games | 243 |
| Their observed actions / explicit passes | 53,151 / 268 |
| Distinct handle groups in those games | 218 |
| Structurally supported score/resignation labels | 216 |
| Two-pass endings / automatically certified endings | 108 / 0 |
| Nonempty duplicate copies removed | 0 |
| Source-admitted games | 0 |
| Shared-training, development, confirmation, adaptation and final games | 0 |

The 243-game diagnostic subset has 102 scored results, 114 resignations, 16
timeouts, 6 forfeits and 5 unknown/missing results. It adds two games to the
original audit's 241 metadata-compatible records: their absent SGF results and
archive forfeits now mask value without invalidating legal policy actions. The
other three such records still fail other structural filters. The eligibility
policy continues to exclude all of them from training.

All 224,274 observed actions are retained locally for diagnosis; 53,151 pass the
replay checks. The earlier native-reference comparison covered the original
52,900 actions, not a new native comparison of the two added games. The 757
replay rejections include unsupported settings and ambiguous lines; they are
not a claim of 757 illegal games. The earlier three excess zero-move sequence
fingerprints remain separate excluded records. No game is admitted by changing
an exclusion denominator.

## Reproduce and validate

From the repository root with the [locked CPU workspace](../docs/workspace.md)
and retained acquisition cache:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv run --locked python -m kenggo.canonical_audit \
  runs/e1-kgs-audit-2026-11-10-v4 \
  runs/e1-canonical-reproduced
uv run --locked python -m unittest discover -s tests -p 'test_canonical*.py' -v
uv run --locked python -m unittest discover -s tests -p test_data_splits.py -v
```

The command requires a fresh destination and performs no network requests. It
verifies inputs before reading only the selected SGFs; extra ZIP members remain
unparsed. Limits are one audit worker, 1 MiB per SGF, 64 MiB of selected SGFs,
256 MiB of declared expanded ZIP content, 256 MiB serialized canonical JSON,
900 CPU seconds and 1,800 wall seconds per full pass. No GPU is used.

The manifest records every complete owner pass and the early failed setup attempt.
That attempt read metadata only and stopped at an incorrect manifest nesting
assumption; its CPU time was not measured. The final two runs reproduce the
canonical/index bytes and complete split manifests exactly; adjudication timing
ledgers vary by execution and are verified against their respective recorded
hashes. Independent review found that the first version omitted adjudicator
proofs and per-call budgets; the final artifacts retain both, with regressions
covering unresolved decisions and successful certificates. Timings cover
input verification, parsing/replay and artifact writing, excluding interpreter
imports, the final summary write, authoring, tests and independent review.
They are cost accounting, not a performance comparison.

Canonical/split implementation is complete while the usable corpus remains
empty. Example encoding and masks belong to `kenggo-6yr.6`; affirmative source
evidence (`6yr.11`) and usable certified endings (`6yr.10`) continue to block
E1 readiness. No model, persona or opponent profile is qualified by this work.
