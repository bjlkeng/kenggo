# Populated E1 data

The population build applies the owner-approved [presumed-human admission
policy](../reports/e1_kgs_admission_2026-11-11.md) to the original 1,000 records and
adds a separately reserved final cohort. It supports exploratory population/rank
baseline work. It does not establish exact-rank opponent profiles, verified human
identities, personalization, timing labels or calibrated playing strength.
The [current population manifest](../manifests/e1_population_v3.json) pins the
inputs after the [owner's two-game exclusion and manual sign-off](../reports/e1_manual_review_2026-11-13.md).
The [version-2 manifest](../manifests/e1_population_v2.json) retains the original
build, acquisition and integrity evidence for historical reproduction.

## Fixed cohorts

The [input manifest](../manifests/e1_population_inputs_v1.json) pins source
decisions, the final index and [split configuration](../manifests/e1_split_config_v2.json).
The original admission screened 241 games. The owner excluded two complete
training games (historical ending cases 044 and 083), leaving these current roles:

| Role | Games | Policy examples | Valid value targets |
| --- | ---: | ---: | ---: |
| Shared-model training | 106 | 22,916 | 20,888 |
| Population development | 14 | 3,363 | 3,363 |
| Population confirmation | 28 | 5,954 | 5,575 |
| Unseen-account development | 10 | 2,563 | 2,563 |
| Unseen-account confirmation | 22 | 4,508 | 3,750 |
| Earlier validation-account prefix | 59 | 0 | 0 |

The prefix contributes no adaptation examples because stable identity is absent.
Training dates precede 1 September 2025, development occupies September–October,
and confirmation starts 1 November. Validation groups are selected from the 679
originally observed handle groups by
`SHA256("kenggo-6yr.7-population-splits-v2:validation:" + group) mod 5 == 0`.
This selects 142 groups before model training. Preserve all 695 exposed groups in
the final exclusion registry, including discovery-only accounts.

The [final reservation](../manifests/e1_final_reservation_v1.json) selects 20
previously unexposed seed groups from retained metadata. Forty fixed November and
December account-month pages supplied the metadata frame. Before fetching SGFs,
the acquisition froze 200 candidate source IDs and all their participant groups,
producing 190 reserved groups. Both sides must be outside the exposure registry;
bot-like handle names are conservatively excluded. Failed or excluded candidates
are retained without replacements. The [final scope
policy](../manifests/e1_final_scope_policy_v1.json) applies the same source/rank/rules/
clock/permission decision to this reservation.

Mechanical processing finds **70 usable final games**; the other 130 candidates
remain sealed and excluded. No final moves, outcomes, rank-performance summaries
or referee judgments were inspected to select settings. These are source-handle
groups, not verified people. Earlier final-account games were not collected, so
this cohort supplies no personalization prefix.

The current build contains **54,527 policy examples**: 39,304 nonfinal and 15,223
sealed final examples, with 36,139 valid nonfinal source outcomes. The derivation
removes 479 training rows and preserves every field of the 22,916 remaining
training rows. All other example shards and all unrelated split files are
byte-identical to version 2. Final arrays were copied and hashed without decoding
their targets. The refitted train-only normalizer and derived split/profile/example
manifests reproduced byte for byte. Historical source replay and encoding
diagnostics remain available in the version-2 evidence.

## Use the current inputs

Reproduce the owner-approved filter from the pinned local version-2 artifacts:

```bash
uv run --locked python tools/apply_e1_review.py runs/e1-manual-filter-repeat
```

Use a new output directory. The command verifies the owner decision and source
pins, changes only the two whole-game memberships, filters stored training rows,
copies other shards without decoding final targets, and refits the normalizer.
It makes no HTTP requests or model calls. The limits are 120 CPU seconds,
180 wall seconds and 256 MiB of output. The existing completed output is
`runs/e1-manual-filter-2026-11-13/`.

Current consumers use the [version-3 profile](../manifests/e1_population_example_profile_v3.json)
and the current example manifest:

```python
from kenggo.examples import load_dataset_profile, load_corpus, read_shards

profile = load_dataset_profile(
    "manifests/e1_population_example_profile_v3.json",
    "3093ee6cb6ea0355ab2314884bce588dbd7f018ea7e809dd63ab1f7be7ea3036",
)
records, split, partitions = load_corpus(
    "runs/e1-population-2026-11-11/canonical.parquet", profile.canonical_sha256,
    "runs/e1-manual-filter-2026-11-13/splits", profile.split_sha256, profile=profile,
)
training = read_shards(
    "runs/e1-manual-filter-2026-11-13/examples",
    "3ba84e660df1e50478994ec98c8c4b25132559034328f3de21a6ac2e527cd4c4", "train",
)
```

Pair these arrays with `runs/e1-manual-filter-2026-11-13/normalizer.json`, pinned
by the current population manifest. Canonical records preserve historical
admission evidence; select usable games through the current memberships. Both
reviewed games have role `excluded` with zero policy/value actions. Preserve the
[manual exclusion policy](../manifests/e1_manual_review_v1.json) in later expansions.

## Reproduce the historical version-2 build

Use the [locked CPU workspace](workspace.md). The original and sealed source
archives must already be available at their pinned paths. Reproduction needs no
KGS requests. Use new output directories; commands never overwrite prior runs.

```bash
uv run --locked python -m kenggo.population_audit \
  manifests/e1_population_inputs_v1.json runs/e1-population-repeat \
  --inputs-sha256 eb757c00a387ee598af672378d6848c826dd064dbeecabfd1921c97798cc8fe0

uv run --locked python -m kenggo.examples_audit runs/e1-population-repeat \
  --profile manifests/e1_population_example_profile_v2.json \
  --profile-sha256 d05c952f9dc5a625c3b4f63e35c2b2805634d99ef08652dd543b563a55e0a836 \
  --probe-train-games 5

uv run --locked python -m kenggo.examples_audit \
  runs/e1-population-repeat runs/e1-examples-repeat \
  --profile manifests/e1_population_example_profile_v2.json \
  --profile-sha256 d05c952f9dc5a625c3b4f63e35c2b2805634d99ef08652dd543b563a55e0a836
```

The canonical build is bounded at 1,200 records, 64 MiB selected SGFs, 256 MiB
expanded ZIP/JSON content, 900 CPU seconds and 1,800 wall seconds. The positive
five-training-game encoding probe produced 1,237 examples in 54.25 CPU seconds.
The full example profile reserves 4,000 CPU seconds and 6,000 wall seconds, with
100,000 examples maximum per role; the completed build used 2,362.27 CPU seconds
and 2,362.24 wall seconds. These are single-run cost observations and
limits, not comparative performance claims. All work is CPU-only and serial
within each builder; external source/referee costs are recorded separately.

Apply the current filter after reproducing historical inputs. The old
canonical/example commands and version-1 pins also remain available for the
earlier empty corpus. New production consumers use the version-3 profile above.
Never infer a trusted hash from the incoming manifest itself.

## Data access and missingness

`load_corpus(..., profile=profile)` excludes sealed source IDs at the Parquet read
and returns only development data. The private mechanical builder can create
final arrays without printing their contents. Diagnostics select only nonfinal
records; the normalizer reads only the `train` shards. Public final-shard access
requires `load_final_release(path, expected_sha256)`, binding canonical, split and
example hashes to an explicit `stage="E7"`, `selection_frozen=true` authorization.
No E7 release exists for this build. This is a workflow guard, not encryption or
an operating-system access boundary.

Preserve the full replay history and original source results. No engine cleanup
becomes a human action or value label. Invalid/absent/timeout/forfeit results stay
value-masked. Every duration target and dynamic-clock input remains masked; only
the static clock tuple is usable. Identity/persona remains the default path.
Seeded training rank masks retain the historical labels in separate provenance.

All proposed raw profiles remain deferred: even pooled usable nonfinal records
have at most 62 game-sides and 25 handle groups at one exact rank, below E0's
100-side/30-account promotion threshold. The training-only cells are smaller.
E2 may investigate a rank-conditioned baseline and report uncertainty; E5 still
requires independent complete-game qualification, and human-rank labels still
require E7 calibration. B continues to start from random weights.
