# KataGo Human SL reference

`HumanSL` returns the auxiliary model's raw `humanPolicy` and its explicit
`rootInfo.humanWinrate`. It uses KataGo 1.18.1, the `b18c384nbt-humanv0` human
checkpoint and the existing b10 main checkpoint. The main network is needed by
this analysis configuration; its policy and values do not become Human SL
outputs. Both networks' execution costs belong to the reference.

The [implementation report](../reports/e2_human_reference_2026-11-13.md) records
the actual check, retained failures and resource costs.

The [profile](../manifests/katago_human_profile_v1.json) pins the executable,
checkpoints, config and bundled libraries. Its independently supplied SHA-256 is
`f78423b5ba58485a7ffeee38bb7b39fcb7c797083f69c1389112ed1e04863d5b`.
The adapter rejects artifact, version or required-setting mismatches. It uses
the [shared Japanese rules](rules_engine.md), komi 6.5, every actual prior move
including passes, one analysis visit, a fixed symmetry and no root noise.
Fixture boards require explicit permission and cannot contain initial prisoner
offsets that the query cannot represent.

The config supplies a `rank_5k` startup profile to avoid KataGo's unset-profile
warning. Every prediction still requires its own explicit rank and overrides
that startup value. There is no default rank at the Python call boundary.

The policy has 362 entries in top-left row-major order, with pass last. Native
illegal entries stay `-1` in `raw_policy`; `policy` exposes them as zero and
preserves every legal probability. The shared rules engine independently checks
the complete legal mask. The legal mass must be within 0.0005 of one, allowing
native decimal rounding without changing the stored probabilities.
Temperature-1 sampling uses a private seeded RNG and
includes the reported pass mass. There is no pass correction, search-policy
sampling or resignation. A future corrected profile needs a separate name,
evaluation and cost record. The value converts the explicitly White-oriented
human win probability to the current mover's `[-1, 1]` perspective.

These field meanings and the raw-play procedure come from KataGo's
[pinned analysis guide](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/docs/Analysis_Engine.md).

## Prepare the pinned files

The [referee setup](rules_engine.md#empirical-ending-referee) supplies the
Linux AVX2 executable, b10 checkpoint and bundled libraries. Those existing
files are reused. The extra human model comes from the official
[v1.15.0 release](https://github.com/lightvector/KataGo/releases/tag/v1.15.0).
Its expected size is 99,066,230 bytes; its SHA-256 is
`637746e44f0efe00ad1245a50aa9bbf0716efe364c43965ead97bd6835d84ab5`.
The [acquisition manifest](../manifests/e2_human_acquisition_v1.json) records the
actual download, source metadata, request/byte caps and measured costs.

If that file is absent, download it into a new ignored directory, within a
recorded acquisition allowance:

```bash
mkdir -p .cache/katago-human
curl --fail --location --max-redirs 5 --retry 0 --max-time 180 --max-filesize 99066230 \
  --output .cache/katago-human/b18c384nbt-humanv0.bin.gz.partial \
  https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz
sha256sum .cache/katago-human/b18c384nbt-humanv0.bin.gz.partial
```

Choose an unused partial-file path for each attempt. After the size and digest match, rename it to
`b18c384nbt-humanv0.bin.gz`. Preserve failed or partial attempts and charge them
to the same allowance. The adapter verifies the digest again before execution.

## Call the adapter

```python
import random
from kenggo.go import Game
from kenggo.katago_human import HumanProfile, HumanSL, ReferenceBudget

profile = HumanProfile.load(
    "manifests/katago_human_profile_v1.json",
    "f78423b5ba58485a7ffeee38bb7b39fcb7c797083f69c1389112ed1e04863d5b",
)
with HumanSL(
    profile,
    engine=".cache/katago-referee/squashfs-root/usr/bin/katago",
    main_model=".cache/katago-referee/kata1-b10c128-s1141046784-d204142634.txt.gz",
    human_model=".cache/katago-human/b18c384nbt-humanv0.bin.gz",
    config="configs/katago_human_v1.cfg",
    runtime_root=".cache/katago-referee/squashfs-root",
    budget=ReferenceBudget(queries=1, cpu_seconds=30, wall_seconds=60),
) as reference:
    reference.query_version()
    prediction = reference.predict(Game.new(), rank="5k")
    action = prediction.sample(random.Random(20261113))
cost = reference.cost()
```

Every client has finite query, CPU, wall and cumulative byte limits. Calls and
traces remain recorded after failure; failures are not retried automatically.
Startup, inference and shutdown enter the cost record. The Eigen backend uses
zero GPU allocation. Analysis requests are counted separately from neural
forward calls, which remain unknown because KataGo's internal cache does not
expose an exact counter through this interface. Future GPU backends require
separate profiles and allocation accounting.

`human_agent_profile` and `HumanMatchFactory` connect raw sampling to the
[match runner](matches.md). The factory checks that the scheduled profile binds
the reference pins and explicit conditioning rank. Missing or corrupted setup
artifacts and administrative adapter-budget exhaustion are infrastructure
failures and remain unresolved. Invalid replies from a running agent retain
their playing-fault attribution. The factory starts the native process and
verifies its version during setup, before the first playing decision. That
metadata request consumes the same CPU/wall/byte budget as later queries.
The schedule still
records untimed play, missing clocks and an unknown opponent rank; the Human SL
model's different fixed defaults are recorded separately.

Pass the same explicit `ReferenceBudget` to the profile helper and factory;
`reference_limits` is part of the scheduled profile. A factory also requires the
configured source revision and rejects a schedule with different pins or limits.

## Bounded implementation check

From the repository root, after staging intended new source files:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
timeout 300s uv run --locked python -m kenggo.human_reference_smoke \
  configs/e2_human_smoke_v1.json runs/e2-human-reference-repeat \
  --config-sha256 4194886877da7220eca46db17d02717dad0a66673382b7c6c913d32a27593f56
```

Use a new output directory and reserve the full allowance before any repeat.
The check is limited to 12 one-visit synthetic queries plus version metadata,
180 CPU seconds, 300 elapsed seconds, 16 MiB of output and no GPU. It preserves
the replies, exact repeat comparisons, raw pass probabilities, private-RNG
samples and a W&B run. Five positions are queried twice at `5k`; the empty board
also uses `15k` and `1d`. Two fixture states exercise immediate ko and ko cleared
after a pass. These are adapter checks, with no source labels, throughput claim
or baseline selection.

Two earlier metadata-only attempts stopped on the unset-profile warning before
any analysis query. Their [preflight costs](../manifests/e2_human_preflight_v1.json)
are retained and deducted from the check's aggregate allowance. Failed traces
now include the bounded stderr tail. Other native warnings still stop the client.

## Comparison limits

The rank labels select model metadata; they do not establish playing strength.
The pinned rank profiles use KGS, a 2020-03-01 date, 20-minute main time,
five 30-second byo-yomi periods and the same rank for both players. These defaults
come from the
[pinned metadata implementation](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/cpp/neuralnet/sgfmetadata.cpp).

The project's KGS games are from 2025, after the July 2024 Human SL release.
Exact clock settings and opponent ranks can differ; source human control and
training overlap remain unverified. Missing dynamic clocks in the project do
not remove the reference's nominal clock metadata. Untimed play against weak
policies also differs from these defaults. The reference receives all available
history, while the compact learner currently encodes zero history. Report that
input difference alongside any paired comparison.

The current adapter accepts explicit `15k` through `1d` profiles. Unknown or
unsupported mover ranks need a declared common-set exclusion or separately
evaluated fallback before a BASE comparison. No rank is silently substituted.
Human-value calibration against recorded outcomes remains a later measurement;
availability of the field does not establish calibration or search suitability.
