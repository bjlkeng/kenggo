# Cached source coverage for E2

The [coverage report](../reports/e2_cached_coverage_2026-11-14.md) records one
bounded, partial audit of additional games in the existing KGS cache. The
[evidence manifest](../manifests/e2_cached_coverage_v1.json) pins the actual
request, frozen candidates, code, execution configuration, outputs and costs.
It does not replace the original E1 admission policy or the existing corpus.

`kenggo.cached_coverage_audit.freeze_inputs` reads pinned archive metadata,
verifies retained response and ZIP hashes, preserves the original groups and
cohorts, and writes the explicit candidate list before inspecting SGFs.
`run_audit` revalidates that freeze, checks selected member headers before action
and result diagnostics, and retains all attempted outcomes. Production requires
exactly 4,269 IDs with the recorded list hash. The separate fixture mode accepts
at most 64 synthetic IDs and is unavailable from the CLI.

The [versioned request](../configs/e2_cached_coverage_v1.json) is byte-identical
to the actual request. It adds new handles without changing prior aliases,
excludes original candidates and both reserved-final participants, and opens
only training and development dates. Confirmation and final targets remain
sealed. Identity, elapsed-duration and dynamic-clock masks stay invalid.

The worker must run inside the process supervisor in
[run_cached_coverage.py](../tools/run_cached_coverage.py). Its default entry point
measures the complete inner wrapper with a pinned GNU `time` binary, adds outer
controller and recorded preparation costs, and checks the total envelope.
The internal `--worker` option is not the execution interface. CPU time includes
retained failures; no failed attempt resets a budget. Final accounting-file
writes are outside the measured boundary.

The actual execution configuration is retained at
`runs/e2-expansion-audit-inputs-2026-11-14/execution-reviewed-v3.json`, hash
`ce75ec8ad71420a11b3a7ace0ad00714bfc2583ac1576d9d45d7d2230ce5d3ca`.
It references the original ignored worker/supervisor snapshots used while the
BASE pilot's Git source was frozen. Their byte-identical versioned copies are
listed in the manifest. Reusing that configuration from a different supervisor
path is rejected. A later attempt needs a newly pinned source/configuration and
a separate bounded decision that retains these historical costs; the existing
output directories cannot be overwritten.

Check the retained records without reopening SGFs or querying a model, from the
repository root with the [locked CPU environment](workspace.md):

```bash
uv run --locked python tools/check_cached_coverage.py \
  runs/e2-expansion-audit-inputs-2026-11-14/execution-reviewed-v3.json \
  runs/e2-cached-coverage-verification.json \
  --sha256 ce75ec8ad71420a11b3a7ace0ad00714bfc2583ac1576d9d45d7d2230ce5d3ca
```

Use an unused output filename. The check verifies the retained prefix, role and
admission counts, source/member pins, quarantine, owner exclusions, masks and
conservative exposure registry. It does not establish legal play or deduplication.

Synthetic tests require no real SGFs, network or models:

```bash
uv run --locked python -m unittest discover -s tests -p 'test_cached_coverage*.py' -v
```

A partial audit is a retained result, not a zero-error statement about all frozen
candidates. Read both `execution_status` and `audit_status`; report unattempted
IDs and do not infer their admission rate. The attempted URL prefix is not a
random sample. Canonical records, whole-game deduplication/splits, legal replay,
examples and a train-only normalizer remain separate work under `kenggo-ydo.8`.
