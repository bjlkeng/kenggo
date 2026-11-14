# BASE pilot interfaces and cost probe

The [E2 plan](../plan/go_project_plan_v0.5.md#e2-build-and-measure-human-baselines)
compares a compact rank-conditioned model, the same unconditioned architecture,
and raw KataGo Human SL. The single-seed pilot has a combined 24 GPU-hour cap.
The separate [E0 throughput allowance](../plan/e0_budgets_and_handoff.md) is
30 GPU minutes across all configurations and failed attempts. Neither allowance
establishes a human rank or qualifies raw play.

`recorded_data.prepare_common_set` accepts independently pinned canonical,
split, example and normalizer inputs plus an explicit role, source list and
aggregate-rank list. It opens only the selected training/debug or development
records. Confirmation and final payloads are unavailable through this interface.
Every selected game is validated through its last action before any observation
is returned. The frozen manifest records ordered row identities, inputs, target
masks and coverage; targets stay separate from model observations. Full legality
history remains in `Game` even though the current model encodes zero prior moves.

`recorded_statistics.evaluate` and `paired_report` use the fixed rank/game-side
hierarchy. Output coverage and legal distributions must be complete. Raw zero
probability produces infinite NLL; missing predictions fail the evaluation.
The weighted dyadic variance groups whole games and repeated unordered account
pairs. It uses globally centered residuals without a degrees-of-freedom
correction. Missing ranks make the frozen aggregate unavailable. Insufficient
account coverage, excessive concentration, and nonpositive or undefined variance
make its interval inconclusive. Reports retain pooled/account diagnostics,
rank/phase/clock coverage and masked recorded-outcome calibration.

`learned_agent.LearnedPolicy` loads pinned checkpoint weights on CPU without
restoring training RNG or allocating CUDA. Its match factory uses raw
temperature-1 sampling, missing clocks and an unknown opponent rank. Model,
normalizer, checkpoint, configuration, provenance and implementation hashes are
part of the scheduled profile. There is no pass correction or resignation.

## Training-only throughput probe

The [frozen probe configuration](../configs/e2_throughput_v1.json) uses all
22,916 training rows for representative batches. It measures both 8-block,
128-channel models at batch sizes 64 and 128. Five repetitions alternate case
order. Each case has two warmup and ten timed updates; exact CUDA checkpoint
continuation adds three actual updates per arm. Total planned work is 246 updates
and 23,424 presented examples, including warmup and continuation controls.

CPU inference uses one fixed 192-move training game at zero-based plies
0, 30, 60, 120 and 180, with a separate warmup at ply 90. The learned models
retain the training examples' frozen rank masks. Human SL uses the recorded
mover rank explicitly and its documented fixed metadata defaults. This measures
cost, with no held-out score or quality comparison. The learned arms make
50 timed and 10 warmup forward calls together. The reference makes 25 timed
analysis requests, five warmup requests and five version requests, restarting
the native process for each repetition to avoid cross-repetition cache reuse.
Native neural-forward counts remain unknown.

The supervisor charges the GPU job from launch through process-tree cleanup,
including imports, data loading, warmup, checkpoint work and idle time. Its
reservation is at most 1,500 seconds plus 10 seconds for shutdown, within the
shared 1,800-second ledger. CPU inference and the Eigen reference run after the
GPU process exits. Their cost records have no GPU allocation. CPU/wall/output
limits and the durable ledger path are in the configuration. A failed attempt
does not reset the ledger. Every probe checkpoint is excluded from pilot
initialization.

From the repository root, with intended new source files staged:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" \
  uv run --locked python -m kenggo.throughput configs/e2_throughput_v1.json \
  runs/e2-throughput-2026-11-14 \
  --config-sha256 356e6f8494affc50d93d6fd7160bdd560fad3530d0b726b8b6e3d6c5e4a53f87
```

Use an unused output directory for an explicitly budgeted repeat. Preserve the
same ledger and all previous charges. Timing a training step does not establish
complete-game throughput; match costs require a separate pilot measurement.

The [probe report](../reports/e2_base_cost_probe_2026-11-14.md) records actual
costs and the resulting [pilot recipe](../configs/e2_base_plan_v1.json).
`python -m kenggo.base_pilot` freezes the common sets, runs both paired training
jobs, evaluates all controls, checks raw play, and computes paired reports.
Each job has its own W&B run and consumes the same SL ledger. Checkpoints are
registered from the owned worker directory without restoring training RNG in
the CPU supervisor. Failed jobs stop the pipeline; unfilled match slots remain
explicitly unresolved and the match run is marked partial.

W&B receives the recorded-position metrics, paired contrasts, sampling
diagnostics, match outcomes and behavior flags. Tagged infinite or unavailable
estimates retain their status. Job CPU, elapsed time and charged GPU allocations
are logged from the shared ledger, including settled failures. Parent overhead
excludes a settled worker even when the supervisor raises after recording it.

The 32-draw sampling report compares the six E1 move categories with recorded
actions using the shared rules classifier. Frequencies and total variation use
equal rank, game-side and position weights, with rank/phase/clock breakdowns and
joint cells. Missing ranks make an aggregate unavailable. These descriptive
pilot diagnostics have no qualification confidence bounds.


## Completed pilot and manual review

The [BASE report](../reports/e2_base_pilot_2026-11-14.md) and
[evidence manifest](../manifests/e2_base_pilot_v1.json) retain the fixed recipe,
checkpoints, development comparison, raw games, failures and costs. The separate
[cached-source audit](cached_coverage.md) is provisional data evidence; it does
not replace the pilot's frozen inputs.

The single-page review export uses saved results and replays original candidate
moves. It needs no server or external assets. Open the generated HTML directly;
use its Games section to inspect endings and pass flags, Compare for rank/phase/
clock scores and value calibration, and Decision for the evidence limits and next
scope choice. Notes stay in local browser storage when available. Export the
notes JSON to communicate feedback; choosing an option does not apply approval.

The versioned tooling is `tools/build_e2_review.py` with
`tools/e2_review/template.html`. Its CLI takes the completed pilot directory,
template, decision JSON and an unused HTML output path. Its adjacent manifest
pins all three inputs and the collector. The page never loads data over HTTP.
The collector refuses incomplete evaluations, changed evidence and unverified
match records, while retaining unresolved and unfilled scheduled slots.

After a completed pilot, `tools/run_e2_postchecks.py` runs a pinned list of
CPU-only saved-artifact checks against the existing SL ledger. The independent
metric checker recomputes rank/game-side means and calibration bins from the
already-opened development shards. The conditioning check inspects training
masks and checkpoint tensors without model inference. The artifact index pins
retained evidence while excluding W&B internal caches. No helper opens final or
confirmation targets, changes checkpoints, or launches GPU work.

```bash
uv run --locked python -m unittest discover -s tests -p 'test_e2_review.py' -v
node tools/e2_review/test-ui.cjs
```

The Node suite tests interactions and data handling; it is not a browser-layout
check. The completed page also needs a rendered inspection before owner review.
`tools/e2_review/check_browser.cjs HTML OUTPUT_DIRECTORY` exercises the actual
export using an existing Playwright/Chromium installation and saves screenshots.
It opens a local file, makes no HTTP requests, and closes its isolated browser.
