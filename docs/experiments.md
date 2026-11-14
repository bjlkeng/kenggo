# Track an experiment and restore its checkpoint

Use the locked [CPU workspace](workspace.md) for setup and the separate
[CUDA environment](gpu_workspace.md) for verified GPU work. W&B SDK **0.31.0**
is pinned in both environments. The project is **`bjlkeng/kenggo`**, with private
access. Authentication uses the existing W&B login; credentials are never copied
into configuration, provenance, checkpoints or reports.

## Check the complete tracking path

From the repository root, with the pinned E1 artifacts available:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
timeout 180s uv run --locked python -m kenggo.tracking_smoke \
  configs/e2_tracking_smoke_v1.json runs/e2-tracking-smoke-repeat \
  --config-sha256 276c2fcc9898e733fd6a6a875e788bd25be2f88d11492ff568484b3e0274e970
```

Choose a new output directory. Stage intended new source files before a run so
the provenance snapshot includes them. The command verifies input pins, records
the Git revision and complete binary patch, and saves package versions. It makes
an actual online W&B run on synthetic CPU tensors, intentionally ends a partial
attempt, resumes the same run, and compares the next stochastic optimizer update
exactly with uninterrupted execution. It uploads a checkpoint, downloads it from
W&B, and verifies its hash, state and progress. E1 inputs are referenced by hash;
no Go positions, sealed targets or source SGFs are loaded or uploaded.

The frozen allowance is 120 calling-process CPU seconds, 180 elapsed seconds,
10 MiB of attempted checkpoint uploads and zero GPU allocation. The outer
timeout also bounds stalled SDK calls. A killed process can leave a running
attempt or outstanding resource reservation; retain it as incomplete evidence
and recover its actual costs before proceeding. Dependency setup is accounted
separately. This check establishes tracking and continuation, not a Go learner,
throughput result, baseline comparison or GPU readiness.

## Run records and checkpoints

`capture_provenance(repository, inputs)` verifies independently supplied input
hashes, requires `data`, `split` and `rules` references, and captures the Git
revision, binary patch and installed environment. `Experiment` then owns one
new run directory. Existing directories are refused unless `resume=True`.
The configured seed is recorded; callers use `seed_all(seed)` before creating
model/optimizer state and initialize any additional RNGs they own.

```python
from kenggo.tracking import Experiment, capture_provenance
from kenggo.checkpoints import seed_all

provenance, patch = capture_provenance(repository, config["inputs"])
seed_all(config["seed"])
with Experiment(output, config, provenance, patch) as run:
    run.log(step, {"train/loss": loss})
    checkpoint = run.checkpoint(model, optimizer, step=step,
                                progress={"epoch": epoch, "next_batch": next_batch},
                                scheduler=scheduler)
    run.finish("partial", reason="Planned checkpoint boundary")
```

Local files include immutable `config.json`, `provenance.json`, `source.patch`,
append-only metrics, a status/attempt record and immutable checkpoint directories.
Each checkpoint manifest binds its tensor payload to the config and provenance.
Checkpoints save model/optimizer state, optional scheduler/scaler state, each
module's train/eval mode, Python/NumPy/Torch RNGs, initialized CUDA RNGs and deterministic backend
settings. Store loader cursor, sampler state and any additional application-owned
RNG state in `progress`; the library cannot infer those from a model.

Resume with the same config, source/environment provenance and `mode="online"`.
W&B must recognize the existing run. Call `run.restore(checkpoint, model,
optimizer, scheduler=...)` before logging or saving resumed work. It verifies the
independent manifest pin, payload hash, component types and model structure,
then restores state with Torch's weights-only loader. The checkpoint's config
and provenance must match the run. Interrupted attempts and their metrics remain
visible; each metric carries its attempt number. Restoring an older checkpoint
may repeat steps, so report those extra costs and select the intended attempt
when analyzing metrics. A completed run cannot be resumed as if it were unfinished.
W&B summary fields retain each attempt separately and aggregate recorded costs.
An interrupted process can leave unknown costs; the summary counts those
incomplete attempts rather than treating the displayed aggregate as a full total.

`completed`, `partial`, `failed` and `cancelled` are distinct local outcomes.
Incomplete outcomes require reasons; uncaught exceptions and tracking failures
retain failed records. A run stays recoverable as `finishing` until tracking
shutdown succeeds. Process death during shutdown becomes an interrupted attempt
on resume, with its missing shutdown costs marked incomplete. Offline mode is explicitly unsynced and cannot satisfy the
online tracking check or resume an online run. CPU/time and process peak-memory
observations are local to the calling process; tracking shutdown time is recorded
separately. Background-service CPU is not claimed as measured total CPU.

Keep runs, checkpoints and downloaded artifacts under ignored `runs/` or
`checkpoints/`. Version the small configuration and final evidence manifest.

## Account for allocated GPU time

`BudgetLedger` implements the [E0 accounting contract](../plan/e0_budgets_and_handoff.md).
Create one ledger per declared budget and keep that identity/cap fixed. Reserve
an upper estimate plus shutdown allowance before launching. Record actual UTC
allocation intervals by device, including idle time while reserved, and settle
failed/partial runs too. Allocation measurement and stopping jobs remain the
caller's responsibility.

The ledger serializes cooperating writers with a directory lock and atomic file
replacement. It rejects overlapping device intervals, duplicate run IDs, missing
reservations and nonfinite costs. Shared arm fractions must sum to one; one
allocation is charged once. Outstanding reservations survive process failure.
Actual reservation overruns or cap breaches are retained and stop new launches;
existing reservations can still settle. There is no automatic reset or budget
increase. Record the next budget decision in Beads before further work.

Use empty GPU intervals for CPU-only runs. Keep the separate throughput probe,
SL pilot and RL pilot in separate ledgers. Never clear failures or create a new
ledger to hide an existing charge.
