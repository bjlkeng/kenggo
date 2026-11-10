# Python research workspace

This workspace targets **Linux x86_64 with glibc 2.28 or newer, CPython 3.12.13 and CPU execution**. It installs the `kenggo` package with PyTorch, NumPy, PyArrow and SGFmill. The [shared rules adapter](rules_engine.md) provides validated ordinary play and conservative adjudication, with ending coverage still blocked. No learned model exists. The [project plan](../plan/go_project_plan_v0.5.md#technology-stack) owns the stack; the [source-audit workflow](e1_source_audit.md) describes the bounded data inspection tools.

## Clean setup

Install **uv 0.11.16** using the [official versioned installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Run the following from the repository root. Initial setup needs network access to download Python and the locked wheels; it needs no dataset, GPU or W&B credentials.

```bash
uv --version
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv python install --no-bin 3.12.13
uv sync --locked
uv run --locked python -m kenggo.smoke
uv run --locked python -m unittest discover -s tests -v
```

The exports keep downloads and the interpreter inside the ignored workspace cache; repeat them in a new shell. `uv sync` creates `.venv/` and installs the package. The smoke command must exit zero and print JSON with `"status": "ok"` and `"device": "cpu"`. It checks fixed tensor forward/backward values, NumPy interoperability and an on-disk Parquet round trip preserving nulls and schema. It removes its temporary file. This is an installation check, not evidence about Go behavior or training performance.

[`.python-version`](../.python-version) selects the exact interpreter patch. [`pyproject.toml`](../pyproject.toml) pins direct dependencies, uv and the build backend; [`uv.lock`](../uv.lock) records resolved dependencies, wheel sources and hashes. Use `--locked` to reject an outdated lock instead of changing dependencies during a run. The CPU PyTorch index is explicit so ordinary CPU setup does not fetch CUDA libraries. See uv's [lock semantics](https://docs.astral.sh/uv/concepts/projects/sync/) and [PyTorch index guidance](https://docs.astral.sh/uv/guides/integration/pytorch/).

To change dependencies, edit the intended pins, run `uv lock`, then repeat the clean setup, smoke and build checks below. Review the lock diff alongside the change. Do not install ad hoc packages into an experiment environment and call it the locked setup. Other operating systems, Python minors and CUDA builds need a separately validated environment revision.

## Checks and packaging

With the cache exports above, run:

```bash
uv lock --check
uv sync --locked --check
uv run --locked python -m kenggo.smoke
uv run --locked python -m unittest discover -s tests -v
uv build --no-sources
git diff --check
```

The build creates a source distribution and wheel in ignored `dist/`. Building the wheel does not check its runtime dependencies; run the smoke check after installation too. Behavior tests cover acquisition/auditing and the rules adapter. Run the separate [pinned native reference checks](rules_engine.md#reproduce-validation) for rules changes; a skipped reference test is not a passing comparison. Splitting, search and RL implementations/tests are still future work. The installation smoke does not validate Go behavior.

## GPU readiness is separate

Run the host check independently of the CPU smoke:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

On **9 November 2026**, this workstation's PCI inventory identified a GeForce RTX 3090, but an unsandboxed `nvidia-smi` exited 18 with `Driver/library version mismatch`: NVML **595.99**, loaded kernel module **595.84**. GPU readiness was therefore not established. This is a host-specific observation, not a claim about another machine or a later repaired driver. Bead `kenggo-ydo.7` owns the host/CUDA follow-up and blocks the E2 GPU pilot.

The locked environment uses a CPU-only PyTorch wheel; `torch_cuda_build: null` is expected and says nothing about physical GPU availability. Before GPU experiments, repair/verify the host driver, select and lock a compatible CUDA build, then execute and synchronize a small CUDA tensor operation and record versions/device details. A successful `nvidia-smi` alone does not validate PyTorch CUDA execution. No driver installation, reboot, CUDA workload or throughput probe is part of this bootstrap. The [E0 cost rules](../plan/e0_budgets_and_handoff.md) still apply to later probes and pilots.

## Artifact locations

Create directories as needed. These paths are relative to the repository root and already ignored; do not force-add their contents.

| Location | Contents |
| --- | --- |
| `data/raw/` | Original SGFs and source downloads, with provenance retained alongside them |
| `data/processed/` | Canonical Parquet tables, split outputs and training shards |
| `checkpoints/` | Model weights, optimizer state and resume state |
| `runs/<run-id>/` | Actual run metrics, cost ledgers, logs, generated match SGFs and local W&B output |
| `.cache/`, `.venv/`, `dist/` | Local interpreter/download cache, environment and package builds |

Keep small reusable fixtures in `tests/fixtures/` when tests are added. Keep reviewable configurations in `configs/`, small reproducibility manifests in `manifests/`, and experiment decision reports in `reports/`; these paths are not ignored. A manifest should identify versions, hashes and artifact locations without embedding the raw corpus or weights. These directories are conventions, not empty scaffolding or a second task tracker. Configure W&B authentication and its destination in the existing experiment-tracking bead before actual runs; this check makes no W&B calls.
