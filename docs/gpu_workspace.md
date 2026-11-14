# CUDA workspace and driver recovery

The CUDA environment is prepared separately from the existing CPU environment.
The 13 November host and synchronized CUDA check passed after the owner
restarted Ubuntu. The [readiness evidence](../manifests/gpu_workspace_2026-11-13.json)
records the matching 595.99.02 loaded/on-disk driver, RTX 3090, Python 3.12.13
and Torch 2.15.0+cu132. This verifies installation readiness on this workstation,
not model throughput or quality.

The [preparation manifest](../manifests/gpu_workspace_2026-11-11.json) records the
11 November environment and failed host check. All 170 tests passed in both the
CPU and CUDA Python environments, with the native rules reference enabled and
zero skips. Those tests and the CPU smoke do not establish GPU execution: the
CUDA-specific check stopped at the host mismatch before allocating GPU work.

The 13 November [experiment-tracking update](experiments.md) adds W&B 0.31.0
and its dependencies to both Python environments. All 32 original CUDA package
versions, including Torch, are preserved. The historical preparation manifest's
dependency hash describes commit `5151045`; the current lock includes tracking.
The dependency update alone did not establish GPU execution; the subsequent
readiness check did. All 211 tests also pass in both environments with the pinned
native rules reference enabled and zero skips. The test suites run on CPU; the
separate CUDA check verifies tensor values, backward gradients and synchronization.

The inner CUDA allocation lasted 3.5339 seconds. The readiness budget
conservatively charges the entire 21.7279-second wrapper interval, including
host discovery, imports and shutdown, against its existing 60-second allowance.
The wrapper measured 4.0787 child-process CPU seconds. Historical failures remain
in the preparation evidence, with zero GPU allocation. No new package download,
Go data access or training was part of the successful check.

## Host diagnosis on 11 November 2026

The workstation runs Ubuntu 22.04.5 and kernel `6.8.0-138-generic`. The loaded
NVIDIA open kernel module is **595.84**. The installed module for that same kernel
and the NVIDIA userspace packages are both **595.99.02**. The host `nvidia-smi`
check exits 18 with a driver/library mismatch. Ubuntu's restart-required flag is
present, and the display manager is active.

The expected repair is to save work and restart Ubuntu so it loads the installed
module. [Ubuntu documents this exact mismatch and recovery](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/#driver-library-version-mismatch-error).
This agent session cannot perform administrator operations without a password.
No driver package, display service or system configuration was changed.

After restarting, run:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cat /proc/driver/nvidia/version
modinfo -F version nvidia
```

The host query must succeed and the loaded/on-disk NVIDIA versions must agree.
If the mismatch persists, retain the new outputs for further diagnosis before
attempting package changes. A successful host query still needs the CUDA check.

## Install the pinned Python environment

Use **uv 0.11.16**, **Python 3.12.13**, Linux x86_64 and the repository root.
The [CUDA dependency lock](../environments/cuda/requirements.lock) selects
**Torch 2.15.0+cu132** and hashes every dependency artifact. The existing
`pyproject.toml`, `uv.lock` and default `.venv` remain the CPU setup.
NVIDIA lists R595 as the CUDA 13.2 driver family in its [compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).
Actual execution is the readiness check; version correspondence alone is not.

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv venv --python 3.12.13 .cache/venvs/cuda
UV_CONCURRENT_DOWNLOADS=2 UV_HTTP_RETRIES=0 timeout 900s uv pip sync \
  --python .cache/venvs/cuda/bin/python --no-sources --require-hashes \
  --index-strategy unsafe-best-match \
  environments/cuda/requirements.lock
uv pip install --python .cache/venvs/cuda/bin/python --no-deps --no-sources -e .
uv pip check --python .cache/venvs/cuda/bin/python
```

The first install step supplies the locked third-party dependencies; the second
installs this checkout as the application without resolving dependencies again.
Record the Git commit alongside the dependency-lock hash. Repeating `pip sync`
removes the editable application, so repeat the following `pip install` step too.
Run this environment's Python directly; ordinary `uv run` selects the CPU setup.

To regenerate the CUDA lock after a deliberate dependency revision:

```bash
uv pip compile pyproject.toml --no-sources \
  --constraint environments/cuda/requirements.lock \
  --index https://download.pytorch.org/whl/cu132 \
  --index-strategy unsafe-best-match \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --emit-index-url \
  --output-file environments/cuda/requirements.lock
```

The existing lock constrains all versions, including the CUDA 13.2 Torch build.
The index strategy allows the CUDA index and PyPI to supply the exact pinned
versions; installation still requires the recorded artifact hashes. This avoids
selecting older HTTP/tracking dependencies merely because the CUDA index lists
them first. Change pins deliberately when revising the environment. The pinned
uv version's `--torch-backend` shortcut does not recognize `cu132`.
`--no-sources` prevents the CPU source setting from overriding the CUDA lock.

## Check actual CUDA execution

```bash
timeout 60s .cache/venvs/cuda/bin/python -m kenggo.cuda_smoke
```

The command requires successful host-driver discovery, CUDA availability, a GPU
matrix operation, backward gradients and explicit synchronization. It exits zero
only with JSON `"status": "ok"`; it never falls back to CPU. It records the host
query, Python/Torch/CUDA/device versions, check costs and failure reason. The
60-second deadline bounds this installation check. GPU allocation time includes
initialization and checks after the first CUDA probe; it is not a kernel-speed
measurement. No dataset or model training is involved.

Restricted sandboxes may hide GPU device nodes. Run the CUDA check on the host or
with approved GPU access; a sandbox failure cannot establish that the host works.
Retain all failed checks. A zero exit from the CPU smoke or package installation
does not close the GPU readiness gate.
