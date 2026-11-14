# Compact population learner

`PopulationModel` implements the E2 starting candidate: eight residual blocks
with 128 channels, a 362-action policy including pass, and a tanh value from the
mover's perspective. The smaller configuration used by the integration check is
for debugging. Neither architecture defines a playing strength or qualified
rank profile.

The model accepts only the shared, zero-history encoding. Its inputs are the
mover-relative stones and ko plane, full legal-action mask, normalized scalar
metadata with validity flags, and rank embeddings. The unknown-rank embedding
is learned. The unconditioned variant has the same parameter shapes but hides
both rank values and rank-validity flags. Invalid scalar and rank payloads do
not enter the network. The shared rules engine retains full legality history;
the model's zero-history input does not shorten that history.

The loss is legal-action cross-entropy plus mean squared error over valid
recorded outcomes. A masked outcome contributes neither a target nor a gradient.
The value predicts recorded human-game results, not an adjudicated score or
optimal play. There is no search-generated target, learned resignation,
personalization or duration head.

## Train from the pinned examples

`TrainingData.load` accepts only the training role. It checks the example,
normalizer and split pins, applies the fitted training-only normalizer and keeps
metadata out of model inputs. A row limit takes a fixed prefix, rather than
choosing positions by target or score. Development, confirmation and final
targets are not accessed by this loader.

```python
from kenggo.checkpoints import seed_all
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.supervised import TrainingData, fit
import torch

seed_all(seed)
data = TrainingData.load(examples_dir, examples_sha256,
                         normalizer_path, normalizer_sha256,
                         split_sha256=split_sha256)
model = PopulationModel(ModelConfig(rank_conditioned=True))
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
next_step = fit(model, optimizer, data, start_step=0, steps=100,
                batch_size=32, seed=seed, clock_dropout=0.1)
```

This API example supplies no experiment budget. Actual experiments must first
freeze their configuration, units and resource limits under the
[E0 budget contract](../plan/e0_budgets_and_handoff.md), and record their runs with
[experiment tracking](experiments.md). GPU work uses the verified
[CUDA environment](gpu_workspace.md) and charges its full allocation interval.

Each epoch is a seeded permutation of the training rows. The next batch depends
only on the explicit seed, batch size and update cursor; it does not depend on
worker timing or hidden loader RNGs. Save `next_step` in checkpoint progress
alongside the model/optimizer and global RNGs. Resume using the same pinned data
and configuration. Clock dropout uses the checkpointed Torch RNG and hides the
whole nominal/dynamic clock family, including validity flags. Existing frozen
mover/opponent rank masks remain unchanged, without a second hiding pass.
Dynamic clock inputs and duration labels remain unavailable under E1.

## Bounded implementation check

From the repository root, with the pinned E1 v3 artifacts present:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
timeout 300s uv run --locked python -m kenggo.learner_smoke \
  configs/e2_learner_smoke_v1.json runs/e2-learner-smoke-repeat \
  --config-sha256 56160d7f0879808bad67a603324d27b05493757d4ddde844ea522de8fd730545
```

Use a new output directory and stage intended new source files before execution
so provenance includes them. The check uses only the first 64 training rows,
16 distinct updates, batch size eight and a one-block/16-channel model on CPU.
One additional control update verifies exact continuation after restoring step
eight. All 17 executed updates and their costs are retained. It uploads a
checkpoint to the private W&B project, downloads it and verifies the saved
state and next-step cursor.

The allowance is 180 calling-process CPU seconds, 300 elapsed seconds, 10 MiB of
attempted checkpoint uploads and zero GPU allocation. Failed outputs remain on
disk. Checkpoints and run output are ignored; configuration and evidence
manifests are versioned. The check is not a held-out BASE comparison, throughput
measurement or qualification result. Small synthetic tests separately verify
that fitting learns distinct actions and opposite mover outcomes, that masks
exclude invalid labels, and that checkpoint continuation crosses an epoch
boundary exactly.
