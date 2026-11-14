"""Train-only shard access and deterministic supervised population updates."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kenggo.checkpoints import read_json
from kenggo.encoding import INPUT_KEYS, apply_normalizer
from kenggo.examples import LABEL_SCHEMA, read_shards
from kenggo.go import PASS


@dataclass(frozen=True)
class TrainingData:
    inputs: dict
    labels: dict
    row_keys: tuple
    examples_sha256: str
    normalizer_sha256: str
    split_sha256: str

    @classmethod
    def load(cls, examples_dir, examples_sha256, normalizer_path, normalizer_sha256, *, split_sha256,
             max_rows=None):
        """Load a pinned train prefix; no evaluation or final role is accepted."""
        if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
            raise ValueError("The training row limit must be a positive integer")
        manifest = read_json(Path(examples_dir) / "manifest.json", examples_sha256)
        normalizer = read_json(normalizer_path, normalizer_sha256)
        if (manifest["split_sha256"] != split_sha256 or manifest["config"]["history"] != 0
                or normalizer["examples_sha256"] != examples_sha256
                or normalizer["fixture_only"] != manifest["fixture_only"]):
            raise ValueError("Training examples and normalizer must bind the requested split and zero history")
        parts = {key: [] for key in (*INPUT_KEYS, *LABEL_SCHEMA)}
        row_keys = []
        for arrays, metadata in read_shards(examples_dir, examples_sha256, "train"):
            take = len(metadata) if max_rows is None else min(len(metadata), max_rows - len(row_keys))
            for key in parts:
                parts[key].append(arrays[key][:take])
            row_keys.extend(row["row_key"] for row in metadata[:take])
            if max_rows is not None and len(row_keys) == max_rows:
                break
        if not row_keys:
            raise ValueError("Training requires at least one training example")
        arrays = {key: np.concatenate(items) for key, items in parts.items()}
        inputs = apply_normalizer({key: arrays[key] for key in INPUT_KEYS}, normalizer)
        labels = {key: arrays[key] for key in LABEL_SCHEMA}
        for array in (*inputs.values(), *labels.values()):
            array.flags.writeable = False
        return cls(inputs, labels, tuple(row_keys), examples_sha256, normalizer_sha256, split_sha256)

    def __len__(self):
        return len(self.row_keys)

    def indices(self, step, batch_size, seed):
        if (type(step) is not int or step < 0 or type(batch_size) is not int or batch_size < 1
                or type(seed) is not int or not 0 <= seed < 2**32 or len(self) == 0):
            raise ValueError("Batches require a nonnegative step, positive size, seed and nonempty data")
        batches_per_epoch = (len(self) + batch_size - 1) // batch_size
        epoch, batch = divmod(step, batches_per_epoch)
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        return generator.permutation(len(self))[batch * batch_size:(batch + 1) * batch_size]

    def batch(self, step, batch_size, seed, device="cpu"):
        indices = self.indices(step, batch_size, seed)
        return tuple({key: torch.from_numpy(array[indices]).to(device) for key, array in group.items()}
                     for group in (self.inputs, self.labels))


def hide_clocks(inputs, probability):
    """Hide the whole nominal/dynamic clock family without changing the batch."""
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Clock dropout must be a finite probability")
    if probability == 0:
        return inputs
    scalars = inputs["scalars"]
    hidden = torch.rand(scalars.shape[0], device=scalars.device) < probability
    result = dict(inputs)
    result["scalars"] = scalars.clone()
    result["scalar_valid"] = inputs["scalar_valid"].clone()
    result["scalars"][hidden, 9:] = 0
    result["scalar_valid"][hidden, 9:] = False
    return result


def supervised_loss(logits, value, labels, legal):
    """Legal human-action CE plus mean MSE over supported mover outcomes."""
    if (logits.ndim != 2 or logits.shape[1] != PASS + 1 or value.shape != logits.shape[:1]
            or legal.shape != logits.shape or legal.dtype != torch.bool or not legal.any(dim=1).all()
            or set(labels) != set(LABEL_SCHEMA)):
        raise ValueError("Loss inputs must use the batched policy/value contract")
    if any(label.shape != value.shape for label in labels.values()):
        raise ValueError("Labels must have one entry per position")
    policy_valid, value_valid = labels["policy_valid"], labels["value_valid"]
    if (policy_valid.dtype != torch.bool or value_valid.dtype != torch.bool
            or labels["action"].dtype not in (torch.int16, torch.int32, torch.int64)
            or not labels["value"].is_floating_point()):
        raise ValueError("Invalid action, outcome or mask dtype")
    targets = labels["action"][policy_valid].long()
    if (targets < 0).any() or (targets > PASS).any():
        raise ValueError("Observed policy actions must be board points or pass")
    selected_legal = legal[policy_valid]
    if not selected_legal.gather(1, targets[:, None]).all():
        raise ValueError("An observed human action is illegal")
    selected_logits = logits[policy_valid].masked_fill(~selected_legal, -torch.inf)
    outcomes = labels["value"][value_valid]
    if (not torch.isfinite(selected_logits[selected_legal]).all()
            or not torch.isfinite(value[value_valid]).all()
            or not ((outcomes == -1) | (outcomes == 1)).all()):
        raise ValueError("Valid losses require finite predictions and signed recorded outcomes")
    policy_loss = (F.cross_entropy(selected_logits, targets) if targets.numel()
                   else selected_logits[selected_legal].sum())
    value_loss = F.mse_loss(value[value_valid], outcomes) if outcomes.numel() else value[value_valid].sum()
    return {"loss": policy_loss + value_loss, "policy_loss": policy_loss, "value_loss": value_loss}


def train_step(model, optimizer, inputs, labels, *, clock_dropout=0.1):
    inputs = hide_clocks(inputs, clock_dropout)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, value = model(inputs)
    losses = supervised_loss(logits, value, labels, inputs["legal"])
    losses["loss"].backward()
    optimizer.step()
    return {**{name: float(loss.detach()) for name, loss in losses.items()},
            "examples": int(value.shape[0]), "policy_count": int(labels["policy_valid"].sum()),
            "value_count": int(labels["value_valid"].sum()),
            "clock_missing_count": int((~inputs["scalar_valid"][:, 9:].any(dim=1)).sum())}


def fit(model, optimizer, data, *, start_step, steps, batch_size, seed, clock_dropout=0.1, on_step=None):
    """Return the next-step cursor; callback receives it after each completed update."""
    if type(start_step) is not int or start_step < 0 or type(steps) is not int or steps < 0:
        raise ValueError("Fit requires a nonnegative starting step and update count")
    data.indices(start_step, batch_size, seed)
    device = next(model.parameters()).device
    for step in range(start_step, start_step + steps):
        inputs, labels = data.batch(step, batch_size, seed, device=device)
        metrics = train_step(model, optimizer, inputs, labels, clock_dropout=clock_dropout)
        if on_step is not None:
            on_step(step + 1, metrics)
    return start_step + steps
