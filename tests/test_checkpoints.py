import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from kenggo.checkpoints import (file_hash, json_hash, load_checkpoint, save_checkpoint,
                                save_json, seed_all)


CONFIG = json_hash({"model": "synthetic-linear", "optimizer": "Adam"})
PROVENANCE = json_hash({"source": "fixture"})


def components():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    return model, optimizer, scheduler


def training_step(model, optimizer, scheduler):
    inputs = torch.randn(5, 3) + float(np.random.random()) + random.random()
    target = inputs.sum(dim=1, keepdim=True)
    optimizer.zero_grad()
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


class CheckpointTests(unittest.TestCase):
    def test_restore_preserves_mixed_module_modes_and_next_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed_all(73)
            model, optimizer, scheduler = components()
            model[1].eval()
            ref = save_checkpoint(temporary, model, optimizer, step=0, progress={},
                config_sha256=CONFIG, provenance_sha256=PROVENANCE, scheduler=scheduler)
            expected_loss = training_step(model, optimizer, scheduler)
            expected = copy.deepcopy(model.state_dict())
            restored, restored_optimizer, restored_scheduler = components()
            load_checkpoint(ref["path"], ref["sha256"], restored, restored_optimizer,
                config_sha256=CONFIG, provenance_sha256=PROVENANCE, scheduler=restored_scheduler)
            self.assertEqual([m.training for m in restored.modules()], [True, True, False, True])
            self.assertEqual(training_step(restored, restored_optimizer, restored_scheduler), expected_loss)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_restore_reproduces_next_stochastic_update_and_optimizer_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed_all(19)
            model, optimizer, scheduler = components()
            training_step(model, optimizer, scheduler)
            ref = save_checkpoint(temporary, model, optimizer, step=1, progress={"epoch": 0, "next_batch": 1},
                config_sha256=CONFIG, provenance_sha256=PROVENANCE, scheduler=scheduler)
            expected_loss = training_step(model, optimizer, scheduler)
            expected = copy.deepcopy(model.state_dict())
            expected_lr = scheduler.get_last_lr()
            seed_all(999)
            restored, restored_optimizer, restored_scheduler = components()
            progress = load_checkpoint(ref["path"], ref["sha256"], restored, restored_optimizer,
                config_sha256=CONFIG, provenance_sha256=PROVENANCE, scheduler=restored_scheduler)
            self.assertEqual(progress, {"step": 1, "progress": {"epoch": 0, "next_batch": 1}})
            self.assertEqual(training_step(restored, restored_optimizer, restored_scheduler), expected_loss)
            self.assertEqual(restored_scheduler.get_last_lr(), expected_lr)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_pins_and_resume_components_are_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, optimizer, scheduler = components()
            ref = save_checkpoint(temporary, model, optimizer, step=0, progress={},
                config_sha256=CONFIG, provenance_sha256=PROVENANCE, scheduler=scheduler)
            for config, pin, component in ((json_hash("other"), ref["sha256"], scheduler),
                                         (CONFIG, "0" * 64, scheduler), (CONFIG, ref["sha256"], None)):
                with self.subTest(config=config, pin=pin), self.assertRaises(ValueError):
                    load_checkpoint(ref["path"], pin, model, optimizer, config_sha256=config,
                                    provenance_sha256=PROVENANCE, scheduler=component)
            blob = Path(ref["path"]).parent / "state.pt"
            blob.write_bytes(blob.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "state differs"):
                load_checkpoint(ref["path"], ref["sha256"], model, optimizer, config_sha256=CONFIG,
                                provenance_sha256=PROVENANCE, scheduler=scheduler)

    def test_checkpoints_are_immutable_and_nonfinite_progress_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, optimizer, _ = components()
            args = (temporary, model, optimizer)
            options = dict(step=1, progress={}, config_sha256=CONFIG, provenance_sha256=PROVENANCE)
            first, second = save_checkpoint(*args, **options), save_checkpoint(*args, **options)
            self.assertNotEqual(first["path"], second["path"])
            self.assertEqual(file_hash(first["path"]), first["sha256"])
            with self.assertRaises(ValueError):
                save_checkpoint(*args, **(options | {"progress": {"loss": float("nan")}}))
            before = Path(first["path"]).read_bytes()
            with self.assertRaises(FileExistsError):
                save_json(first["path"], {"overwrite": True})
            self.assertEqual(Path(first["path"]).read_bytes(), before)

    def test_resume_rejects_a_different_optimizer_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, optimizer, _ = components()
            ref = save_checkpoint(temporary, model, optimizer, step=0, progress={},
                config_sha256=CONFIG, provenance_sha256=PROVENANCE)
            wrong = torch.optim.SGD(model.parameters(), lr=0.01)
            with self.assertRaisesRegex(ValueError, "component"):
                load_checkpoint(ref["path"], ref["sha256"], model, wrong,
                                config_sha256=CONFIG, provenance_sha256=PROVENANCE)


if __name__ == "__main__":
    unittest.main()
