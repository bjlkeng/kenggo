from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

from kenggo.checkpoints import file_hash, save_checkpoint, save_json
from kenggo.tracking import Experiment, capture_provenance


CONFIG = {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
          "base_url": "https://api.wandb.ai", "comparison": "test", "arm": "test", "purpose": "test",
          "seed": 3, "parameters": {"width": 2}, "inputs": {}, "budget_id": "fixture"}
PROVENANCE = {"version": "experiment_provenance_v1", "inputs": {},
              "source": {"commit": "fixture", "patch_sha256": sha256(b"").hexdigest()}, "environment": {}}


class Sink:
    def __init__(self, directory, config, provenance, run_id, *, mode, resume):
        self.reference = {"id": run_id, "mode": mode, "url": "https://example.invalid/fixture" if mode == "online" else None}
        self.events = []
    def log(self, step, metrics, attempt):
        self.events.append((step, metrics, attempt))
    def artifact(self, name, files, metadata):
        assert all(Path(path).exists() for path in files.values())
        return {"name": name, "mode": self.reference["mode"], "synced": self.reference["mode"] == "online"}
    def finish(self, status, summary):
        self.outcome = (status, summary)


def components():
    model = torch.nn.Linear(2, 1)
    return model, torch.optim.Adam(model.parameters())


class ExitDuringFinish(Sink):
    def finish(self, *args):
        os._exit(17)


def exit_during_finish(root):
    with Experiment(root, CONFIG, PROVENANCE, sink_factory=Sink) as run:
        ref = run.checkpoint(*components(), step=0, progress={})
        run.finish("partial", reason="Planned boundary")
    with Experiment(root, CONFIG, PROVENANCE, resume=True,
                    sink_factory=ExitDuringFinish) as resumed:
        resumed.restore(ref, *components())


class TrackingTests(unittest.TestCase):
    def test_worker_checkpoint_registers_without_restoring_rng_and_remains_restorable(self):
        with tempfile.TemporaryDirectory() as temporary:
            with Experiment(Path(temporary) / "run", CONFIG, PROVENANCE, sink_factory=Sink) as run:
                model, optimizer = components()
                ref = save_checkpoint(run.directory / "worker/checkpoints", model, optimizer,
                    step=7, progress={"next_step": 7}, config_sha256=run.config_sha256,
                    provenance_sha256=run.provenance_sha256)
                state = torch.get_rng_state().clone()
                with mock.patch("kenggo.tracking.load_checkpoint", side_effect=AssertionError("No restore during upload")):
                    registered = run.register_checkpoint(ref)
                torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
                self.assertEqual(registered["sha256"], ref["sha256"])
                self.assertEqual(run.state["checkpoints"][0]["step"], 7)
                self.assertTrue(run.state["checkpoints"][0]["wandb"])
                self.assertTrue(run.state["checkpoints"][0]["path"].startswith("worker/checkpoints/"))
                with self.assertRaisesRegex(ValueError, "already registered"):
                    run.register_checkpoint(ref)
                other, other_optimizer = components()
                restored = run.restore(ref, other, other_optimizer)
                self.assertEqual(restored, {"step": 7, "progress": {"next_step": 7}})
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)

    def test_worker_checkpoint_rejects_external_torn_or_wrong_run_files_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with Experiment(root / "run", CONFIG, PROVENANCE, sink_factory=Sink) as run:
                model, optimizer = components()
                def saved(directory):
                    return save_checkpoint(directory, model, optimizer, step=0, progress={},
                        config_sha256=run.config_sha256, provenance_sha256=run.provenance_sha256)
                external = saved(root / "external")
                with self.assertRaisesRegex(ValueError, "inside its owning run"):
                    run.register_checkpoint(external)
                ref = saved(run.directory / "worker")
                path = Path(ref["path"])
                original = json.loads(path.read_text())
                for field, value in (("config_sha256", "a" * 64), ("provenance_sha256", "b" * 64),
                                     ("step", True), ("version", "wrong")):
                    changed = dict(original, **{field: value})
                    pin = save_json(path, changed, replace=True)
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        run.register_checkpoint({"path": str(path), "sha256": pin})
                pin = save_json(path, original, replace=True)
                (path.parent / "state.pt").write_bytes(b"partial checkpoint")
                with self.assertRaises(ValueError):
                    run.register_checkpoint({"path": str(path), "sha256": pin})
                self.assertEqual(run.state["checkpoints"], [])
                self.assertEqual(run.state["artifact_bytes_attempted"], 0)

    def test_process_exit_during_finish_leaves_a_recoverable_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            process = multiprocessing.get_context("spawn").Process(target=exit_during_finish, args=(root,))
            process.start()
            process.join(timeout=20)
            if process.is_alive():
                process.kill()
                process.join()
                self.fail("Finish-interruption fixture exceeded its deadline")
            self.assertEqual(process.exitcode, 17)
            state = json.loads((root / "run.json").read_text())
            self.assertEqual(state["status"], "finishing")
            self.assertFalse(state["tracking_finished"])
            ref = state["checkpoints"][0]
            with Experiment(root, CONFIG, PROVENANCE, resume=True, sink_factory=Sink) as resumed:
                resumed.restore({"path": str(root / ref["path"]), "sha256": ref["sha256"]}, *components())
            self.assertEqual(resumed.state["attempts"][1]["status"], "interrupted")
            self.assertEqual(resumed.sink.outcome[1]["costs/incomplete_attempts"], 1)
            self.assertTrue(resumed.state["tracking_finished"])

    def test_partial_checkpoint_resumes_under_same_run_and_preserves_all_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            model, optimizer = components()
            with Experiment(root, CONFIG, PROVENANCE, sink_factory=Sink) as run:
                run.log(1, {"loss": 3.0})
                ref = run.checkpoint(model, optimizer, step=1, progress={"next_batch": 1})
                original_id = run.state["id"]
                run.log(2, {"loss": 2.0})
                run.finish("partial", reason="Test interruption")
            with Experiment(root, CONFIG, PROVENANCE, resume=True, sink_factory=Sink) as resumed:
                self.assertEqual(resumed.state["id"], original_id)
                with self.assertRaisesRegex(ValueError, "Restore"):
                    resumed.log(2, {"loss": 1.0})
                restored = resumed.restore(ref, *components())
                self.assertEqual(restored["step"], 1)
                resumed.log(2, {"loss": 1.0})
            summary = resumed.sink.outcome[1]
            self.assertEqual(summary["attempts/count"], 2)
            self.assertEqual(summary["attempts/1/status"], "partial")
            self.assertEqual(summary["attempts/2/status"], "completed")
            self.assertGreaterEqual(summary["costs/process_cpu_seconds"],
                                    sum(a["cpu_seconds"] for a in resumed.state["attempts"]))
            state = json.loads((root / "run.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual([a["status"] for a in state["attempts"]], ["partial", "completed"])
            rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([(r["attempt"], r["step"]) for r in rows], [(1, 1), (1, 2), (2, 2)])
            self.assertEqual(state["attempts"][1]["restored_checkpoint"], ref["sha256"])
            self.assertGreaterEqual(state["attempts"][0]["cpu_seconds"], 0)

    def test_failures_are_retained_and_existing_directories_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with Experiment(root, CONFIG, PROVENANCE, sink_factory=Sink) as run:
                    run.log(0, {"loss": 4})
                    raise RuntimeError("training failed")
            state = json.loads((root / "run.json").read_text())
            self.assertEqual(state["status"], "failed")
            before = (root / "run.json").read_bytes()
            with self.assertRaises(FileExistsError):
                Experiment(root, CONFIG, PROVENANCE, sink_factory=Sink)
            self.assertEqual((root / "run.json").read_bytes(), before)

    def test_initialization_failure_and_tracking_shutdown_failure_remain_failed(self):
        class InitFailure(Sink):
            def __init__(self, *args, **kwargs):
                raise ConnectionError("Unavailable")
        class FinishFailure(Sink):
            def finish(self, *args):
                raise TimeoutError("Upload deadline")
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "init", Path(temporary) / "finish"
            with self.assertRaises(ConnectionError):
                Experiment(first, CONFIG, PROVENANCE, sink_factory=InitFailure)
            self.assertEqual(json.loads((first / "run.json").read_text())["status"], "failed")
            with self.assertRaises(TimeoutError):
                with Experiment(second, CONFIG, PROVENANCE, sink_factory=FinishFailure):
                    pass
            self.assertEqual(json.loads((second / "run.json").read_text())["status"], "failed")

    def test_offline_is_explicit_and_resume_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            with Experiment(root, CONFIG, PROVENANCE, mode="offline", sink_factory=Sink) as run:
                self.assertIsNone(run.state["wandb"]["url"])
                run.finish("partial", reason="Offline fixture")
            before = (root / "run.json").read_bytes()
            with self.assertRaises(ValueError):
                Experiment(root, CONFIG, PROVENANCE, mode="offline", resume=True, sink_factory=Sink)
            with self.assertRaises(ValueError):
                Experiment(root, CONFIG | {"seed": 4}, PROVENANCE, resume=True, sink_factory=Sink)
            self.assertEqual((root / "run.json").read_bytes(), before)

    def test_writer_lock_and_metric_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            with Experiment(root, CONFIG, PROVENANCE, sink_factory=Sink) as run:
                with self.assertRaises(BlockingIOError):
                    Experiment(root, CONFIG, PROVENANCE, resume=True, sink_factory=Sink)
                run.log(2, {"loss": 1.0})
                for step, values in ((1, {"loss": 1}), (3, {"loss": float("nan")}), (3, {"attempt": 2})):
                    with self.subTest(values=values), self.assertRaises(ValueError):
                        run.log(step, values)
            self.assertEqual(len((root / "metrics.jsonl").read_text().splitlines()), 1)

    def test_summary_cannot_override_the_actual_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            with Experiment(Path(temporary) / "run", CONFIG, PROVENANCE, sink_factory=Sink) as run:
                with self.assertRaisesRegex(ValueError, "reserved"):
                    run.finish("failed", reason="Expected fixture failure", summary={"outcome": "completed"})
                run.finish("failed", reason="Expected fixture failure")

    def test_artifact_limit_is_checked_before_upload(self):
        class NoUpload(Sink):
            def artifact(self, *args):
                raise AssertionError("An over-budget artifact reached the uploader")
        config = CONFIG | {"parameters": {"max_artifact_bytes": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            with self.assertRaisesRegex(ValueError, "byte allowance"):
                with Experiment(root, config, PROVENANCE, sink_factory=NoUpload) as run:
                    run.checkpoint(*components(), step=0, progress={})
            state = json.loads((root / "run.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["artifact_bytes_attempted"], 0)
            self.assertIsNone(state["checkpoints"][0]["wandb"])

    def test_provenance_checks_inputs_and_captures_the_complete_source_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
            git("init", "-q")
            for name in ("data", "split", "rules", "source.py"):
                (root / name).write_text("original\n")
            git("add", ".")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
            refs = {name: {"path": name, "sha256": file_hash(root / name)} for name in ("data", "split", "rules")}
            (root / "source.py").write_text("updated\n")
            provenance, patch = capture_provenance(root, refs)
            self.assertEqual(provenance["source"]["commit"], git("rev-parse", "HEAD").decode().strip())
            self.assertIn(b"+updated", patch)
            self.assertEqual(provenance["source"]["patch_sha256"], sha256(patch).hexdigest())
            (root / "new.py").write_text("untracked")
            with self.assertRaisesRegex(ValueError, "Stage"):
                capture_provenance(root, refs)
            git("add", "new.py")
            self.assertIn(b"+untracked", capture_provenance(root, refs)[1])
            (root / "data").write_text("changed")
            with self.assertRaisesRegex(ValueError, "input differs"):
                capture_provenance(root, refs)

    def test_interrupted_attempt_keeps_unknown_costs_explicit_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run = Experiment(root, CONFIG, PROVENANCE, sink_factory=Sink)
            ref = run.checkpoint(*components(), step=0, progress={})
            run.lock.close()
            run.closed = True
            with Experiment(root, CONFIG, PROVENANCE, resume=True, sink_factory=Sink) as resumed:
                resumed.restore(ref, *components())
            self.assertEqual(resumed.state["attempts"][0]["status"], "interrupted")
            self.assertNotIn("cpu_seconds", resumed.state["attempts"][0])
            self.assertEqual(resumed.sink.outcome[1]["costs/incomplete_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
