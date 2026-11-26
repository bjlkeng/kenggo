"""Policy trial ordering, selection and shared-budget behavior without training."""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kenggo import policy_study as policy
from kenggo.checkpoints import file_hash, json_bytes, save_json
from kenggo.continuation_data import reference
from kenggo.progressive_training import StopStudy


class FakeStudy:
    def __init__(self, root, *, rounds=8, budget_cycles=None, fail=None):
        self.output = Path(root)
        self.plan = {"inputs": {"fitting_summary": {}}, "parameters": {
            "base_inputs": {}, "model": {}, "training_rows": 3999997}}
        self.parent = {"path": "parent", "sha256": "a" * 64, "config": {}, "provenance": {}, "step": 125000}
        self.state = {"parent_checkpoint": self.parent, "trials": [
            {"id": item["id"], "parameters": {k: v for k, v in item.items() if k != "id"},
             "training": [], "status": "pending"} for item in policy.TRIALS]}
        self.calls, self.cycles = [], 0
        self.rounds, self.budget_cycles, self.fail = rounds, budget_cycles, fail

    def persist(self):
        pass

    def config(self, **kwargs):
        return kwargs

    def require(self, *args, **kwargs):
        self.cycles += 1
        if self.budget_cycles is not None and self.cycles > self.budget_cycles:
            raise StopStudy("Fixture cycle allowance spent")

    def stage(self, name, config, **kwargs):
        def saved(filename, value):
            filename.write_bytes(json_bytes(value))
        self.calls.append((name, deepcopy(config)))
        if name == self.fail:
            raise RuntimeError("Fixture interval failed")
        epoch = config["parameters"]["target_round"]
        identity = config["arm"]
        directory = self.output / name
        directory.mkdir()
        config_path, provenance_path = directory / "config.json", directory / "provenance.json"
        saved(config_path, config)
        saved(provenance_path, {"inputs": config["inputs"]})
        checkpoint_path = directory / "manifest.json"
        payload = directory / "state.pt"
        payload.write_bytes(b"synthetic checkpoint payload")
        step = 125000 + epoch * policy.ROUND_UPDATES
        saved(checkpoint_path, {"version": "training_checkpoint_v2", "file": "state.pt",
            "sha256": file_hash(payload), "step": step, "config_sha256": file_hash(config_path),
            "provenance_sha256": file_hash(provenance_path)})
        checkpoint = {**reference(checkpoint_path), "config": reference(config_path),
            "provenance": reference(provenance_path), "step": step, "kind": "epoch"}
        metrics = directory / "metrics.json"
        score = 2.5 - epoch / 100
        saved(metrics, {"round": epoch, "sampler_step": epoch * policy.ROUND_UPDATES,
            "lifetime_updates": step, "validation": {"nll": score,
                "diagnostic_sha256": config["inputs"]["epoch_validation"].get("sha256")}})
        commit = directory / "commit.json"
        saved(commit, {"version": "continuation_epoch_commit_v1", "round": epoch,
            "checkpoint": checkpoint, "best_checkpoint": checkpoint, "metrics": reference(metrics)})
        summary = {"status": "completed", "latest_checkpoint": checkpoint,
            "commits": [reference(commit)],
            "best_checkpoint": checkpoint, "patience_exhausted": epoch >= self.rounds,
            "progress": {"stopping": {"best_score": score}, "completed_rounds": epoch}}
        summary_path = directory / "training_summary.json"
        saved(summary_path, summary)
        return {"status": "completed", "summary": reference(summary_path),
                "config": reference(config_path), "provenance": reference(provenance_path)}, summary


class PolicyStudyTests(unittest.TestCase):
    def test_committed_selection_rejects_a_changed_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            study = FakeStudy(root, rounds=0)
            self.train(study)
            trial = study.state["trials"][0]
            (Path(trial["selected"]["path"]).parent / "state.pt").write_bytes(b"changed payload")
            with self.assertRaisesRegex(ValueError, "payload changed"):
                policy.select_committed(trial, {}, 125000)

    def test_completed_measurement_survives_outer_limit_and_tracking_failures(self):
        for outcome in ("partial", "tracking_failure"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as root:
                study = FakeStudy(root)
                original = study.stage
                def failing_stage(name, config, **options):
                    stage, summary = original(name, config, **options)
                    if name == "lr-3e-4-epoch01":
                        if outcome == "partial":
                            stage["status"] = "partial"
                        else:
                            study.state["failed_attempts"] = [{"name": name, "reason": "Tracking failed",
                                "directory": str(study.output / name), "artifacts": {
                                    "worker/training_summary.json": stage["summary"],
                                    "config.json": stage["config"], "provenance.json": stage["provenance"]}}]
                            raise RuntimeError("Tracking failed after a complete worker")
                    return stage, summary
                study.stage = failing_stage
                self.train(study)
                self.assertEqual(study.state["selected_trial_id"], "lr-3e-4")
                trial = next(item for item in study.state["trials"] if item["id"] == "lr-3e-4")
                self.assertEqual(trial["best_validation_nll"], 2.49)
                self.assertEqual(trial["completed_rounds"], 1)
                self.assertTrue(study.state["training_interrupted"])
                self.assertEqual(len(study.calls), 5)

    def test_supervisor_saved_config_loads_for_mixed_komi_assessment(self):
        from kenggo import learned_agent
        from kenggo.learner import ModelConfig
        from kenggo.continuation_data import read
        from test_continuation_data import fixture, refs
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            fitted, current, _ = fixture(root)
            inputs = refs(current, fitted)
            profile = root / "profile.json"
            save_json(profile, {"fixture_only": True})
            inputs["profile"] = reference(profile)
            study = FakeStudy(root, rounds=0)
            model = ModelConfig(blocks=1, channels=16)
            study.plan["parameters"].update(model=model.__dict__,
                training_rows=read(inputs["examples"])["counts"]["train"]["examples"])
            with patch.object(policy, "dataset_inputs", return_value=inputs):
                policy.train(study, {"train_monitor": {}, "epoch_validation": {}})
            config = {"version": "experiment_config_v1", **study.calls[0][1]}
            config_path = root / "trial-config.json"
            config_pin = save_json(config_path, config)
            artifacts = learned_agent.LearnedArtifacts("unused-checkpoint", "a" * 64,
                str(config_path), config_pin, "b" * 64, str(fitted.normalizer_path),
                fitted.normalizer_sha256, current.examples_sha256, model)
            _, training_komi, inference_komi = learned_agent._normalizer(artifacts)
            self.assertEqual(training_komi, (1, 13))
            self.assertEqual(inference_komi, (1, 13))

    def test_failed_stage_keeps_sealed_progress_in_trial_exposure(self):
        from kenggo.policy_review import _trial
        with tempfile.TemporaryDirectory() as root:
            study = FakeStudy(root, fail="lr-3e-4-epoch01")
            original = study.stage
            def failing_stage(name, config, **options):
                try:
                    return original(name, config, **options)
                except RuntimeError:
                    filename = Path(root) / "training_progress.json"
                    save_json(filename, {"status": "training", "completed_updates": 100,
                        "attempted_updates": 101, "uncertain_updates": 1, "commits": [],
                        "progress": {"sampler_step": 100, "epoch_metrics": [],
                                     "partial_round": {"updates": 100, "presented_examples": 12800}}})
                    study.state["failed_attempts"] = [{"name": name, "reason": "Fixture killed",
                        "directory": str(root), "artifacts": {"worker/training_progress.json": reference(filename)}}]
                    raise
            study.stage = failing_stage
            self.train(study)
            trial = next(item for item in study.state["trials"] if item["id"] == "lr-3e-4")
            failed = trial["training"][-1]
            self.assertEqual(failed["stage"]["status"], "failed")
            details, curves = _trial({"id": trial["id"], "parameters": {}, "training": [failed]})
            self.assertEqual(details["completed_updates"], 100)
            self.assertEqual(details["presented_examples"], 12800)
            self.assertEqual(details["uncertain_update_attempts"], 1)
            self.assertIsNone(curves)

    def train(self, study):
        examples = study.output / "fitting-examples.json"
        examples.write_bytes(json_bytes({"counts": {"train": {"examples": 3999997}}}))
        with patch.object(policy, "dataset_inputs", return_value={"examples": reference(examples)}):
            policy.train(study, {"train_monitor": {}, "epoch_validation": {}})

    def test_rotation_same_parent_and_validation_then_selected_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            study = FakeStudy(root)
            self.train(study)
        ids = [item["id"] for item in policy.TRIALS]
        self.assertEqual([config["arm"] for _, config in study.calls[:4]], ids)
        self.assertEqual([config["arm"] for _, config in study.calls[4:8]], ids[1:] + ids[:1])
        for _, config in study.calls[:4]:
            self.assertEqual(config["inputs"]["parent_checkpoint"]["path"], "parent")
            self.assertEqual(config["parameters"]["value_loss_weight"], 0)
            self.assertEqual(config["inputs"]["epoch_validation"], {})
        for trial in study.state["trials"]:
            self.assertEqual(trial["completed_rounds"], 8)
            self.assertEqual(trial["stop_reason"], "validation_plateau")
        self.assertEqual(study.state["selected_trial_id"], ids[0])

    def test_budget_stops_before_starting_an_incomplete_cycle(self):
        with tempfile.TemporaryDirectory() as root:
            study = FakeStudy(root, budget_cycles=3)
            self.train(study)
        self.assertEqual(len(study.calls), 12)
        self.assertTrue(all(trial["completed_rounds"] == 2 for trial in study.state["trials"]))
        self.assertTrue(all(trial["stop_reason"] == "shared_budget" for trial in study.state["trials"]))

    def test_failure_is_not_retried_and_preserves_complete_selections(self):
        with tempfile.TemporaryDirectory() as root:
            study = FakeStudy(root, fail="lr-3e-4-epoch01")
            self.train(study)
        self.assertEqual(len(study.calls), 5)
        self.assertTrue(study.state["training_interrupted"])
        self.assertTrue(all("epoch00" in trial["selected"]["path"] for trial in study.state["trials"]))

    def test_no_selection_without_all_declared_trials(self):
        with tempfile.TemporaryDirectory() as root:
            study = FakeStudy(root, fail="lr-1e-3-epoch00")
            with self.assertRaisesRegex(StopStudy, "Every declared"):
                self.train(study)
        self.assertEqual(len(study.calls), 1)

    def test_live_allowance_cannot_be_reset_and_preserves_reserves(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "study"
            plan = json.loads(Path("configs/e2_policy_tuning_v1.json").read_text())
            plan["parameters"]["output"] = str(output)
            filename = Path(root) / "plan.json"
            save_json(filename, plan)
            with patch.object(policy, "validate_plan"):
                study = policy.PolicyStudy(plan, reference(filename), output)
                with self.assertRaises(FileExistsError):
                    policy.PolicyStudy(plan, reference(filename), output)
            self.assertEqual((study.deadline - study.launch).total_seconds(), 259200)
            study.production_end = study.launch + timedelta(seconds=100)
            with self.assertRaises(StopStudy):
                study.require(3000, 3000, 1000, gpu=3000)
            study.finish("fixture", "No work ran")

    def test_frozen_protocol_rejects_allowance_and_trial_changes_before_io(self):
        plan = json.loads(Path("configs/e2_policy_tuning_v1.json").read_text())
        for mutation in ("wall", "trial", "output", "seed"):
            changed = deepcopy(plan)
            if mutation == "wall":
                changed["parameters"]["limits"]["wall_seconds"] += 1
            elif mutation == "trial":
                changed["parameters"]["trials"].pop()
            elif mutation == "output":
                changed["parameters"]["output"] += "-again"
            else:
                changed["seed"] += 1
            with patch.object(policy, "reference", side_effect=AssertionError("Unexpected input read")):
                with self.assertRaisesRegex(ValueError, "frozen inputs"):
                    policy.validate_plan(changed)


if __name__ == "__main__":
    unittest.main()
