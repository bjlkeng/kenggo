"""Synthetic recorded sets and CPU/native fakes; no held-out or network access."""

from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kenggo import baseline_evaluation as worker
from kenggo.canonical_data import encoded
from kenggo.checkpoints import file_hash, save_json
from kenggo.examples import ExampleConfig
from kenggo.go import Game, PASS
from kenggo.katago_human import (ENGINE_CONTEXT, ENGINE_REVISION, ENGINE_VERSION, POLICY,
    RANKS, REQUIRED_CONFIG, SETTINGS, HumanSL, ReferenceBudgetExceeded)
from kenggo.learned_agent import LearnedArtifacts
from kenggo.learner import ModelConfig
from kenggo.recorded_data import prepare_common_set
from kenggo.recorded_statistics import Prediction
from encoding_fixtures import record
from test_katago_human import FakeTransport, response
from test_recorded_data import bundle, source
from go_fixtures import ordinary_cases, point


SOURCE = {"commit": "a" * 40, "patch_sha256": "b" * 64}


class SamplingCategoryTests(unittest.TestCase):
    def test_categories_use_equal_rank_and_game_side_weights_with_separate_cells(self):
        rows = [dict(row_key=key, source_url=source_id, mover="B", source_rank_mover=rank,
                     group_mover=source_id, group_opponent="opponent", move_bin=phase,
                     main_time_bin=clock, action=action)
                for key, source_id, rank, phase, clock, action in (
                    ("a1", "a", "5k", "1-60", "20-30", 0),
                    ("a2", "a", "5k", "1-60", "20-30", 0),
                    ("b", "b", "5k", "181+", "10-<20", PASS),
                    ("c", "c", "1d", "61-180", "20-30", 0))]
        predictions = [Prediction(row["row_key"], tuple(0 if action == (0 if row["row_key"] == "b" else PASS)
                           else -math.inf for action in range(PASS + 1)), 0) for row in rows]
        report = worker._sampling_report(rows, predictions, seed=1, variant="ordinary", ranks=["5k", "1d"],
                                         games={row["row_key"]: Game.new() for row in rows})
        categories = report["move_categories"]
        self.assertEqual(categories["by_rank"]["5k"]["model"]["pass"], .5)
        self.assertEqual(categories["by_rank"]["5k"]["recorded"]["pass"], .5)
        self.assertEqual(categories["by_rank"]["5k"]["total_variation"], 0)
        self.assertEqual(categories["overall"]["model"]["pass"], .75)
        self.assertEqual(categories["overall"]["total_variation"], .5)
        self.assertIsNone(categories["by_phase"]["1-60"]["total_variation"])
        cell = next(item for item in categories["joint_cells"]
                    if (item["rank"], item["phase"], item["clock"]) == ("5k", "1-60", "20-30"))
        self.assertEqual(cell["total_variation"], 1)
        self.assertEqual(cell["counts"]["game_sides"], 1)
        self.assertEqual(cell["counts"]["moves"], 2)
        self.assertFalse(cell["promotion_supported"])
        self.assertEqual(cell["uncertainty"]["status"], "inconclusive")

    def test_category_comparison_uses_the_shared_capture_priority_for_both_paths(self):
        game = Game.from_fixture(ordinary_cases()[0]["board"])
        action = point(9, 9)
        row = dict(row_key="capture", source_url="source", mover="B", source_rank_mover="5k",
                   group_mover="a", group_opponent="b", move_bin="1-60", main_time_bin="20-30", action=action)
        prediction = Prediction("capture", tuple(0 if index == action else -math.inf for index in range(PASS + 1)), 0)
        result = worker._sampling_report([row], [prediction], seed=1, variant="ordinary", ranks=["5k"],
                                          games={"capture": game})["move_categories"]["overall"]
        self.assertEqual(result["model"]["capture"], 1)
        self.assertEqual(result["recorded"]["capture"], 1)
        self.assertEqual(result["total_variation"], 0)


class FakeLearnedPolicy:
    def __init__(self, artifacts, *, failure=None):
        self.artifacts, self.failure, self.calls, self.closed = artifacts, failure, [], False

    def predict_batch(self, inputs, *, normalizer_sha256):
        if normalizer_sha256 != self.artifacts.normalizer_sha256:
            raise AssertionError("Incorrect normalizer pin reached model")
        if set(inputs) != set(worker.INPUT_KEYS) or any(value.device.type != "cpu" for value in inputs.values()):
            raise AssertionError("Labels or non-CPU inputs reached model")
        self.calls.append({key: value.clone() for key, value in inputs.items()})
        logits = torch.full(inputs["legal"].shape, -1000., dtype=torch.float32)
        logits[:, PASS] = 0
        logits.masked_fill_(~inputs["legal"], -torch.inf)
        values = torch.full((len(logits),), .25, dtype=torch.float32)
        if self.failure == "missing":
            return logits[:-1], values[:-1]
        if self.failure == "invalid":
            logits[:, PASS] = torch.nan
        if self.failure == "second_call" and len(self.calls) == 2:
            raise RuntimeError("synthetic failed batch")
        return logits, values

    def close(self):
        self.closed = True


class BaselineEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_patch = patch.object(worker, "_source_identity", return_value=dict(SOURCE))
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        train = record("train", moves=";B[aa];W[bb];B[]")
        population = record("population", day="2025-10-01", moves=";B[cc];W[dd];B[]")
        validation = record("validation", day="2025-10-02", groups=("held", "other"),
                            moves=";B[ee];W[ff];B[]", result="W+T")
        self.inputs = bundle(self.root / "corpus", [train, population, validation],
                             config=ExampleConfig(hide_mover_per_mille=0, hide_opponent_per_mille=0))
        self.common = {}
        for name, role, item in (("population", "population_development", population),
                                  ("validation", "validation_development", validation)):
            self.common[name] = prepare_common_set(self.inputs, role=role, source_ids=[source(item)],
                                                   aggregate_ranks=["5k", "1d"], fixture_only=True)
        self.clients = []

    def json_ref(self, name, value, *, common=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if common:
            path.write_text(encoded(value))
            pin = file_hash(path)
        else:
            pin = save_json(path, value, replace=path.exists())
        return {"path": str(path), "sha256": pin}

    def ref(self, path):
        return {"path": str(path), "sha256": file_hash(path)}

    def config(self, arm="conditioned", *, diagnostic=False):
        data = self.json_ref("data.json", {"source": "synthetic"})
        refs = {"data": data, "canonical": self.ref(self.inputs.canonical_path),
            "split": self.ref(self.inputs.split_dir / "manifest.json"),
            "examples": self.ref(self.inputs.examples_dir / "manifest.json"),
            "normalizer": self.ref(self.inputs.normalizer_path),
            "rules": self.json_ref("rules.json", {"rules": "synthetic-pinned-rules"})}
        specs = []
        for name, common in self.common.items():
            refs[name + "_common"] = self.json_ref(name + ".json", common.manifest, common=True)
            specs.append({"id": name, "role": common.manifest["role"],
                "source_ids": common.manifest["source_ids"], "aggregate_ranks": common.manifest["aggregate_ranks"],
                "manifest_input": name + "_common"})
        params = {"common_sets": specs, "batch_size": 2, "samples_per_position": 32,
                  "unknown_rank_diagnostic": diagnostic}
        config = {"version": "experiment_config_v1", "entity": "synthetic", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "synthetic-evaluation", "arm": arm,
            "purpose": "baseline-evaluation", "seed": 123, "parameters": params, "inputs": refs,
            "budget_id": "synthetic-budget"}
        if arm != "human-reference":
            model = ModelConfig(1, 16, arm == "conditioned")
            training_inputs = {key: value for key, value in refs.items() if key in ("examples", "normalizer")}
            refs["training_config"] = self.json_ref("training-config.json", {
                "version": "experiment_config_v1", "parameters": {"model": asdict(model)}, "inputs": training_inputs})
            refs["training_provenance"] = self.json_ref("training-provenance.json", {
                "version": "experiment_provenance_v1", "inputs": training_inputs, "source": SOURCE,
                "environment": {"fixture_only": True}})
            refs["checkpoint"] = self.json_ref("checkpoint.json", {"fixture": "fake weights"})
            params["artifacts"] = asdict(LearnedArtifacts(refs["checkpoint"]["path"], refs["checkpoint"]["sha256"],
                refs["training_config"]["path"], refs["training_config"]["sha256"], refs["training_provenance"]["sha256"],
                refs["normalizer"]["path"], refs["normalizer"]["sha256"], refs["examples"]["sha256"], model))
        else:
            paths = {name: self.root / name for name in ("engine", "main_model", "human_model", "config")}
            for name, path in paths.items():
                path.write_text(name if name != "config" else "\n".join(
                    f"{key} = {value}" for key, value in REQUIRED_CONFIG.items()) + "\n")
            settings = {"schema": "katago_human_profile_v1", "profile_id": POLICY,
                "engine_version": ENGINE_VERSION, "engine_revision": ENGINE_REVISION,
                "main_model_name": "kata1-b10c128-s1141046784-d204142634", "human_model_name": "b18c384nbt-humanv0",
                "artifacts": {name: file_hash(path) for name, path in paths.items()}, "runtime_files": {},
                "library_dirs": [], "ranks": list(RANKS), "settings": dict(SETTINGS), "engine_context": dict(ENGINE_CONTEXT)}
            refs["human_profile"] = self.json_ref("human-profile.json", settings)
            refs["human_config"] = self.ref(paths.pop("config"))
            params["human_paths"] = {**{name: str(path) for name, path in paths.items()}, "runtime_root": str(self.root)}
            params["reference_budget"] = {"queries": 6, "cpu_seconds": 10, "wall_seconds": 20,
                "request_bytes": 400_000, "response_bytes": 400_000, "stderr_bytes": 1000, "trace_bytes": 500_000}
        return config

    def run_config(self, config, *, failure=None, native_handler=response):
        provenance = {"version": "experiment_provenance_v1", "inputs": config["inputs"],
                      "source": dict(SOURCE), "environment": {"fixture_only": True}}
        config_ref = self.json_ref("evaluation-config.json", config)
        provenance_ref = self.json_ref("evaluation-provenance.json", provenance)
        output = self.root / f"job-{len(list(self.root.glob('job-*')))}"
        output.mkdir()
        self.output = output

        def learned_factory(artifacts):
            client = FakeLearnedPolicy(artifacts, failure=failure)
            self.clients.append(client)
            return client

        def native_factory(profile, **kwargs):
            client = HumanSL(profile, **kwargs, _analysis=FakeTransport(native_handler))
            self.clients.append(client)
            return client

        with patch.object(worker.LearnedPolicy, "load", side_effect=learned_factory), patch.object(
                worker, "HumanSL", side_effect=native_factory):
            return worker.run_evaluation(config_ref["path"], config_ref["sha256"],
                provenance_ref["path"], provenance_ref["sha256"], output, fixture_only=True)

    def test_learned_roundtrip_preserves_exact_rows_masks_cpu_and_finite_log_losses(self):
        config = self.config()
        with ExitStack() as stack:
            for name in ("is_available", "is_initialized", "get_rng_state_all", "set_rng_state_all",
                         "manual_seed_all", "init", "synchronize"):
                stack.enter_context(patch("torch.cuda." + name, side_effect=AssertionError("Evaluation touched GPU")))
            stack.enter_context(patch("kenggo.checkpoints.restore_rng", side_effect=AssertionError("Restored training RNG")))
            python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
            result = self.run_config(config)
        self.assertEqual(result["completed_rows"], 6)
        self.assertEqual(result["attempted_rows"], 6)
        self.assertEqual(result["learned_batches_attempted"], 4)
        self.assertEqual(result["gpu_seconds"], 0)
        self.assertEqual(result["promotion"], "not_promoted")
        self.assertTrue(self.clients[0].closed)
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        for name, common in self.common.items():
            keys = [position.observation.row_key for position in common.positions()]
            directory = self.output / name / "ordinary"
            predictions = worker.load_predictions(directory / "predictions.npz", expected_row_keys=keys)
            self.assertEqual([item.row_key for item in predictions], keys)
            self.assertTrue(all(item.value == .25 for item in predictions))
            self.assertEqual(predictions[0].log_policy[0], -1000)
            with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
                self.assertTrue(np.all(archive["sampled_actions"] == PASS))
                self.assertTrue(np.all(archive["raw_pass_mass"] == 1))
                self.assertEqual(archive["row_keys"].dtype.kind, "U")
            metrics = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(metrics["overall"]["metrics"]["value_mse"]["masked_moves"], 3 if name == "validation" else 0)
            self.assertFalse(json.loads((directory / "timing.json").read_text())["per_position_latency_measured"])
            self.assertEqual(file_hash(self.output / name / "common-set.json"), common.sha256)
            with self.assertRaisesRegex(ValueError, "reordered"):
                worker.load_predictions(directory / "predictions.npz", expected_row_keys=keys[::-1])

    def test_unknown_rank_diagnostic_masks_both_ranks_and_changes_nothing_else(self):
        self.run_config(self.config(diagnostic=True))
        calls = self.clients[0].calls
        self.assertEqual(len(calls), 8)
        for ordinary, diagnostic in zip(calls[:2] + calls[4:6], calls[2:4] + calls[6:8]):
            self.assertTrue(ordinary["rank_valid"].all())
            self.assertFalse(diagnostic["rank_valid"].any())
            self.assertFalse(diagnostic["ranks"].any())
            for name in worker.INPUT_KEYS:
                if name not in ("ranks", "rank_valid"):
                    torch.testing.assert_close(ordinary[name], diagnostic[name], rtol=0, atol=0)
        for name, common in self.common.items():
            self.assertTrue(next(common.positions()).observation.inputs["rank_valid"].all())
            for variant in ("ordinary", "unknown_ranks"):
                metrics = json.loads((self.output / name / variant / "metrics.json").read_text())
                self.assertEqual(metrics["expected_outputs"], len(common))

    def test_every_common_set_must_match_before_any_model_is_loaded(self):
        for kind in ("content", "newline"):
            with self.subTest(kind=kind):
                config = self.config()
                ref = config["inputs"]["validation_common"]
                path = Path(ref["path"])
                if kind == "content":
                    manifest = json.loads(path.read_text())
                    manifest["rows"].pop()
                    path.write_text(encoded(manifest))
                else:
                    path.write_text(path.read_text() + "\n")
                ref["sha256"] = file_hash(path)
                with self.assertRaisesRegex(ValueError, "frozen manifest"):
                    self.run_config(config)
                self.assertFalse(self.clients)
                failure = json.loads((self.output / "failure.json").read_text())
                self.assertEqual(failure["attempted_rows"], 0)
                self.assertEqual(failure["completed_rows"], 0)

    def test_invalid_missing_and_failed_predictions_never_filter_or_retry(self):
        for kind in ("missing", "invalid", "second_call"):
            with self.subTest(kind=kind):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_config(self.config(), failure=kind)
                failure = json.loads((self.output / "failure.json").read_text())
                self.assertEqual(failure["automatic_retries"], 0)
                self.assertEqual(failure["completed_rows"], 2 if kind == "second_call" else 0)
                self.assertEqual(failure["attempted_rows"], 3 if kind == "second_call" else 2)
                self.assertEqual(len(self.clients[-1].calls), 2 if kind == "second_call" else 1)
                self.assertTrue(self.clients[-1].closed)
                self.assertFalse((self.output / "population/ordinary/metrics.json").exists())
                if kind == "second_call":
                    unit = failure["last_unit"]["prediction_chunk"]
                    self.assertEqual(file_hash(self.output / unit["path"]), unit["sha256"])

    def test_native_client_is_shared_budgeted_and_keeps_raw_policy_value_history_and_traces(self):
        config = self.config("human-reference")
        result = self.run_config(config)
        self.assertEqual(len(self.clients), 1)
        client = self.clients[0]
        self.assertEqual(result["native_cost"]["limits"], config["parameters"]["reference_budget"])
        self.assertEqual(result["native_cost"]["queries"], 6)
        self.assertEqual(result["native_cost"]["requests"], 7)
        self.assertEqual(result["trace_count"], 7)
        self.assertEqual(result["metadata_requests_completed"], 1)
        self.assertIsNone(result["native_neural_forwards"])
        requests = client.analysis.requests
        self.assertEqual([len(row["moves"]) for row in requests[1:]], [0, 1, 2, 0, 1, 2])
        self.assertEqual([row["overrideSettings"]["humanSLProfile"] for row in requests[1:]],
                         ["rank_5k", "rank_1d", "rank_5k"] * 2)
        for name, common in self.common.items():
            directory = self.output / name / "ordinary"
            predictions = worker.load_predictions(directory / "predictions.npz",
                expected_row_keys=[position.observation.row_key for position in common.positions()])
            np.testing.assert_allclose([item.value for item in predictions], [-.6, .6, -.6], rtol=0, atol=1e-15)
            self.assertEqual(predictions[0].log_policy[0], -math.inf)
            self.assertEqual(predictions[0].log_policy[PASS], 0)
            metrics = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(metrics["overall"]["metrics"]["nll"]["estimate"]["status"], "positive_infinity")
            self.assertTrue(json.loads((directory / "timing.json").read_text())["per_position_latency_measured"])
        for trace in (self.output / "traces").glob("*.json"):
            self.assertIn("response_sha256", json.loads(trace.read_text()))

    def test_native_failure_retains_attempts_cost_and_trace_without_retry(self):
        calls = 0

        def failing(request):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise ReferenceBudgetExceeded("synthetic_budget_exhausted")
            return response(request)

        with self.assertRaisesRegex(ValueError, "synthetic_budget_exhausted"):
            self.run_config(self.config("human-reference"), native_handler=failing)
        failure = json.loads((self.output / "failure.json").read_text())
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(calls, 3)
        self.assertEqual(failure["attempted_rows"], 2)
        self.assertEqual(failure["completed_rows"], 1)
        self.assertEqual(failure["trace_count"], 3)
        self.assertEqual(failure["native_cost"]["queries"], 2)
        self.assertEqual(failure["native_cost"]["requests"], 3)
        self.assertTrue(failure["native_cost"]["closed"])
        self.assertEqual(failure["automatic_retries"], 0)
        self.assertEqual(len(list((self.output / "traces").glob("*.json"))), 3)
        self.assertFalse((self.output / "validation").joinpath("ordinary").exists())

    def test_illegal_roles_artifact_substitution_and_invalid_limits_fail_before_prediction(self):
        cases = ("role", "artifact", "budget", "wrong_arm", "source")
        for kind in cases:
            with self.subTest(kind=kind):
                config = self.config("human-reference" if kind == "budget" else "conditioned")
                if kind == "role":
                    config["parameters"]["common_sets"][0]["role"] = "population_confirmation"
                elif kind == "artifact":
                    config["parameters"]["artifacts"]["config_sha256"] = "0" * 64
                elif kind == "budget":
                    config["parameters"]["reference_budget"]["wall_seconds"] = -1
                elif kind == "wrong_arm":
                    config["arm"] = "unconditioned"
                before = len(self.clients)
                context = patch.object(worker, "_source_identity", return_value={}) if kind == "source" else ExitStack()
                with context, self.assertRaises(ValueError):
                    self.run_config(config)
                self.assertEqual(len(self.clients), before)

    def test_raw_sampling_is_private_deterministic_legal_and_preserves_pass_probability(self):
        logs = tuple(math.log(.25) if action == 0 else math.log(.75) if action == PASS else -math.inf
                     for action in range(PASS + 1))
        prediction = Prediction("synthetic-row", logs, -.3)
        before = random.getstate()
        expected = worker.sample_actions(prediction, seed=5, variant="ordinary")
        self.assertEqual(worker.sample_actions(prediction, seed=5, variant="ordinary"), expected)
        self.assertNotEqual(worker.sample_actions(prediction, seed=5, variant="unknown_ranks"), expected)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(set(expected), {0, PASS})
        self.assertEqual(len(expected), 32)
        path = self.root / "synthetic.npz"
        worker._save_predictions(path, [prediction], seed=5, variant="ordinary")
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(archive["raw_pass_mass"][0], .75)
        self.assertEqual(worker.load_predictions(path, expected_row_keys=["synthetic-row"]), [prediction])

    def test_source_identity_checks_commit_patch_and_untracked_inputs(self):
        self.source_patch.stop()
        with patch.object(worker.subprocess, "check_output", side_effect=[b"", b"c" * 40 + b"\n", b"patch"]):
            self.assertEqual(worker._source_identity(), {"commit": "c" * 40, "patch_sha256": sha256(b"patch").hexdigest()})
        with patch.object(worker.subprocess, "check_output", return_value=b"untracked.py\n"):
            with self.assertRaisesRegex(ValueError, "Stage"):
                worker._source_identity()


if __name__ == "__main__":
    unittest.main()
