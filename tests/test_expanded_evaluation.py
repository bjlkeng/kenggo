"""Synthetic raw common sets and new checkpoints; no production model or target access."""

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import torch

from kenggo import baseline_evaluation as worker
from kenggo.canonical_data import encoded
from kenggo.checkpoints import file_hash, save_checkpoint, save_json
from kenggo.encoding import INPUT_KEYS, apply_normalizer
from kenggo.examples import ExampleConfig, build_examples, fit_normalizer
from kenggo.go import Game
from kenggo.learned_agent import (EXPANDED_RECIPE, KOMI_INFERENCE_CONTRACT, LearnedArtifacts,
                                 LearnedPolicy, learned_agent_profile)
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.recorded_data import CommonInputs, RAW_VERSION, prepare_common_set
from kenggo.recorded_statistics import paired_report
from encoding_fixtures import corpus
from test_komi_data import record


SOURCE = {"commit": "a" * 40, "patch_sha256": "b" * 64}
RANKS = ["5k", "1d"]


def bundle(root, records, *, komi_study):
    root.mkdir()
    groups = {item["eligibility"][color + "_group"] for item in records for color in ("black", "white")}
    canonical, canonical_pin, splits, split_pin = corpus(root / "source", records,
        validation=("held",) if "held" in groups else (), test=("reserved",) if "reserved" in groups else ())
    examples = root / "examples"
    pin = build_examples(canonical, canonical_pin, splits, split_pin, examples, fixture_only=True,
        config=ExampleConfig(shard_size=2, hide_mover_per_mille=1000, hide_opponent_per_mille=1000),
        komi_study=komi_study)
    normalizer = root / "normalizer.json"
    normalizer_pin = save_json(normalizer, fit_normalizer(examples, pin, komi_study=komi_study))
    return CommonInputs(canonical, canonical_pin, splits, split_pin, examples, pin, normalizer, normalizer_pin)


def artifacts(root, inputs, *, training_komi, conditioned=True, expanded=True, changes=None):
    root.mkdir()
    profile = root / "profile.json"
    profile_pin = save_json(profile, {"fixture": "training dataset profile"})
    model_config = ModelConfig(1, 16, conditioned)
    parameters = {"model": asdict(model_config)}
    if expanded:
        parameters.update(recipe=EXPANDED_RECIPE, inference_contract=KOMI_INFERENCE_CONTRACT,
                          training_komi_halfpoints=list(training_komi))
    parameters.update(changes or {})
    refs = {"examples": {"path": str(inputs.examples_dir / "manifest.json"), "sha256": inputs.examples_sha256},
            "normalizer": {"path": str(inputs.normalizer_path), "sha256": inputs.normalizer_sha256},
            "profile": {"path": str(profile), "sha256": profile_pin}}
    config = root / "config.json"
    config_pin = save_json(config, {"version": "experiment_config_v1", "parameters": parameters, "inputs": refs})
    provenance = root / "provenance.json"
    provenance_pin = save_json(provenance, {"version": "experiment_provenance_v1", "inputs": refs,
                                           "source": SOURCE, "environment": {"fixture_only": True}})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(31)
        model = PopulationModel(model_config)
    checkpoint = save_checkpoint(root / "checkpoints", model, torch.optim.Adam(model.parameters()),
        step=1, progress={"next_step": 1}, config_sha256=config_pin, provenance_sha256=provenance_pin)
    return LearnedArtifacts(checkpoint["path"], checkpoint["sha256"], str(config), config_pin,
        provenance_pin, str(inputs.normalizer_path), inputs.normalizer_sha256, inputs.examples_sha256, model_config)


class ExpandedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        train6 = record("train6", komi=6.5, moves=";B[aa];W[bb]")
        train0 = record("train0", komi=0.5, moves=";B[cc];W[dd];B[ee];W[ff]")
        self.selected = []
        for index, (role, komi) in enumerate((
                ("population_development", 6.5), ("population_development", 0.5),
                ("validation_development", 6.5), ("validation_development", 0.5))):
            point = chr(ord("g") + index)
            item = record(f"development-{index}", komi=komi, day="2025-10-01",
                          moves=f";B[{point}a];W[{point}b];B[]", result="W+T" if komi == .5 else "B+R")
            if role == "validation_development":
                item["eligibility"]["black_group"] = "held"
            self.selected.append((f"set-{index}", role, item))
        confirmation = record("confirmation", komi=6.5, day="2025-11-03", moves=";B[na];W[nb]")
        final = record("final", komi=6.5, day="2025-11-04", moves=";B[oa];W[ob]")
        final["eligibility"]["black_group"] = "reserved"
        self.control = bundle(self.root / "control", [train6], komi_study=False)
        self.combined = bundle(self.root / "combined", [train6, train0,
            *(item for _, _, item in self.selected), confirmation, final], komi_study=True)
        self.common = []
        for name, role, item in self.selected:
            self.common.append((name, prepare_common_set(self.combined, role=role,
                source_ids=[item["source"]["source_url"]], aggregate_ranks=RANKS,
                fixture_only=True, raw_inputs=True, komi_study=True)))

    def ref(self, path):
        return {"path": str(path), "sha256": file_hash(path)}

    def config(self, learned=None, *, diagnostic=False):
        inputs = self.combined
        data, rules = self.root / "data.json", self.root / "rules.json"
        for path in (data, rules):
            save_json(path, {"fixture": path.stem}, replace=path.exists())
        refs = {"data": self.ref(data), "canonical": self.ref(inputs.canonical_path),
            "split": self.ref(inputs.split_dir / "manifest.json"),
            "examples": self.ref(inputs.examples_dir / "manifest.json"),
            "normalizer": self.ref(inputs.normalizer_path), "rules": self.ref(rules)}
        specs = []
        for name, common in self.common:
            path = self.root / (name + ".json")
            path.write_text(encoded(common.manifest))
            refs[name] = self.ref(path)
            specs.append({"id": name, "role": common.manifest["role"], "source_ids": common.manifest["source_ids"],
                          "aggregate_ranks": RANKS, "manifest_input": name})
        params = {"common_sets": specs, "common_input_mode": worker.RAW_COMMON_INPUT_MODE,
                  "batch_size": 2, "samples_per_position": 32, "unknown_rank_diagnostic": diagnostic}
        if learned is not None:
            training = json.loads(Path(learned.config_path).read_text())
            refs.update(training_examples=training["inputs"]["examples"],
                training_normalizer=self.ref(learned.normalizer_path),
                checkpoint=self.ref(learned.checkpoint_manifest), training_config=self.ref(learned.config_path),
                training_provenance=self.ref(Path(learned.config_path).parent / "provenance.json"))
            params["artifacts"] = asdict(learned)
        return {"version": "experiment_config_v1", "entity": "synthetic", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "expanded-evaluation", "seed": 20261118,
            "arm": "human-reference" if learned is None else "conditioned" if learned.model_config.rank_conditioned
                   else "unconditioned", "purpose": "baseline-evaluation", "budget_id": "synthetic-budget",
            "parameters": params, "inputs": refs}

    def run_config(self, config):
        output = self.root / f"evaluation-{len(list(self.root.glob('evaluation-*')))}"
        output.mkdir()
        config_path, provenance_path = output / "config.json", output / "provenance.json"
        config_pin = save_json(config_path, config)
        provenance_pin = save_json(provenance_path, {"version": "experiment_provenance_v1",
            "inputs": config["inputs"], "source": SOURCE, "environment": {"fixture_only": True}})
        with patch.object(worker, "_source_identity", return_value=SOURCE):
            result = worker.run_evaluation(config_path, config_pin, provenance_path, provenance_pin,
                                           output, fixture_only=True)
        return output, result

    def test_raw_sets_preserve_replay_labels_masks_and_only_decode_selected_sources(self):
        for _, common in self.common:
            self.assertEqual(common.manifest["version"], RAW_VERSION)
            for position in common.positions():
                game, row = position.observation.game, position.observation.inputs
                self.assertEqual(game.snapshot, position.metadata["state_sha256"])
                self.assertEqual(position.metadata["komi_halfpoints"], game.komi_halfpoints)
                self.assertEqual(float(row["scalars"][5]), position.metadata["ply"])
                self.assertEqual(float(row["scalars"][1]), game.komi_halfpoints / 2 *
                                 (-1 if position.metadata["mover"] == "B" else 1))
                self.assertTrue(row["legal"][position.targets["action"]])
                self.assertEqual(position.targets["value_valid"], game.komi_halfpoints == 13)
                self.assertFalse(position.targets["duration_valid"])
                self.assertFalse(row["scalar_valid"][12:].any())
                self.assertTrue(row["rank_valid"].all())
                with self.assertRaises(ValueError):
                    row["scalars"].setflags(write=True)
        chosen = self.selected[1][2]["source"]["source_url"]
        with patch("kenggo.examples.pq.read_table", wraps=pq.read_table) as reader:
            prepare_common_set(self.combined, role="population_development", source_ids=[chosen],
                aggregate_ranks=RANKS, fixture_only=True, raw_inputs=True, komi_study=True)
        payload_calls = [call for call in reader.call_args_list if
                         "canonical_json" in call.kwargs.get("columns", [])]
        self.assertEqual(len(payload_calls), 1)
        self.assertEqual(payload_calls[0].kwargs["filters"], [("source_url", "in", [chosen])])
        with self.assertRaises(ValueError):
            prepare_common_set(self.combined, role="population_development", source_ids=[chosen],
                               aggregate_ranks=RANKS, fixture_only=True)

    def test_each_checkpoint_uses_its_training_normalizer_without_mutating_common_rows(self):
        common = self.common[1][1]
        raw = worker.input_batch(list(common.positions()))
        frozen = {key: value.clone() for key, value in raw.items()}
        observed = []
        for index, (inputs, komi) in enumerate(((self.control, [13]), (self.combined, [1, 13]))):
            learned = artifacts(self.root / f"model-{index}", inputs, training_komi=komi)
            policy = LearnedPolicy.load(learned)
            self.addCleanup(policy.close)
            calls = []
            handle = policy._model.register_forward_pre_hook(lambda model, args:
                calls.append({key: value.clone() for key, value in args[0].items()}))
            self.addCleanup(handle.remove)
            actual = policy.predict_raw_batch(raw)
            expected = apply_normalizer({key: value.numpy() for key, value in raw.items()},
                                         json.loads(inputs.normalizer_path.read_text()))
            for key in INPUT_KEYS:
                np.testing.assert_array_equal(calls[0][key].numpy(), expected[key])
                torch.testing.assert_close(raw[key], frozen[key], rtol=0, atol=0)
            normalized = {key: torch.from_numpy(value) for key, value in expected.items()}
            repeated = policy.predict_batch(normalized, normalizer_sha256=inputs.normalizer_sha256)
            for first, second in zip(actual, repeated):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            policy.predict_raw_batch(raw, unknown_ranks=True)
            self.assertFalse(calls[-1]["rank_valid"].any())
            self.assertFalse(calls[-1]["ranks"].any())
            observed.append(calls[0]["scalars"][:, 5])
            self.assertEqual(policy.training_komi_halfpoints, tuple(komi))
            self.assertEqual(learned_agent_profile("fixture", learned, "5k", source_revision="a" * 40)
                             .data["komi_halfpoints"], 13)
        self.assertFalse(torch.equal(*observed))

    def test_old_checkpoints_and_incomplete_opt_ins_reject_new_komi(self):
        old = artifacts(self.root / "old", self.control, training_komi=[13], expanded=False)
        policy = LearnedPolicy.load(old)
        self.addCleanup(policy.close)
        raw = worker.input_batch(list(self.common[1][1].positions()))
        with self.assertRaisesRegex(ValueError, "6.5-point komi"):
            policy.predict_raw_batch(raw)
        with self.assertRaisesRegex(ValueError, "6.5-point komi"):
            policy.predict(Game.new(komi_halfpoints=1))
        for index, changes in enumerate(({"inference_contract": "unsupported"},
                {"training_komi_halfpoints": [True, 13]}, {"training_komi_halfpoints": [1]},
                {"recipe": "old_recipe"})):
            learned = artifacts(self.root / f"invalid-{index}", self.control, training_komi=[13], changes=changes)
            with self.assertRaises(ValueError):
                LearnedPolicy.load(learned)

    def test_four_arms_score_identical_sets_with_per_komi_masks_sampling_and_paired_statistics(self):
        outputs = []
        arms = []
        for index, (inputs, komis, conditioned) in enumerate((
                (self.control, [13], True), (self.control, [13], False),
                (self.combined, [1, 13], True), (self.combined, [1, 13], False))):
            arms.append((artifacts(self.root / f"arm-{index}", inputs,
                                   training_komi=komis, conditioned=conditioned), komis, conditioned))
        with ExitStack() as stack:
            for name in ("is_available", "is_initialized", "init", "synchronize", "manual_seed_all"):
                stack.enter_context(patch("torch.cuda." + name, side_effect=AssertionError("GPU access")))
            for learned, komis, conditioned in arms:
                output, result = self.run_config(self.config(learned, diagnostic=conditioned))
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["gpu_seconds"], 0)
                reports = [report for report in result["reports"] if report["variant"] == "ordinary"]
                self.assertEqual([report["common_set_sha256"] for report in reports],
                                 [common.sha256 for _, common in self.common])
                for report, (_, common) in zip(reports, self.common):
                    komi = common.manifest["rows"][0]["komi_halfpoints"]
                    self.assertEqual(report["out_of_training_support_komi_halfpoints"],
                                     [] if komi in komis else [komi])
                    metrics = json.loads((output / report["artifacts"]["metrics.json"]["path"]).read_text())
                    sampling = json.loads((output / report["artifacts"]["sampling.json"]["path"]).read_text())
                    self.assertEqual(set(metrics["by_komi"]), {str(komi)})
                    self.assertEqual(set(sampling["by_komi"]), {str(komi)})
                    self.assertEqual(sampling["draws_per_position"], 32)
                    self.assertEqual(metrics["overall"]["metrics"]["value_mse"]["masked_moves"],
                                     3 if komi == 1 else 0)
                outputs.append((output, reports))
        rows = worker.rows_for_common(self.common[0][1])
        predicted = [worker.load_predictions(output / reports[0]["artifacts"]["predictions.npz"]["path"],
                     expected_row_keys=[row["row_key"] for row in rows]) for output, reports in outputs]
        contrast = paired_report(rows, predicted[2], predicted[0], aggregate_ranks=RANKS)
        self.assertEqual(contrast["kind"], "candidate_minus_control")
        self.assertEqual(contrast["expected_outputs"], len(rows))

    def test_overlap_row_budget_and_unsupported_reference_fail_before_inference(self):
        learned = artifacts(self.root / "bounded", self.control, training_komi=[13])
        config = self.config(learned)
        config["parameters"]["common_sets"][1]["source_ids"] = config["parameters"]["common_sets"][0]["source_ids"]
        with patch.object(worker.LearnedPolicy, "load", side_effect=AssertionError("Inference loaded")):
            with self.assertRaisesRegex(ValueError, "nonoverlapping"):
                self.run_config(config)
            with patch.object(worker, "MAX_RAW_COMMON_ROWS", 2):
                with self.assertRaisesRegex(ValueError, "12,000-row"):
                    self.run_config(self.config(learned))
        with patch.object(worker, "HumanSL", side_effect=AssertionError("Native reference started")):
            with self.assertRaisesRegex(ValueError, "6.5-point komi"):
                self.run_config(self.config())


if __name__ == "__main__":
    unittest.main()
