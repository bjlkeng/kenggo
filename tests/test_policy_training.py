"""Small CPU checks for policy-only updates and recoverable tuning state."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kenggo import continuation_training as worker
from kenggo import learned_agent
from kenggo import policy_recipe
from kenggo.checkpoints import (file_hash, json_hash, load_checkpoint, read_json, restore_rng,
                                rng_state, save_checkpoint, save_json, seed_all)
from kenggo.go import PASS
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.supervised import supervised_loss, train_step
import test_continuation_training as continuation_fixture
from test_continuation_data import fixture as lineage_fixture, refs as lineage_refs
from test_learner import model_inputs
from test_supervised import labels


class CPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_exact(left, right)
        else:
            self.assertEqual(actual, expected)


class PolicyLossTests(CPUChecks):
    def setUp(self):
        self.addCleanup(restore_rng, rng_state())
        seed_all(71)

    def test_default_loss_keeps_policy_and_signed_outcome_gradients(self):
        logits = torch.zeros(2, PASS + 1, requires_grad=True)
        values = torch.tensor([0.25, -0.5], requires_grad=True)
        legal = torch.zeros_like(logits, dtype=torch.bool)
        legal[:, [0, 1, PASS]] = True
        targets = labels()
        losses = supervised_loss(logits, values, targets, legal)
        expected_policy = torch.log(torch.tensor(3.0))
        expected_value = ((values - targets["value"]) ** 2).mean()
        self.assert_exact(losses["policy_loss"], expected_policy)
        self.assert_exact(losses["value_loss"], expected_value)
        self.assert_exact(losses["loss"], expected_policy + expected_value)
        losses["loss"].backward()
        self.assert_exact(values.grad, values.detach() - targets["value"])
        self.assertTrue((logits.grad[~legal] == 0).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_value_labels_cannot_change_policy_gradients_or_updates(self):
        first = PopulationModel(ModelConfig(blocks=1, channels=16))
        second = copy.deepcopy(first)
        inputs = model_inputs()
        first_optimizer = torch.optim.AdamW(first.parameters(), lr=0.0003, weight_decay=0.0001)
        second_optimizer = torch.optim.AdamW(second.parameters(), lr=0.0003, weight_decay=0.0001)
        first_metrics = train_step(first, first_optimizer, inputs, labels(values=(1, 1)),
                                   clock_dropout=0, value_weight=0)
        second_metrics = train_step(second, second_optimizer, inputs, labels(values=(-1, -1)),
                                    clock_dropout=0, value_weight=0)
        self.assertEqual(first_metrics["loss"], first_metrics["policy_loss"])
        self.assertEqual(first_metrics["loss"], second_metrics["loss"])
        self.assertNotEqual(first_metrics["value_loss"], second_metrics["value_loss"])
        self.assert_exact(first.state_dict(), second.state_dict())
        self.assert_exact(first_optimizer.state_dict(), second_optimizer.state_dict())
        for (name, parameter), (_, other) in zip(first.named_parameters(), second.named_parameters()):
            if name.startswith("value."):
                self.assertIsNone(parameter.grad)
                self.assertIsNone(other.grad)
            else:
                self.assertIsNotNone(parameter.grad)
                self.assert_exact(parameter.grad, other.grad)

    def test_frozen_head_survives_adamw_decay_while_policy_trunk_changes(self):
        model = PopulationModel(ModelConfig(blocks=1, channels=16))
        before = copy.deepcopy(model.state_dict())
        params = policy_recipe.DEFAULTS | {"weight_decay": 0.0001}
        optimizer = worker._policy_optimizer(model, params)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0001)
        self.assertTrue(all(not parameter.requires_grad and id(parameter) not in optimized
                            for parameter in model.value.parameters()))
        for _ in range(3):
            train_step(model, optimizer, model_inputs(), labels(), clock_dropout=0, value_weight=0)
        for name, parameter in model.named_parameters():
            if name.startswith("value."):
                self.assertIsNone(parameter.grad)
                self.assert_exact(parameter, before[name])
        self.assertFalse(torch.equal(model.stem.weight, before["stem.weight"]))
        self.assertFalse(torch.equal(model.pass_policy.weight, before["pass_policy.weight"]))

    def test_invalid_value_weights_are_rejected(self):
        logits = torch.zeros(2, PASS + 1)
        values = torch.zeros(2)
        legal = torch.ones_like(logits, dtype=torch.bool)
        for weight in (True, -1, float("inf"), float("nan"), "0"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                supervised_loss(logits, values, labels(), legal, value_weight=weight)


class PolicyScheduleTests(CPUChecks):
    def test_plateau_reduces_lr_then_meaningful_progress_resets_its_counter(self):
        params = policy_recipe.DEFAULTS | {"min_delta": 0.002}
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.0003)
        progress = {}
        expected = ((2.0, 0.0003, 0), (1.999, 0.0003, 1), (2.0, 0.00009, 0),
                    (1.998, 0.00009, 0), (1.998, 0.00009, 1), (1.998, 0.000027, 0))
        for index, (score, rate, bad_rounds) in enumerate(expected):
            before = optimizer.param_groups[0]["lr"]
            used, following = policy_recipe.update_schedule(progress, optimizer, params, score,
                                                             initial=index == 0)
            self.assertEqual(used, before)
            self.assertAlmostEqual(following, rate)
            self.assertEqual(progress["policy_schedule"]["bad_rounds"], bad_rounds)
            self.assertEqual(optimizer.param_groups[0]["lr"], following)
        for _ in range(30):
            policy_recipe.update_schedule(progress, optimizer, params, 2.0, initial=False)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.000001)
        reductions = progress["policy_schedule"]["reductions"]
        for _ in range(4):
            policy_recipe.update_schedule(progress, optimizer, params, 2.0, initial=False)
        self.assertEqual(progress["policy_schedule"]["reductions"], reductions)

    def test_stopping_requires_both_minimum_rounds_and_patience(self):
        params = policy_recipe.DEFAULTS
        for rounds, bad_rounds, expected in ((7, 7, False), (8, 5, False), (8, 6, True), (9, 0, False)):
            with self.subTest(rounds=rounds, bad_rounds=bad_rounds):
                progress = {"completed_rounds": rounds, "stopping": {"bad_rounds": bad_rounds}}
                self.assertEqual(policy_recipe.stopped(progress, params), expected)

    def test_saved_schedule_requires_a_finite_numeric_reference(self):
        progress = {"initial_evaluation_complete": True, "completed_rounds": 2,
            "stopping": {"bad_rounds": 2, "tolerance_reference": 2.0},
            "policy_schedule": {"reference": 2.0, "bad_rounds": 0, "reductions": 1,
                                "learning_rate": 0.00009}}
        policy_recipe.validate_schedule(progress, policy_recipe.DEFAULTS)
        for reference in (None, True, "2.0", float("inf"), float("nan")):
            changed = copy.deepcopy(progress)
            changed["stopping"]["tolerance_reference"] = reference
            changed["policy_schedule"]["reference"] = reference
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                policy_recipe.validate_schedule(changed, policy_recipe.DEFAULTS)


class PolicyInferenceTests(CPUChecks):
    def test_policy_checkpoint_keeps_old_fitting_normalizer_and_mixed_komi_inference(self):
        self.addCleanup(restore_rng, rng_state())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted, current, _ = lineage_fixture(root)
            inputs = lineage_refs(current, fitted)
            profile = root / "profile.json"
            inputs["profile"] = {"path": str(profile),
                                 "sha256": save_json(profile, {"fixture_only": True})}
            model_config = ModelConfig(blocks=1, channels=16)
            config = {"version": "experiment_config_v1", "inputs": inputs,
                "parameters": {"recipe": policy_recipe.RECIPE, "model": model_config.__dict__,
                    "inference_contract": "komi_comparison_v1", "training_komi_halfpoints": [1, 13]}}
            config_path = root / "config.json"
            config_pin = save_json(config_path, config)
            provenance_pin = json_hash({"synthetic_policy_inference": True})
            model = PopulationModel(model_config)
            optimizer = worker._policy_optimizer(model, policy_recipe.DEFAULTS)
            checkpoint = save_checkpoint(root / "checkpoints", model, optimizer, step=0, progress={},
                config_sha256=config_pin, provenance_sha256=provenance_pin)
            artifacts = learned_agent.LearnedArtifacts(checkpoint["path"], checkpoint["sha256"],
                str(config_path), config_pin, provenance_pin, str(fitted.normalizer_path),
                fitted.normalizer_sha256, current.examples_sha256, model_config)
            policy = learned_agent.LearnedPolicy.load(artifacts)
            self.addCleanup(policy.close)
            self.assertEqual(policy._normalizer["examples_sha256"], fitted.examples_sha256)
            self.assertNotEqual(fitted.examples_sha256, current.examples_sha256)
            self.assertEqual(policy.training_komi_halfpoints, (1, 13))
            self.assertEqual(policy.inference_komi_halfpoints, (1, 13))
            for komi in (0.5, 6.5):
                batch = model_inputs()
                batch["scalars"][:, 1] = (1 - 2 * batch["scalars"][:, 0]) * komi
                logits, values = policy.predict_raw_batch(batch)
                self.assertTrue(torch.isfinite(logits[batch["legal"]]).all())
                self.assertTrue(torch.isfinite(values).all())
                self.assertFalse(logits.requires_grad)


class PolicyContinuationTests(CPUChecks):
    def setUp(self):
        self.fixture = continuation_fixture.ContinuationTrainingTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.config["arm"] = "lr-3e-4"
        self.fixture.config["parameters"].update(policy_recipe.DEFAULTS,
            recipe=policy_recipe.RECIPE, stage_id="policy-fixture")
        self.root, self.data = self.fixture.root, self.fixture.data

    def payload(self, checkpoint):
        manifest = read_json(checkpoint["path"], checkpoint["sha256"])
        return torch.load(Path(checkpoint["path"]).parent / manifest["file"],
                          map_location="cpu", weights_only=True)

    def restore(self, checkpoint):
        config = read_json(checkpoint["config"]["path"], checkpoint["config"]["sha256"])
        model = PopulationModel(ModelConfig(**config["parameters"]["model"]))
        params, _ = worker._parameters(config, fixture_only=True)
        optimizer = worker._policy_optimizer(model, params)
        cursor = load_checkpoint(checkpoint["path"], checkpoint["sha256"], model, optimizer,
            config_sha256=checkpoint["config"]["sha256"],
            provenance_sha256=checkpoint["provenance"]["sha256"])
        return model, optimizer, cursor

    def plateau(self):
        evaluate = worker.evaluate_epoch

        def measure(model, diagnostic, **options):
            result = evaluate(model, diagnostic, **options)
            if diagnostic.purpose == "epoch_validation":
                result["nll"] = 2.0
            return result

        return mock.patch.object(worker, "evaluate_epoch", side_effect=measure)

    def initial(self, name="initial"):
        config = copy.deepcopy(self.fixture.config)
        config["parameters"]["evaluate_only"] = True
        return self.fixture.execute(name, config)

    def tamper(self, checkpoint, name, change):
        payload = self.payload(checkpoint)
        change(payload)
        directory = self.root / name
        directory.mkdir()
        torch.save(payload, directory / "state.pt")
        manifest = read_json(checkpoint["path"], checkpoint["sha256"])
        manifest["sha256"] = file_hash(directory / "state.pt")
        path = directory / "manifest.json"
        return {**checkpoint, "path": str(path), "sha256": save_json(path, manifest)}

    def test_all_arms_reset_parent_optimizer_and_share_exact_initial_weights_and_rng(self):
        parent = self.payload(self.fixture.config["inputs"]["parent_checkpoint"])
        self.assertTrue(parent["optimizer"]["state"])
        for trial in policy_recipe.TRIALS:
            with self.subTest(arm=trial["id"]):
                config = copy.deepcopy(self.fixture.config)
                config["arm"] = trial["id"]
                config["parameters"].update(evaluate_only=True,
                    **{key: value for key, value in trial.items() if key != "id"})
                result = self.fixture.execute(trial["id"], config)
                saved = self.payload(result["latest_checkpoint"])
                self.assert_exact(saved["model"], parent["model"])
                self.assert_exact(saved["rng"], parent["rng"])
                self.assertEqual(saved["optimizer"]["state"], {})
                group = saved["optimizer"]["param_groups"][0]
                self.assertEqual(group["lr"], trial["learning_rate"])
                self.assertEqual(group["weight_decay"], trial["weight_decay"])
                self.assertFalse(any(name.startswith("value.") for name in group["param_names"]))
                self.assertEqual(result["progress"]["sampler_step"], 0)
                self.assertEqual(result["progress"]["lifetime_updates"], 1)
                self.assertEqual(result["best_checkpoint"], result["latest_checkpoint"])

    def test_same_stage_resume_keeps_optimizer_moments_and_reduced_lr(self):
        with self.plateau():
            initial = self.initial()
            first = self.fixture.execute("round-one", self.fixture.continued(initial["latest_checkpoint"]))
            second = self.fixture.execute("round-two", self.fixture.continued(first["latest_checkpoint"]))
            saved = self.payload(second["latest_checkpoint"])
            self.assertTrue(saved["optimizer"]["state"])
            self.assertAlmostEqual(saved["optimizer"]["param_groups"][0]["lr"], 0.00009)
            self.assertEqual(second["progress"]["policy_schedule"]["reductions"], 1)
            observed = []

            def step(model, optimizer, *args, **options):
                if not observed:
                    observed.append(copy.deepcopy(optimizer.state_dict()))
                    self.assertTrue(all(not parameter.requires_grad for parameter in model.value.parameters()))
                return train_step(model, optimizer, *args, **options)

            with mock.patch.object(worker, "train_step", side_effect=step):
                third = self.fixture.execute("round-three", self.fixture.continued(second["latest_checkpoint"]))
        self.assert_exact(observed[0], saved["optimizer"])
        self.assertEqual(third["progress"]["sampler_step"], 12)
        self.assertEqual(third["progress"]["lifetime_updates"], 13)
        self.assertEqual(third["progress"]["policy_schedule"]["bad_rounds"], 1)
        self.assertEqual(third["progress"]["stopping"]["bad_rounds"], 3)
        self.assertFalse(third["patience_exhausted"])
        config = self.fixture.continued(second["latest_checkpoint"], stage_id="fresh-policy-stage", evaluate_only=True)
        reset = self.fixture.execute("fresh-stage", config)
        fresh = self.payload(reset["latest_checkpoint"])
        self.assert_exact(fresh["model"], saved["model"])
        self.assertEqual(fresh["optimizer"]["state"], {})
        self.assertEqual(fresh["optimizer"]["param_groups"][0]["lr"], 0.0003)
        self.assertEqual(reset["progress"]["sampler_step"], 0)
        self.assertEqual(reset["progress"]["lifetime_updates"], 9)

    def test_partial_round_resume_matches_rng_optimizer_metrics_and_next_update(self):
        initial = self.initial()
        config = self.fixture.continued(initial["latest_checkpoint"])
        uninterrupted = self.fixture.execute("uninterrupted", config)
        updates = []

        def step(*args, **options):
            result = train_step(*args, **options)
            updates.append(result)
            return result

        def stop(_):
            if len(updates) == 2:
                raise worker.BudgetStop("synthetic stop inside a policy interval")

        with mock.patch.object(worker, "train_step", side_effect=step), \
                mock.patch.object(worker._Budget, "check", stop):
            partial = self.fixture.execute("partial", config)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["progress"]["partial_round"]["updates"], 2)
        self.assertEqual(partial["progress"]["policy_schedule"], initial["progress"]["policy_schedule"])
        self.assertEqual(partial["progress"]["stopping"], initial["progress"]["stopping"])
        resumed = self.fixture.execute("resumed", self.fixture.continued(partial["latest_checkpoint"]))
        self.assertEqual(resumed["completed_updates"], 2)
        for key in ("sampler_step", "lifetime_updates", "lifetime_examples", "epoch_metrics",
                    "policy_schedule", "stopping"):
            self.assertEqual(resumed["progress"][key], uninterrupted["progress"][key])
        expected = self.payload(uninterrupted["latest_checkpoint"])
        actual = self.payload(resumed["latest_checkpoint"])
        for key in ("model", "optimizer", "rng", "module_training"):
            self.assert_exact(actual[key], expected[key])
        batch = self.data.batch(4, 4, 19, full_batch=True)
        expected_model, expected_optimizer, _ = self.restore(uninterrupted["latest_checkpoint"])
        expected_metrics = train_step(expected_model, expected_optimizer, *batch, value_weight=0)
        actual_model, actual_optimizer, _ = self.restore(resumed["latest_checkpoint"])
        actual_metrics = train_step(actual_model, actual_optimizer, *batch, value_weight=0)
        self.assertEqual(actual_metrics, expected_metrics)
        self.assert_exact(actual_model.state_dict(), expected_model.state_dict())

    def test_checkpoint_evaluation_preserves_mixed_modes_buffers_rng_and_next_policy_update(self):
        diagnostic = worker.load_diagnostic(self.fixture.validation, "epoch_validation",
            self.data.normalizer_sha256, fixture_only=True)
        model = continuation_fixture.StatefulModel()
        optimizer = worker._policy_optimizer(model, policy_recipe.DEFAULTS)
        model.observation_dropout.eval()
        model.value.eval()
        before = copy.deepcopy(model.state_dict())
        modes = [module.training for module in model.modules()]
        random_state = rng_state()
        batch = self.data.batch(0, 4, 19, full_batch=True)
        expected_metrics = train_step(model, optimizer, *batch, value_weight=0)
        expected_weights = copy.deepcopy(model.state_dict())
        model.load_state_dict(before)
        optimizer = worker._policy_optimizer(model, policy_recipe.DEFAULTS)
        for module, mode in zip(model.modules(), modes):
            module.training = mode
        restore_rng(random_state)
        result = worker.evaluate_epoch(model, diagnostic, batch_size=2)
        self.assertEqual(result["rows"], 2)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assert_exact(model.state_dict(), before)
        self.assert_exact(rng_state(), random_state)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.value.parameters()))
        self.assertEqual(train_step(model, optimizer, *batch, value_weight=0), expected_metrics)
        self.assert_exact(model.state_dict(), expected_weights)

    def test_resume_rejects_changed_optimizer_semantics_before_any_update(self):
        previous = self.fixture.execute("previous")
        mutations = {
            "names": lambda state: state["optimizer"]["param_groups"][0]["param_names"].reverse(),
            "learning-rate": lambda state: state["optimizer"]["param_groups"][0].update(lr=0.001),
            "betas": lambda state: state["optimizer"]["param_groups"][0].update(betas=(0.5, 0.999)),
            "weight-decay": lambda state: state["optimizer"]["param_groups"][0].update(weight_decay=0.0001),
            "schedule": lambda state: state["progress"]["policy_schedule"].update(learning_rate=0.001),
        }
        for name, change in mutations.items():
            altered = self.tamper(previous["latest_checkpoint"], "tampered-" + name, change)
            config = self.fixture.continued(altered)
            with self.subTest(field=name), mock.patch.object(worker, "train_step") as update:
                with self.assertRaisesRegex(ValueError, "optimizer"):
                    self.fixture.execute("rejected-" + name, config)
                update.assert_not_called()

    def test_resume_rejects_tampered_plateau_state_before_any_update(self):
        with self.plateau():
            initial = self.initial()
            first = self.fixture.execute("first", self.fixture.continued(initial["latest_checkpoint"]))
            previous = self.fixture.execute("second", self.fixture.continued(first["latest_checkpoint"]))
        self.assertEqual(previous["progress"]["completed_rounds"], 2)
        self.assertEqual(previous["progress"]["policy_schedule"]["reductions"], 1)

        def change_rate(state):
            state["optimizer"]["param_groups"][0]["lr"] = 0.00015
            state["progress"]["policy_schedule"]["learning_rate"] = 0.00015

        mutations = {
            "reference": lambda state: state["progress"]["policy_schedule"].update(reference=2.1),
            "bad-rounds": lambda state: state["progress"]["policy_schedule"].update(bad_rounds=1),
            "reductions": lambda state: state["progress"]["policy_schedule"].update(reductions=2),
            "fractional-reductions": lambda state: state["progress"]["policy_schedule"].update(reductions=1.0),
            "formula": change_rate,
            "missing-field": lambda state: state["progress"]["policy_schedule"].pop("bad_rounds"),
            "extra-field": lambda state: state["progress"]["policy_schedule"].update(unexpected=1),
            "unevaluated": lambda state: state["progress"].update(initial_evaluation_complete=False),
        }
        for name, change in mutations.items():
            altered = self.tamper(previous["latest_checkpoint"], "schedule-" + name, change)
            config = self.fixture.continued(altered)
            with self.subTest(field=name), mock.patch.object(worker, "train_step") as update:
                with self.assertRaisesRegex(ValueError, "schedule"):
                    self.fixture.execute("reject-schedule-" + name, config)
                update.assert_not_called()

    def test_malformed_recipe_and_undeclared_trial_pairs_fail_validation(self):
        mutations = ({"recipe": "unknown"}, {"value_loss_weight": 1}, {"learning_rate": True},
            {"learning_rate": float("nan")}, {"learning_rate": 0.01}, {"weight_decay": -1},
            {"learning_rate": 0.001, "weight_decay": 0.0001},
            {"learning_rate": 0.0001, "weight_decay": 0.0001},
            {"lr_patience": 2.0}, {"lr_patience": True}, {"min_rounds": 8.0},
            {"min_rounds": 7}, {"patience": 3}, {"lr_factor": 0.5}, {"min_lr": 0},
            {"clock_dropout": 0.2}, {"min_delta": 0.003}, {"unrecognized": 1})
        for mutation in mutations:
            config = copy.deepcopy(self.fixture.config)
            config["parameters"].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker._parameters(config, fixture_only=True)


if __name__ == "__main__":
    unittest.main()
