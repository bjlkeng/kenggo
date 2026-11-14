"""Synthetic CPU inference and match-boundary checks; no trained-model claims."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kenggo.checkpoints import file_hash, json_hash, save_checkpoint, save_json
from kenggo.encoding import (Context, INPUT_KEYS, NORMALIZED, SCALARS, VERSION,
                             apply_normalizer, encode)
from kenggo.go import Game, PASS
from kenggo.kgs_source import RANKS
from kenggo.learned_agent import (CPU_THREADS, POLICY, LearnedArtifacts, LearnedMatchFactory,
                                 LearnedPolicy, learned_agent_profile)
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.matches import ExperimentBudget, InfrastructureError, MatchLimits, run_game, weak_profile


REVISION = "a" * 40
PROVENANCE = json_hash({"source": "synthetic-inference-test"})
EXAMPLES = json_hash({"train": "synthetic-examples"})


@dataclass(frozen=True)
class InvalidOutputFactory:
    factory: LearnedMatchFactory

    def __call__(self, profile, seed):
        agent = self.factory(profile, seed)
        with torch.no_grad():
            agent.client._model.pass_policy.bias.fill_(torch.nan)
        return agent


class LearnedAgentTests(unittest.TestCase):
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
        self.original_rng = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, self.original_rng)
        torch.random.default_generator.manual_seed(19)
        self.artifacts, self.model, self.normalizer = self.make_artifacts()

    def make_artifacts(self, *, conditioned=True, zero=False, pass_bias=0.0):
        directory = self.root / f"fixture-{len(list(self.root.iterdir()))}"
        directory.mkdir()
        model_config = ModelConfig(1, 16, conditioned)
        model = PopulationModel(model_config)
        if zero:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.pass_policy.bias.fill_(pass_bias)
                model.value[-2].bias.fill_(math.atanh(0.25))
        count = [4 if index in NORMALIZED else 0 for index in range(len(SCALARS))]
        mean, scale = [0.0] * len(SCALARS), [1.0] * len(SCALARS)
        mean[5], scale[5] = 2.0, 2.0
        normalizer = {"version": "scalar_normalizer_v1", "encoding": VERSION, "fit_role": "train",
                      "examples_sha256": EXAMPLES, "fixture_only": True, "fitted": True,
                      "scalars": list(SCALARS), "count": count, "mean": mean, "scale": scale}
        normalizer_path = directory / "normalizer.json"
        normalizer_pin = save_json(normalizer_path, normalizer)
        config = {"version": "experiment_config_v1", "parameters": {"model": asdict(model_config)},
                  "inputs": {"normalizer": {"path": "normalizer.json", "sha256": normalizer_pin},
                             "examples": {"path": "examples/manifest.json", "sha256": EXAMPLES}}}
        config_path = directory / "config.json"
        config_pin = save_json(config_path, config)
        checkpoint = save_checkpoint(directory / "checkpoints", model,
            torch.optim.Adam(model.parameters()), step=3, progress={"next_step": 3},
            config_sha256=config_pin, provenance_sha256=PROVENANCE)
        artifacts = LearnedArtifacts(checkpoint["path"], checkpoint["sha256"], str(config_path), config_pin,
            PROVENANCE, str(normalizer_path), normalizer_pin, EXAMPLES, model_config)
        return artifacts, model, normalizer

    def mutate_checkpoint(self, mutation, *, artifacts=None):
        artifacts = artifacts or self.artifacts
        path = Path(artifacts.checkpoint_manifest)
        payload = torch.load(path.parent / "state.pt", map_location="cpu", weights_only=True)
        mutation(payload)
        torch.save(payload, path.parent / "state.pt")
        manifest = json.loads(path.read_text())
        manifest["sha256"] = file_hash(path.parent / "state.pt")
        return replace(artifacts, checkpoint_sha256=save_json(path, manifest, replace=True))

    def replace_normalizer(self, normalizer):
        pin = save_json(self.artifacts.normalizer_path, normalizer, replace=True)
        config = json.loads(Path(self.artifacts.config_path).read_text())
        config["inputs"]["normalizer"]["sha256"] = pin
        config_pin = save_json(self.artifacts.config_path, config, replace=True)
        return replace(self.artifacts, normalizer_sha256=pin, config_sha256=config_pin)

    def batch(self, games=None, context=Context()):
        rows = [apply_normalizer(encode(game, context), self.normalizer)
                for game in (games or [Game.new(), Game.new().play("B", 0)])]
        return {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in INPUT_KEYS}

    def profile(self, artifacts=None, rank="5k"):
        return learned_agent_profile("candidate", artifacts or self.artifacts, rank,
                                     source_revision=REVISION)

    def test_serialized_weights_and_batch_predictions_match_without_gradients(self):
        policy = LearnedPolicy.load(self.artifacts)
        self.assertEqual(torch.get_num_threads(), CPU_THREADS)
        self.assertTrue(all(not module.training for module in policy._model.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in policy._model.parameters()))
        for name, value in policy._model.state_dict().items():
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)
        inputs = self.batch(context=Context("5k", "1d"))
        with torch.inference_mode():
            expected = self.model.eval()(inputs)
        predicted = policy.predict_batch(inputs, normalizer_sha256=self.artifacts.normalizer_sha256)
        for actual, target in zip(predicted, expected):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
            self.assertFalse(actual.requires_grad)
            self.assertIsNone(actual.grad_fn)
        self.assertTrue(torch.isneginf(predicted[0][~inputs["legal"]]).all())
        self.assertTrue((predicted[0].softmax(1)[:, PASS] > 0).all())

    def test_cpu_loading_ignores_saved_cuda_optimizer_modes_and_all_training_rngs(self):
        def mutation(payload):
            payload["rng"]["cuda"] = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
            payload["optimizer"] = {"deliberately": "unusable resume state"}
            payload["module_training"] = {name: True for name in payload["module_training"]}
        artifacts = self.mutate_checkpoint(mutation)
        torch_before, python_before, numpy_before = torch.get_rng_state(), random.getstate(), np.random.get_state()
        with ExitStack() as stack:
            for target in ("torch.cuda.is_available", "torch.cuda.is_initialized", "torch.cuda.device_count",
                           "torch.cuda.get_rng_state_all", "torch.cuda.set_rng_state_all",
                           "torch.cuda.manual_seed_all", "torch.cuda.init", "torch.cuda.synchronize",
                           "kenggo.checkpoints.restore_rng", "torch.optim.Adam.load_state_dict"):
                stack.enter_context(patch(target, side_effect=AssertionError("Inference touched training/CUDA state")))
            policy = LearnedPolicy.load(artifacts)
            policy.predict(Game.new())
            agent = LearnedMatchFactory(artifacts, REVISION)(self.profile(artifacts).data, 123)
            agent.choose(Game.new())
            agent.close()
            policy.close()
        torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_artifact_pins_config_shape_and_conditioning_cannot_be_substituted(self):
        changes = [{name: "0" * 64} for name in ("checkpoint_sha256", "config_sha256", "provenance_sha256",
                                                 "normalizer_sha256", "examples_sha256")]
        changes += [{"model_config": ModelConfig(1, 16, False)}, {"model_config": ModelConfig(2, 16)},
                    {"model_config": ModelConfig(1, 24)}, {"checkpoint_manifest": str(self.root / "missing")}]
        for change in changes:
            with self.subTest(change=change):
                artifacts = replace(self.artifacts, **change)
                rank = "5k" if artifacts.model_config.rank_conditioned else None
                with self.assertRaises(InfrastructureError):
                    LearnedMatchFactory(artifacts, REVISION)(self.profile(artifacts, rank).data, 123)
        with self.assertRaises(ValueError):
            replace(self.artifacts, normalizer_sha256="not-a-pin")

    def test_changed_state_bytes_fail_before_deserialization(self):
        blob = Path(self.artifacts.checkpoint_manifest).parent / "state.pt"
        blob.write_bytes(blob.read_bytes() + b"changed")
        with patch("torch.load", side_effect=AssertionError("Unverified state was deserialized")):
            with self.assertRaisesRegex(ValueError, "state differs"):
                LearnedPolicy.load(self.artifacts)

    def test_valid_hashes_do_not_hide_payload_structure_or_bad_weights(self):
        mutations = [lambda x: x.update(version="other"), lambda x: x.update(step=True),
            lambda x: x.update(config_sha256="0" * 64), lambda x: x.update(provenance_sha256="0" * 64),
            lambda x: x["component_types"].update(model="torch.nn.Linear"),
            lambda x: x["model"].pop("pass_policy.bias"),
            lambda x: x["model"].update(**{"pass_policy.bias": torch.zeros(2)}),
            lambda x: x["model"].update(**{"pass_policy.bias": torch.zeros(1, dtype=torch.float64)}),
            lambda x: x["model"]["pass_policy.bias"].fill_(torch.nan),
            lambda x: x["model"]["pass_policy.bias"].fill_(torch.inf),
            lambda x: x["module_training"].update(extra=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                artifacts, _, _ = self.make_artifacts()
                artifacts = self.mutate_checkpoint(mutation, artifacts=artifacts)
                with self.assertRaises(ValueError):
                    LearnedPolicy.load(artifacts)

    def test_normalizer_must_be_fitted_on_the_pinned_training_examples(self):
        cases = [("fit_role", "population_validation"), ("fitted", False), ("examples_sha256", "0" * 64),
                 ("encoding", "other"), ("fixture_only", "true"), ("scale", [0.0] * len(SCALARS)),
                 ("count", [1.5] * len(SCALARS))]
        for name, value in cases:
            with self.subTest(name=name):
                artifacts = self.replace_normalizer({**self.normalizer, name: value})
                with self.assertRaises(ValueError):
                    LearnedPolicy.load(artifacts)

    def test_model_configuration_rejects_boolean_integer_aliases(self):
        config = json.loads(Path(self.artifacts.config_path).read_text())
        config["parameters"]["model"]["rank_conditioned"] = 1
        pin = save_json(self.artifacts.config_path, config, replace=True)
        with self.assertRaises(ValueError):
            LearnedPolicy.load(replace(self.artifacts, config_sha256=pin))

    def test_batch_requires_normalizer_pin_shared_inputs_and_no_future_labels(self):
        policy = LearnedPolicy.load(self.artifacts)
        with self.assertRaises(ValueError):
            policy.predict_batch(self.batch(), normalizer_sha256="0" * 64)
        for mutate in (lambda x: x.update(action=torch.zeros(2)),
                       lambda x: x.update(value=torch.zeros(2)),
                       lambda x: x.update(future_board=x["board"]),
                       lambda x: x.update(history_actions=torch.zeros((2, 8), dtype=torch.int16)),
                       lambda x: x["legal"].fill_(False),
                       lambda x: x["scalars"].__setitem__((0, 0), torch.nan)):
            with self.subTest(mutation=mutate):
                inputs = self.batch()
                mutate(inputs)
                with self.assertRaises(ValueError):
                    policy.predict_batch(inputs, normalizer_sha256=self.artifacts.normalizer_sha256)

    def test_game_boundary_rejects_forged_full_history_or_nonplaying_state(self):
        policy = LearnedPolicy.load(self.artifacts)
        played = Game.new().play("B", 0)
        forged = [replace(played, positions=(played.positions[0], replace(played.position, ko=2))),
                  replace(played, segment_start=1), replace(played, moves=()),
                  Game.new().play("B", PASS).play("W", PASS),
                  Game.from_fixture(["." * 19] * 19)]
        for game in forged:
            with self.subTest(game=game), self.assertRaises(ValueError):
                policy.predict(game)
        logits, _ = policy.predict(played)
        self.assertTrue(torch.isneginf(logits[0]))
        self.assertTrue(torch.isfinite(logits[PASS]))

    def test_match_inputs_follow_mover_rank_unknown_opponent_and_missing_clocks(self):
        artifacts, _, _ = self.make_artifacts(zero=True)
        agent = LearnedMatchFactory(artifacts, REVISION)(self.profile(artifacts).data, 123)
        seen = []
        def capture(model, args):
            self.assertFalse(model.training)
            self.assertTrue(torch.is_inference_mode_enabled())
            seen.append({key: value.clone() for key, value in args[0].items()})
        handle = agent.client._model.register_forward_pre_hook(capture)
        self.addCleanup(handle.remove)
        for game in (Game.new(), Game.new().play("B", 0)):
            decision = agent.choose(game)
            self.assertTrue(game.legal_mask()[decision.action])
            self.assertEqual((decision.network_calls, decision.gpu_seconds), (1, 0.0))
            _, value = agent.client.predict(game)
            self.assertAlmostEqual(value, 0.25)
        for index, inputs in enumerate((seen[0], seen[2])):
            self.assertEqual(set(inputs), set(INPUT_KEYS))
            self.assertEqual(inputs["rank_valid"].tolist(), [[True, False]])
            self.assertEqual(inputs["ranks"][0, 0], RANKS.index("5k") + 1)
            self.assertEqual(inputs["ranks"][0, 1], 0)
            self.assertFalse(inputs["scalar_valid"][0, 9:].any())
            self.assertTrue((inputs["scalars"][0, 9:] == 0).all())
            self.assertEqual(inputs["history_actions"].shape, (1, 0))
            self.assertEqual(float(inputs["scalars"][0, 0]), 1.0 - index)
            self.assertEqual(float(inputs["scalars"][0, 1]), -6.5 if index == 0 else 6.5)
            self.assertEqual(float(inputs["scalars"][0, 5]), -1.0 if index == 0 else -0.5)
        self.assertEqual(seen[0]["ranks"].tolist(), seen[2]["ranks"].tolist())

    def test_unconditioned_inference_is_rank_invariant_and_profile_is_unranked(self):
        artifacts, _, _ = self.make_artifacts(conditioned=False)
        policy = LearnedPolicy.load(artifacts)
        first = policy.predict_batch(self.batch(context=Context("5k", "1d")),
                                     normalizer_sha256=artifacts.normalizer_sha256)
        second = policy.predict_batch(self.batch(context=Context()),
                                      normalizer_sha256=artifacts.normalizer_sha256)
        for actual, expected in zip(first, second):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(self.profile(artifacts, None).data["human_rank"])
        with self.assertRaises(ValueError):
            self.profile(artifacts, "5k")

    def test_private_seeded_sampling_preserves_legal_pass_probability(self):
        artifacts, _, _ = self.make_artifacts(zero=True, pass_bias=math.log(PASS))
        factory = LearnedMatchFactory(artifacts, REVISION)
        first, second = factory(self.profile(artifacts).data, 123), factory(self.profile(artifacts).data, 123)
        game = Game.new().play("B", 0)
        logits, _ = first.client.predict(game)
        self.assertAlmostEqual(float(logits.double().softmax(0)[PASS]), PASS / (2 * PASS - 1), places=7)
        actions = [first.choose(game).action for _ in range(12)]
        self.assertEqual(actions, [second.choose(game).action for _ in range(12)])
        self.assertIn(PASS, actions)
        self.assertTrue(any(action < PASS for action in actions))
        self.assertTrue(all(game.legal_mask()[action] for action in actions))
        first.close()
        first.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            first.choose(game)

    def test_profile_factory_pin_and_weight_validation_and_pickling(self):
        profile, factory = self.profile(), LearnedMatchFactory(self.artifacts, REVISION)
        self.assertEqual(pickle.loads(pickle.dumps(factory)), factory)
        self.assertEqual(profile.data["family"], "learned_population")
        self.assertEqual(profile.data["history"], 0)
        self.assertEqual(profile.data["legality_history"], "full")
        self.assertIsNone(profile.data["human_rank"])
        for name, value in (("family", "other"), ("policy", "argmax"), ("human_rank", "5k"),
                            ("cpu_threads", 2), ("sampling", "argmax"), ("conditioning_rank", "100d"),
                            ("pins", {}), ("model", {}), ("source_revision", "0" * 40),
                            ("checkpoint_sha256", "0" * 64)):
            with self.subTest(name=name), self.assertRaises(InfrastructureError):
                factory({**profile.data, name: value}, 123)
        for weight in (0, -1, True, float("inf"), 1.1):
            with self.subTest(weight=weight), self.assertRaises(InfrastructureError):
                factory({**profile.data, "weight": weight}, 123)
        agent = factory({**profile.data, "weight": 0.25}, 123)
        agent.close()
        unknown = factory(self.profile(rank=None).data, 123)
        self.assertIsNone(unknown.rank)
        unknown.close()

    def run_match(self, factory, profile):
        opponent = weak_profile("opponent", source_revision=REVISION)
        slot = {"id": "synthetic", "block_id": "b0", "arm": profile.id, "profile_id": opponent.id,
                "candidate_color": "B", "candidate_seed": 10, "opponent_seed": 20}
        return run_game(slot, profile, opponent, budget=ExperimentBudget(20, 30),
            limits=MatchLimits(response_seconds=5, game_seconds=20), agent_factories={POLICY: factory})

    def test_runner_attributes_setup_failures_to_infrastructure_and_numerics_to_agent(self):
        factory = LearnedMatchFactory(self.artifacts, REVISION)
        missing = replace(factory, artifacts=replace(self.artifacts, normalizer_path=str(self.root / "absent")))
        result = self.run_match(missing, self.profile())
        self.assertEqual(result.record["result"], "U")
        self.assertEqual(result.record["fault"]["source"], "infrastructure")
        self.assertIsNone(result.record["fault"]["responsible"])
        result = self.run_match(InvalidOutputFactory(factory), self.profile())
        self.assertEqual(result.record["result"], "L")
        self.assertEqual(result.record["fault"]["source"], "agent")
        self.assertEqual(result.record["fault"]["responsible"], "B")


if __name__ == "__main__":
    unittest.main()
