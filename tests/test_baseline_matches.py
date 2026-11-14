import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kenggo import baseline_matches as pilot
from kenggo.adjudication import Limits, adjudicate, adjudicate_empirical, judge_pass
from kenggo.checkpoints import file_hash, json_hash, save_json
from kenggo.go import PASS, canonical, digest
from kenggo.katago_human import ReferenceBudget
from kenggo.katago_referee import PassDecision, RefereeDecision
from kenggo.learner import ModelConfig
from kenggo.matches import Decision, ExperimentBudget, InfrastructureError, MatchLimits, _Failure


SOURCE = {"commit": "a" * 40, "patch_sha256": "b" * 64}


class SyntheticAgent:
    def choose(self, game):
        return Decision(0 if not game.moves else PASS, 1, 0.0)

    def close(self):
        pass


def synthetic_factory(profile, seed):
    return SyntheticAgent()


class SyntheticReferee:
    def __init__(self, pin):
        self.profile = SimpleNamespace(sha256=pin)
        self.verifications = 0

    def adjudicate(self, game):
        reason = "winner_stable_empirical_cleanup"
        evidence = canonical({"snapshot": game.snapshot, "original": game.action_record(),
            "profile_sha256": self.profile.sha256, "status": "resolved", "reason": reason,
            "white_minus_black_halfpoints": 13, "calls": [{"kind": "analysis"}] * 2,
            "runs": [{"terminal": {"white_minus_black_halfpoints": value}} for value in (11, 13)]})
        return RefereeDecision("resolved", reason, game.snapshot, 13, evidence)

    def judge_pass(self, game):
        evidence = canonical({"snapshot": game.snapshot, "profile_sha256": self.profile.sha256,
                              "status": "acceptable", "reason": "fixture_acceptable"})
        return PassDecision("acceptable", "fixture_acceptable", game.snapshot, (0.0, 0.0), evidence)

    def verify(self, game, evidence, *, authorized_hashes=frozenset()):
        expected = self.adjudicate(game)
        if digest(json.loads(evidence)) not in authorized_hashes or evidence != expected.evidence:
            raise ValueError("Synthetic replay lacks authorization or has changed evidence")
        self.verifications += 1
        return expected

    def close(self):
        pass


class InlineWorker:
    """Exercise runner records with synthetic clients and no spawned engines."""

    def __init__(self, role, factory, profile, seed, meter):
        self.role, self.factory, self.profile, self.seed = role, factory, profile, seed
        self.obj = None

    def initialize(self, *args, **kwargs):
        try:
            if self.role == "agent":
                self.obj = self.factory(self.profile, self.seed)
            elif self.role == "empirical":
                self.obj = self.factory()
        except InfrastructureError as error:
            return _Failure(str(error), False)
        return None

    def call(self, operation, game, limits, *args, **kwargs):
        exact = Limits(cpu_seconds=limits.exact_cpu_seconds, wall_seconds=limits.exact_wall_seconds)
        if operation == "choose":
            result = self.obj.choose(game)
        elif operation == "end_exact":
            result = adjudicate(game, limits=exact)
        elif operation == "pass_exact":
            result = judge_pass(game, limits=exact)
        elif operation == "end_empirical":
            result = adjudicate_empirical(game, self.obj)
        else:
            result = self.obj.judge_pass(game)
        return result, {"cpu_seconds": 0.0, "wall_seconds": 0.001}

    def close(self):
        if self.obj is not None:
            self.obj.close()


class BaselineMatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.refs = {}
        for name in ("data", "split", "rules", "cpu_environment", "human_engine", "human_main_model",
                     "human_model", "human_config", "referee_engine", "referee_model", "referee_config",
                     "referee_native"):
            path = self.root / name
            path.write_bytes(("synthetic " + name).encode())
            self.refs[name] = {"path": str(path), "sha256": file_hash(path)}
        self.save_ref("examples", {"fixture_only": True})
        self.save_ref("normalizer", {"fixture_only": True, "examples_sha256": self.refs["examples"]["sha256"]})
        self.artifacts = {}
        for arm in pilot.ARMS[:2]:
            model = ModelConfig(1, 16, arm == "conditioned")
            training = {"purpose": "baseline-training", "seed": pilot.SEED,
                        "parameters": {"model": asdict(model)},
                        "inputs": {name: self.refs[name] for name in ("data", "split", "rules", "normalizer", "examples")}}
            self.save_ref(arm + "_config", training)
            self.save_ref(arm + "_provenance", {"version": "experiment_provenance_v1", "inputs": training["inputs"]})
            self.save_ref(arm + "_checkpoint", {"synthetic": arm})
            self.artifacts[arm] = {"checkpoint_manifest": self.refs[arm + "_checkpoint"]["path"],
                "checkpoint_sha256": self.refs[arm + "_checkpoint"]["sha256"],
                "config_path": self.refs[arm + "_config"]["path"],
                "config_sha256": self.refs[arm + "_config"]["sha256"],
                "provenance_sha256": self.refs[arm + "_provenance"]["sha256"],
                "normalizer_path": self.refs["normalizer"]["path"], "normalizer_sha256": self.refs["normalizer"]["sha256"],
                "examples_sha256": self.refs["examples"]["sha256"], "model_config": asdict(model)}
        self.paths = {}
        for role, name in (("human", "katago_human_profile_v1.json"), ("referee", "e1_katago_referee_profile_v2.json")):
            profile = json.loads((pilot.ROOT / "manifests" / name).read_text())
            paths = {}
            for artifact in profile["artifacts"]:
                key = "human_model" if role == "human" and artifact == "human_model" else f"{role}_{artifact}"
                profile["artifacts"][artifact] = self.refs[key]["sha256"]
                paths[artifact] = self.refs[key]["path"]
            profile.update(runtime_files={}, library_dirs=[])
            self.save_ref(role + "_profile", profile)
            self.paths[role] = dict(paths, runtime_root=str(self.root))
        self.config = {"version": "experiment_config_v1", "entity": "fixture", "project": "fixture",
            "base_url": "https://api.wandb.ai", "comparison": "fixture", "arm": "raw-matches",
            "purpose": "baseline-matches", "seed": pilot.SEED, "budget_id": "fixture", "inputs": self.refs,
            "parameters": {**copy.deepcopy(pilot.RECIPE), "learned_artifacts": self.artifacts,
                "human_paths": self.paths["human"], "referee_paths": self.paths["referee"],
                "human_budget": asdict(ReferenceBudget(401, 30, 30)),
                "match_limits": asdict(MatchLimits(response_seconds=.05, game_seconds=30,
                    exact_cpu_seconds=.005, exact_wall_seconds=.05, empirical_cpu_seconds=1, empirical_wall_seconds=2)),
                "runner_budget": asdict(ExperimentBudget(30, 60, 16 * 1024**2))}}
        source = patch.object(pilot, "_source_identity", return_value=SOURCE)
        source.start()
        self.addCleanup(source.stop)

    def save_ref(self, name, value):
        path = self.root / (name + ".json")
        pin = save_json(path, value, replace=path.exists())
        self.refs[name] = {"path": str(path), "sha256": pin}
        return pin

    def provenance(self, config=None):
        return {"version": "experiment_provenance_v1", "source": SOURCE, "environment": {},
                "inputs": (config or self.config)["inputs"]}

    def validate(self, config=None):
        config = config or self.config
        provenance = self.provenance(config)
        return pilot._validate(config, provenance, json_hash(provenance), True)

    def components(self):
        return pilot._components(self.config, self.provenance(), self.validate())

    def invoke(self, output_name="output"):
        output = self.root / output_name
        output.mkdir(exist_ok=True)
        config_path, provenance_path = self.root / "config.json", self.root / "provenance.json"
        config_pin = save_json(config_path, self.config, replace=config_path.exists())
        provenance_pin = save_json(provenance_path, self.provenance(), replace=provenance_path.exists())
        result = pilot.run_matches(config_path, config_pin, provenance_path, provenance_pin, output, fixture_only=True)
        return result, output

    def synthetic_run(self, **kwargs):
        referees = []
        def referee(factory):
            client = SyntheticReferee(factory.profile_sha256)
            referees.append(client)
            return client
        with patch("kenggo.matches._Worker", InlineWorker), \
             patch("kenggo.matches._resolve_factory", return_value=synthetic_factory), \
             patch.object(pilot.RefereeFactory, "__call__", referee), \
             patch("torch.cuda.init", side_effect=AssertionError("CPU worker touched CUDA")), \
             patch("torch.cuda.is_available", side_effect=AssertionError("CPU worker probed CUDA")), \
             patch("kenggo.learned_agent.LearnedPolicy.load", side_effect=AssertionError("Fixture loaded model")):
            result, output = self.invoke(**kwargs)
        return result, output, referees

    def test_twelve_slots_share_block_seeds_and_alternate_colors_and_arms(self):
        schedule, factories, referee, budget = self.components()
        self.assertEqual(schedule, self.components()[0])
        self.assertEqual(len(schedule["slots"]), 12)
        self.assertEqual((schedule["seed"], schedule["scope"], schedule["control_arm"]),
                         (pilot.SEED, "setup", "unconditioned"))
        slots = schedule["slots"]
        self.assertEqual([slot["arm"] for slot in slots[:6]], list(pilot.ARMS) + list(reversed(pilot.ARMS)))
        self.assertEqual([slot["arm"] for slot in slots[6:]], list(reversed(pilot.ARMS)) + list(pilot.ARMS))
        self.assertEqual([slot["candidate_color"] for slot in slots], ["B"] * 3 + ["W"] * 6 + ["B"] * 3)
        seeds = []
        for opponent in schedule["profiles"]:
            group = [slot for slot in slots if slot["profile_id"] == opponent["id"]]
            pairs = {(slot["candidate_seed"], slot["opponent_seed"]) for slot in group}
            self.assertEqual(len(pairs), 1)
            seeds.extend(pairs.pop())
            self.assertIsNone(opponent["human_rank"])
        self.assertEqual(len(set(seeds)), 4)
        profiles = dict(zip(pilot.ARMS, schedule["arm_profiles"]))
        self.assertEqual([profiles[arm]["conditioning_rank"] for arm in pilot.ARMS], ["5k", None, "5k"])
        for profile in profiles.values():
            self.assertIsNone(profile["human_rank"])
            self.assertEqual(profile["resignation"], "never")
            self.assertEqual(profile["device"], "cpu")
            self.assertEqual(profile["panel"], "candidate")
            self.assertIn(profile["pass"], ("raw_policy", "raw_humanPolicy"))
        self.assertEqual(pickle.loads(pickle.dumps(factories)), factories)
        self.assertEqual(pickle.loads(pickle.dumps(referee)), referee)

    def test_referee_factory_uses_only_its_explicit_pinned_paths(self):
        _, _, factory, _ = self.components()
        with patch.object(pilot, "KataGoReferee") as constructor:
            factory()
        args, kwargs = constructor.call_args
        self.assertEqual(args[0].sha256, self.refs["referee_profile"]["sha256"])
        self.assertEqual(kwargs, self.paths["referee"])

    def test_learned_dispatch_preserves_each_arm_and_setup_failure(self):
        schedule, factories, _, _ = self.components()
        factory = factories[pilot.LEARNED_POLICY]
        for profile in schedule["arm_profiles"][:2]:
            with patch.object(pilot, "LearnedMatchFactory") as constructor:
                constructor.return_value.side_effect = InfrastructureError("missing checkpoint")
                with self.assertRaisesRegex(InfrastructureError, "missing checkpoint"):
                    factory(profile, 13)
            self.assertEqual(constructor.call_args.args[0], getattr(factory, profile["id"]))
            self.assertEqual(constructor.call_args.args[1], SOURCE["commit"])
            constructor.return_value.assert_called_once_with(profile, 13)
        with self.assertRaises(InfrastructureError):
            factory({"id": "unknown"}, 13)

    def test_seed_rank_device_and_limits_cannot_change(self):
        cases = [("device", "cuda"), ("conditioning_rank", "1d"), ("pairs_per_profile", 2),
                 ("bootstrap_seed", 1), ("bootstrap_resamples", 20000),
                 ("adjudication", "exact_only"), ("opponent_policies", ["capture_first", "uniform"])]
        for key, value in cases:
            config = copy.deepcopy(self.config)
            config["parameters"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(config)
        for key, value in (("seed", 1), ("purpose", "final")):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(config)
        config = copy.deepcopy(self.config)
        config["parameters"]["runner_budget"]["cpu_seconds"] = 3601
        with self.assertRaises(ValueError):
            self.validate(config)
        config = copy.deepcopy(self.config)
        config["parameters"]["match_limits"]["game_seconds"] = 901
        with self.assertRaises(ValueError):
            self.validate(config)

    def test_artifact_bundles_cannot_disagree_with_input_pins(self):
        for key in ("checkpoint_sha256", "config_sha256", "provenance_sha256", "examples_sha256", "normalizer_sha256"):
            config = copy.deepcopy(self.config)
            config["parameters"]["learned_artifacts"]["conditioned"][key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(config)
        for role in ("human", "referee"):
            config = copy.deepcopy(self.config)
            config["parameters"][role + "_paths"]["config"] = self.refs["rules"]["path"]
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.validate(config)
        config = copy.deepcopy(self.config)
        config["parameters"]["learned_artifacts"]["unconditioned"]["model_config"]["rank_conditioned"] = True
        with self.assertRaises(ValueError):
            self.validate(config)

    def test_source_and_provenance_and_fixture_scope_must_match(self):
        provenance = self.provenance()
        provenance["inputs"] = {}
        with self.assertRaises(ValueError):
            pilot._validate(self.config, provenance, json_hash(provenance), True)
        with patch.object(pilot, "_source_identity", return_value={**SOURCE, "commit": "c" * 40}):
            with self.assertRaises(ValueError):
                self.validate()
        self.save_ref("examples", {"fixture_only": False})
        with self.assertRaisesRegex(ValueError, "fixture scope"):
            self.validate()

    def test_saved_synthetic_games_authorize_passes_and_replay_both_states(self):
        result, output, referees = self.synthetic_run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual((result["scheduled_games"], result["records_replayed"]), (12, 12))
        self.assertTrue(result["replay_completed"])
        self.assertFalse(result["qualification_claim"])
        self.assertTrue(result["exploratory"])
        self.assertEqual(sum(ref.verifications for ref in referees), 24)
        for arm, counts in result["counts"].items():
            self.assertEqual(counts, {"N": 4, "W": 2, "L": 2, "D": 0, "U": 0, "missing": 0})
            behavior = result["behavior"][arm]
            self.assertEqual(behavior["passes"]["acceptable"], 4)
            self.assertEqual(behavior["flags"]["dispute"]["unknown_or_missing"], 4)
            self.assertEqual(behavior["flags"]["dispute"]["adverse"], 4)
            self.assertEqual(len(behavior["sgfs"]), 4)
        manifest = json.loads((output / "evidence_authorization.json").read_text())
        self.assertEqual(file_hash(output / "evidence_authorization.json"), result["evidence_authorization_sha256"])
        self.assertEqual(len(manifest["records"]), 12)
        self.assertEqual(len(manifest["evidence"]), 24)
        self.assertEqual({item["method"] for item in manifest["evidence"]}, {"end_empirical", "pass_empirical"})
        self.assertEqual(manifest["config_file_sha256"], result["config_file_sha256"])
        self.assertEqual(manifest["provenance_sha256"], result["provenance_sha256"])
        self.assertEqual(manifest["schedule_sha256"], result["schedule_sha256"])
        self.assertEqual(result["costs"]["gpu_seconds"], 0)
        self.assertEqual(len(list((output / "matches").glob("*.sgf"))), 12)
        self.assertEqual(len(result["sgfs"]), 12)
        self.assertTrue(all(entry["replayed"] for entry in result["sgfs"]))
        with self.assertRaisesRegex(ValueError, "never resets"):
            self.invoke()

    def test_exhausted_budget_retains_all_missing_slots_as_u(self):
        self.config["parameters"]["runner_budget"] = asdict(ExperimentBudget(1, .1, 1024**2))
        with patch.object(pilot.RefereeFactory, "__call__", side_effect=AssertionError("No game should start")):
            result, output = self.invoke()
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["replay_completed"])
        self.assertEqual(result["records_replayed"], 0)
        self.assertEqual(len(result["missing_slots"]), 12)
        self.assertEqual(len(json.loads((output / "schedule.json").read_text())["slots"]), 12)
        for arm, counts in result["counts"].items():
            self.assertEqual(counts, {"N": 4, "W": 0, "L": 0, "D": 0, "U": 4, "missing": 4})
            self.assertIsNone(result["behavior"][arm]["passes"]["premature_rate"])

    def test_replay_failure_retains_outputs_and_does_not_report_verified_scores(self):
        with patch.object(pilot, "read_records", side_effect=ValueError("native revalidation failed")):
            with self.assertRaisesRegex(ValueError, "native revalidation"):
                self.synthetic_run()
        output = self.root / "output"
        result = json.loads((output / "summary.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["replay_completed"])
        self.assertEqual(result["records_replayed"], 0)
        self.assertEqual(len(result["unverified_slots"]), 12)
        self.assertTrue((output / "failure.json").is_file())
        self.assertTrue((output / "evidence_authorization.json").is_file())
        self.assertEqual(len(list((output / "matches").glob("*.sgf"))), 12)
        self.assertTrue(all(not entry["replayed"] for entry in result["sgfs"]))
        self.assertTrue(all(counts["U"] == 4 for counts in result["counts"].values()))

    def test_runner_error_keeps_completed_records_without_a_rerun(self):
        original = pilot.run_schedule
        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("synthetic failure after writing records")
        with patch.object(pilot, "run_schedule", side_effect=interrupted) as runner:
            with self.assertRaisesRegex(RuntimeError, "after writing records"):
                self.synthetic_run()
        self.assertEqual(runner.call_count, 1)
        result = json.loads((self.root / "output/summary.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["replay_completed"])
        self.assertEqual(result["records_replayed"], 12)
        self.assertEqual(result["unverified_slots"], [])
        failure = json.loads((self.root / "output/failure.json").read_text())
        self.assertEqual(failure["records_replayed"], 12)

    def test_changed_generated_evidence_cannot_be_authorized(self):
        _, output, _ = self.synthetic_run()
        (output / "evidence_authorization.json").unlink()
        schedule = json.loads((output / "schedule.json").read_text())
        generated = pilot._generated_records(output / "matches", schedule)
        first = generated[schedule["slots"][0]["id"]]
        first["adjudication_attempts"][-1]["evidence"]["profile_sha256"] = "0" * 64
        save_json(output / "matches" / (first["slot"]["id"] + ".json"), first, replace=True)
        with self.assertRaisesRegex(ValueError, "snapshot or referee"):
            pilot.authorize_evidence(output, schedule, generated, config_pin="c" * 64, provenance_pin="d" * 64)


if __name__ == "__main__":
    unittest.main()
