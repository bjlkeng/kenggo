"""Bounded CPU raw-play pilot; the supervisor owns tracking and allocation costs."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import time

from kenggo.checkpoints import file_hash, json_hash, read_json, save_json
from kenggo.go import digest
from kenggo.katago_human import (HumanMatchFactory, HumanProfile, ReferenceBudget,
                                 human_agent_profile, POLICY as HUMAN_POLICY)
from kenggo.katago_referee import KataGoReferee, Profile
from kenggo.learned_agent import (LearnedArtifacts, LearnedMatchFactory, learned_agent_profile,
                                 POLICY as LEARNED_POLICY)
from kenggo.learner import ModelConfig
from kenggo.match_statistics import summarize
from kenggo.matches import (ExperimentBudget, InfrastructureError, MatchLimits, make_schedule,
                             read_records, run_schedule, validate_schedule, weak_profile)
from kenggo.tracking import validate_config


ROOT = Path(__file__).resolve().parents[2]
SEED = 20261114
ARMS = ("conditioned", "unconditioned", "human_sl")
RECIPE = {"pairs_per_profile": 1, "conditioning_rank": "5k",
          "opponent_policies": ["uniform", "capture_first"], "device": "cpu",
          "adjudication": "exact_first_empirical_v2", "bootstrap_seed": SEED,
          "bootstrap_resamples": 10000}
RUNNER_CAP = ExperimentBudget(3600, 4800, 512 * 1024**2)
LIMITS = MatchLimits(game_seconds=900)
OUTPUTS = ("matches", "schedule.json", "progress.json", "evidence_authorization.json",
           "scores.json", "summary.json", "failure.json")
SCOPE = ("Exploratory raw play against two unranked weak opponents, with one color pair per "
         "opponent and candidate. No full validation panel, qualification, or human-rank claim.")


def _source_identity():
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(ROOT), *arguments], timeout=10)
    if git("ls-files", "--others", "--exclude-standard").strip():
        raise ValueError("Stage intended new source files before running a pinned worker")
    return {"commit": git("rev-parse", "HEAD").decode().strip(),
            "patch_sha256": sha256(git("diff", "--binary", "HEAD")).hexdigest()}


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Every artifact and runtime root requires an explicit path")
    return str((ROOT / value).resolve())


def _bind(refs, name, path, pin):
    if name not in refs or _path(refs[name]["path"]) != _path(path) or refs[name]["sha256"] != pin:
        raise ValueError(f"Artifact differs from its independent input binding: {name}")


def _validate(config, provenance, provenance_pin, fixture_only):
    validate_config(config)
    parameters, refs = config["parameters"], config["inputs"]
    required = set(RECIPE) | {"learned_artifacts", "human_paths", "human_budget", "referee_paths",
                              "match_limits", "runner_budget"}
    if (config["purpose"] != "baseline-matches" or config["seed"] != SEED
            or not required <= parameters.keys() or parameters.keys() - required - {"max_artifact_bytes"}
            or any(parameters.get(key) != value for key, value in RECIPE.items())
            or type(parameters["pairs_per_profile"]) is not int
            or type(parameters["bootstrap_seed"]) is not int
            or type(parameters["bootstrap_resamples"]) is not int):
        raise ValueError("Raw matches require the frozen three-candidate CPU pilot recipe")
    limits, budget = MatchLimits(**parameters["match_limits"]), ExperimentBudget(**parameters["runner_budget"])
    if (set(parameters["match_limits"]) != set(asdict(LIMITS))
            or set(parameters["runner_budget"]) != set(asdict(RUNNER_CAP))
            or any(getattr(limits, key) > value for key, value in asdict(LIMITS).items())
            or limits.shutdown_seconds != LIMITS.shutdown_seconds
            or any(getattr(budget, key) > value for key, value in asdict(RUNNER_CAP).items())
            or not fixture_only and any(getattr(limits, key) != getattr(LIMITS, key) for key in
                ("exact_wall_seconds", "exact_cpu_seconds", "empirical_wall_seconds", "empirical_cpu_seconds"))):
        raise ValueError("Match and runner limits must fit the frozen pilot allowances")
    if (not isinstance(provenance, dict) or provenance.get("version") != "experiment_provenance_v1"
            or provenance.get("inputs") != refs or not isinstance(provenance.get("environment"), dict)
            or json_hash(provenance) != provenance_pin):
        raise ValueError("Canonical provenance must bind the configured inputs")
    source = provenance.get("source", {})
    if (not isinstance(source, dict) or set(source) != {"commit", "patch_sha256"}
            or not isinstance(source["commit"], str) or re.fullmatch(r"[a-f0-9]{40}", source["commit"]) is None
            or not isinstance(source["patch_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", source["patch_sha256"]) is None
            or source != _source_identity()):
        raise ValueError("Worker source differs from the pinned provenance")
    if not {"data", "split", "rules", "examples", "normalizer", "cpu_environment"} <= refs.keys():
        raise ValueError("Matches require pinned data, split, rules, examples, normalizer and CPU environment")
    for ref in refs.values():
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["sha256"], str) or re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]) is None
                or file_hash(_path(ref["path"])) != ref["sha256"]):
            raise ValueError("Every configured match input must match its artifact pin")
    for name in ("examples", "normalizer"):
        value = read_json(_path(refs[name]["path"]), refs[name]["sha256"])
        if value.get("fixture_only") is not fixture_only:
            raise ValueError("Match input scope differs from the explicit fixture scope")
        if name == "normalizer" and value.get("examples_sha256") != refs["examples"]["sha256"]:
            raise ValueError("Normalizer does not bind the pinned examples")

    bundles = parameters["learned_artifacts"]
    if not isinstance(bundles, dict) or set(bundles) != set(ARMS[:2]):
        raise ValueError("Both learned arms require independent artifact bundles")
    learned = {}
    for arm in ARMS[:2]:
        fields = dict(bundles[arm])
        model = ModelConfig(**fields["model_config"])
        if (fields["model_config"] != asdict(model) or model.rank_conditioned != (arm == "conditioned")
                or (model.blocks, model.channels) != ((1, 16) if fixture_only else (8, 128))):
            raise ValueError("Learned arms require the frozen architecture and conditioning treatment")
        fields["model_config"] = model
        artifacts = LearnedArtifacts(**fields)
        for suffix, path, pin in (("checkpoint", artifacts.checkpoint_manifest, artifacts.checkpoint_sha256),
                                  ("config", artifacts.config_path, artifacts.config_sha256)):
            _bind(refs, f"{arm}_{suffix}", path, pin)
        _bind(refs, "normalizer", artifacts.normalizer_path, artifacts.normalizer_sha256)
        if (artifacts.examples_sha256 != refs["examples"]["sha256"]
                or refs[f"{arm}_provenance"]["sha256"] != artifacts.provenance_sha256):
            raise ValueError("Learned example or provenance pin differs from its input binding")
        training = read_json(_path(artifacts.config_path), artifacts.config_sha256)
        training_provenance = read_json(_path(refs[f"{arm}_provenance"]["path"]), artifacts.provenance_sha256)
        if (training.get("purpose") != "baseline-training" or training.get("seed") != SEED
                or json_hash(training) != artifacts.config_sha256
                or training.get("parameters", {}).get("model") != asdict(model)
                or training_provenance.get("version") != "experiment_provenance_v1"
                or json_hash(training_provenance) != artifacts.provenance_sha256
                or training_provenance.get("inputs") != training.get("inputs")):
            raise ValueError("Checkpoint training configuration and provenance disagree")
        for name in ("normalizer", "examples"):
            _bind(training["inputs"], name, refs[name]["path"], refs[name]["sha256"])
        fields.update(checkpoint_manifest=_path(artifacts.checkpoint_manifest),
                      config_path=_path(artifacts.config_path), normalizer_path=_path(artifacts.normalizer_path))
        learned[arm] = LearnedArtifacts(**fields)

    human = HumanProfile.load(_path(refs["human_profile"]["path"]), refs["human_profile"]["sha256"])
    referee = Profile.load(_path(refs["referee_profile"]["path"]), refs["referee_profile"]["sha256"])
    settings = referee.data
    if (settings["schema"] != "katago_referee_profile_v2"
            or settings["cleanup_visits"] != [32, 128] or settings["pass_visits"] != [32, 128]
            or settings["cleanup_move_limit"] != 64 or settings["acceptable_gain"] != 0.5
            or settings["premature_gain"] != 1.5 or settings["no_result_limit"] != 0.01
            or settings["limits"]["cpu_seconds"] != 120 or settings["limits"]["wall_seconds"] != 180
            or settings["limits"]["nodes"] != 150000 or "EigenAVX2" not in settings["engine_version"]):
        raise ValueError("Matches require the pinned version-2 CPU empirical referee contract")
    paths = {}
    for name, profile in (("human", human), ("referee", referee)):
        declared = parameters[f"{name}_paths"]
        if not isinstance(declared, dict) or set(declared) != set(profile.data["artifacts"]) | {"runtime_root"}:
            raise ValueError("All engine artifacts and their runtime root must be explicit")
        for artifact, pin in profile.data["artifacts"].items():
            key = "human_model" if name == "human" and artifact == "human_model" else f"{name}_{artifact}"
            _bind(refs, key, declared[artifact], pin)
        paths[name] = {key: _path(value) for key, value in declared.items()}
    human_budget = ReferenceBudget(**parameters["human_budget"])
    if set(parameters["human_budget"]) != set(asdict(human_budget)):
        raise ValueError("Pin every Human SL reference allowance explicitly")
    return learned, human, human_budget, paths, limits, budget


@dataclass(frozen=True)
class LearnedFactories:
    conditioned: LearnedArtifacts
    unconditioned: LearnedArtifacts
    source_revision: str

    def __call__(self, profile, seed):
        if profile.get("id") not in ARMS[:2]:
            raise InfrastructureError("Unknown learned pilot arm")
        return LearnedMatchFactory(getattr(self, profile["id"]), self.source_revision)(profile, seed)


@dataclass(frozen=True)
class RefereeFactory:
    profile_path: str
    profile_sha256: str
    engine: str
    model: str
    config: str
    native: str
    runtime_root: str

    def __call__(self):
        profile = Profile.load(self.profile_path, self.profile_sha256)
        return KataGoReferee(profile, engine=self.engine, model=self.model, config=self.config,
                             native=self.native, runtime_root=self.runtime_root)


def _components(config, provenance, validated):
    learned, human, human_budget, paths, limits, budget = validated
    revision = provenance["source"]["commit"]
    candidates = tuple(learned_agent_profile(arm, learned[arm], "5k" if arm == "conditioned" else None,
                        source_revision=revision, panel="candidate") for arm in ARMS[:2])
    candidates += (human_agent_profile("human_sl", human, "5k", source_revision=revision,
                                      budget=human_budget, panel="candidate"),)
    ref = config["inputs"]["referee_profile"]
    referee_factory = RefereeFactory(_path(ref["path"]), ref["sha256"], **paths["referee"])
    factories = {LEARNED_POLICY: LearnedFactories(**learned, source_revision=revision),
                 HUMAN_POLICY: HumanMatchFactory(human, **paths["human"], budget=human_budget,
                                                 source_revision=revision)}
    schedule = build_schedule(candidates, provenance=provenance, limits=limits,
                              referee_profile={"sha256": ref["sha256"], "device": "cpu"})
    return schedule, factories, referee_factory, budget


def build_schedule(candidates, *, provenance, limits=LIMITS, referee_profile):
    """Freeze paired seeds and alternate both candidate and color execution order."""
    if tuple(candidate.id for candidate in candidates) != ARMS:
        raise ValueError("The raw-play pilot requires its three candidates in the declared order")
    opponents = tuple(weak_profile(f"weak_{policy}", policy, source_revision=provenance["source"]["commit"])
                      for policy in RECIPE["opponent_policies"])
    schedule = make_schedule(arms=candidates, profiles=opponents, pairs=1, seed=SEED,
        purpose="validation", scope="setup", control_arm="unconditioned", limits=limits,
        referee_profile=referee_profile, provenance=provenance)
    ordered = []
    for index, opponent in enumerate(opponents):
        colors = ("B", "W") if index == 0 else ("W", "B")
        for color_index, color in enumerate(colors):
            arms = ARMS if (index + color_index) % 2 == 0 else ARMS[::-1]
            ordered.extend(next(slot for slot in schedule["slots"] if slot["profile_id"] == opponent.id
                                and slot["candidate_color"] == color and slot["arm"] == arm) for arm in arms)
    schedule["slots"] = [dict(slot, id=f"s{index:08d}") for index, slot in enumerate(ordered)]
    validate_schedule(schedule, for_execution=True)
    return schedule


def _generated_records(directory, schedule):
    slots = {slot["id"]: slot for slot in schedule["slots"]}
    records = {}
    for path in sorted(directory.glob("s*.json")):
        if path.name == "schedule.json":
            continue
        record = json.loads(path.read_text())
        if (path.stem not in slots or record.get("slot") != slots[path.stem]
                or record.get("sgf_file") != path.with_suffix(".sgf").name):
            raise ValueError("Generated match record differs from the frozen schedule")
        records[path.stem] = record
    return records


def authorize_evidence(output, schedule, records, *, config_pin, provenance_pin):
    """Authorize only evidence just generated in this owned, pinned run directory."""
    output = Path(output)
    if json.loads((output / "matches/schedule.json").read_text()) != schedule:
        raise ValueError("Generated schedule differs from the pre-run schedule")
    entries, files = [], {}
    for slot in schedule["slots"]:
        if slot["id"] not in records:
            continue
        record = records[slot["id"]]
        path = output / "matches" / f"{slot['id']}.json"
        sgf = output / "matches" / f"{slot['id']}.sgf"
        if (record["slot"] != slot or json.loads(path.read_text()) != record
                or record["referee_profile"] != schedule["referee_profile"]
                or file_hash(sgf) != record["sgf_sha256"]):
            raise ValueError("Generated record or SGF differs from the run's retained evidence")
        files[slot["id"]] = {"record_sha256": file_hash(path), "sgf_sha256": file_hash(sgf)}
        attempts = record["adjudication_attempts"] + [attempt for passed in record["candidate_passes"]
                                                       for attempt in passed["attempts"]]
        for attempt in attempts:
            if attempt["method"] not in ("end_empirical", "pass_empirical") or "evidence" not in attempt:
                continue
            evidence = attempt["evidence"]
            if (not isinstance(evidence, dict) or evidence.get("snapshot") != attempt["snapshot"]
                    or evidence.get("profile_sha256") != schedule["referee_profile"]["sha256"]):
                raise ValueError("Generated empirical evidence differs from its snapshot or referee")
            entries.append({"slot_id": slot["id"], "method": attempt["method"],
                            "snapshot": attempt["snapshot"], "sha256": digest(evidence)})
    manifest = {"version": "baseline_match_evidence_authorization_v1", "config_file_sha256": config_pin,
                "provenance_sha256": provenance_pin, "source": schedule["provenance"]["source"],
                "schedule_sha256": digest(schedule), "schedule_file_sha256": file_hash(output / "matches/schedule.json"),
                "referee_profile": schedule["referee_profile"], "records": files, "evidence": entries,
                "scope": "Evidence generated by this pinned run; empirical outcomes still require native replay."}
    pin = save_json(output / "evidence_authorization.json", manifest)
    authorized = read_json(output / "evidence_authorization.json", pin)
    if authorized != manifest:
        raise ValueError("Saved replay authorization differs from generated evidence")
    return pin, frozenset(entry["sha256"] for entry in authorized["evidence"])


def _quantiles(values, probabilities):
    ordered = sorted(values)
    return [ordered[math.ceil(len(ordered) * probability) - 1] if ordered else None for probability in probabilities]


def behavior_summary(schedule, records):
    result = {}
    for arm in ARMS:
        slots = [slot for slot in schedule["slots"] if slot["arm"] == arm]
        observed = [records[slot["id"]] for slot in slots if slot["id"] in records]
        flags = {}
        for flag in ("runtime_failure", "cap", "unresolved", "dispute", "premature_pass", "length_tail"):
            values = [record["behavior"][flag] for record in observed]
            unknown = sum(value is None for value in values) + len(slots) - len(observed)
            flags[flag] = {"true": sum(value is True for value in values),
                           "false": sum(value is False for value in values), "unknown_or_missing": unknown,
                           "adverse": sum(value is True for value in values) + unknown, "scheduled": len(slots)}
        passes = [passed for record in observed for passed in record["candidate_passes"]]
        pass_counts = {verdict: sum(passed["verdict"] == verdict for passed in passes)
                       for verdict in ("acceptable", "premature", "unjudgeable")}
        candidate_decisions = [decision for record in observed for decision in record["decisions"]
                               if decision["color"] == record["slot"]["candidate_color"]]
        latency = [decision["cost"]["wall_seconds"] for decision in candidate_decisions]
        calls = [decision["network_calls"] for decision in candidate_decisions]
        lengths = [{"slot_id": record["slot"]["id"], "actions": record["actual_actions"],
                    "censored": record["censored"]} for record in observed]
        result[arm] = {"scheduled": len(slots), "recorded": len(observed), "flags": flags,
            "passes": {**pass_counts, "total": len(passes),
                "premature_rate": pass_counts["premature"] / len(passes) if passes else None,
                "unjudgeable_rate": pass_counts["unjudgeable"] / len(passes) if passes else None},
            "lengths": lengths, "length_quantiles": {"probabilities": [.5, .9, .99], "method": "nearest_rank",
                "true_terminations": _quantiles([item["actions"] for item in lengths if not item["censored"]], [.5, .9, .99]),
                "all_observed": _quantiles([item["actions"] for item in lengths], [.5, .9, .99])},
            "latency": {"decisions": len(latency), "method": "nearest_rank",
                        "median_p95_seconds": _quantiles(latency, [.5, .95])},
            "candidate_network_calls": {"known": sum(call for call in calls if call is not None),
                                        "unknown_decisions": sum(call is None for call in calls)},
            "referee_network_calls": {"known": sum(record["cost"]["referee_network_calls"] or 0 for record in observed),
                "unknown_games": sum(record["cost"]["referee_network_calls"] is None for record in observed)},
            "disagreements": [{"slot_id": record["slot"]["id"], **record["disagreement"]} for record in observed],
            "sgfs": [f"matches/{record['sgf_file']}" for record in observed]}
    return result


def _cpu_usage():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def run_matches(config_path, config_pin, provenance_path, provenance_pin, output, *, fixture_only=False):
    """Execute once, authorize newly generated evidence, and independently replay it."""
    if type(fixture_only) is not bool:
        raise ValueError("Fixture scope must be explicit")
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("The supervisor must create the job output directory")
    output = output.resolve()
    if any(os.path.lexists(output / name) for name in OUTPUTS):
        raise ValueError("A match worker never resets or retries an existing output")
    started, cpu_started = time.monotonic(), _cpu_usage()
    schedule, generated, replayed, runner_cost = None, {}, {}, None
    outside_before, after_runner, authorization_pin = 0.0, None, None
    state = {"version": "baseline_match_progress_v1", "status": "preparing", "fixture_only": fixture_only,
             "config_file_sha256": config_pin, "provenance_sha256": provenance_pin,
             "scheduled_games": 12, "recorded_games": 0, "records_replayed": 0,
             "replay_completed": False, "qualification_claim": False, "gpu_seconds": 0}

    def costs():
        known = _cpu_usage() - cpu_started
        charged = (outside_before + runner_cost["cpu_seconds"] + _cpu_usage() - after_runner
                   if runner_cost is not None and after_runner is not None else known)
        return {"cpu_seconds": max(known, charged), "wall_seconds": time.monotonic() - started,
                "gpu_seconds": 0, "cpu_scope": "Worker and reaped children, including native replay; conservative runner charge retained."}

    def persist():
        state["costs"] = costs()
        save_json(output / "progress.json", state, replace=True)

    def finish(status):
        state.update(status=status, recorded_games=len(generated), records_replayed=len(replayed))
        result = dict(state, version="baseline_matches_v1", scope=SCOPE,
                      human_rank_claim=False, full_validation_panel=False,
                      evidence_authorization_sha256=authorization_pin, runner_cost=runner_cost)
        if schedule is not None:
            scores = summarize(schedule, replayed, bootstrap_seed=SEED, resamples=10000)
            save_json(output / "scores.json", scores)
            result.update(counts={arm: scores["arms"][arm]["counts"] for arm in ARMS},
                paired_conditioned_unconditioned=scores["comparisons"]["conditioned"],
                exploratory=scores["exploratory"], behavior=behavior_summary(schedule, replayed),
                missing_slots=[slot["id"] for slot in schedule["slots"] if slot["id"] not in generated],
                unverified_slots=sorted(set(generated) - set(replayed)),
                sgfs=[{"slot_id": slot_id, "path": f"matches/{record['sgf_file']}",
                       "sha256": record["sgf_sha256"], "replayed": slot_id in replayed}
                      for slot_id, record in generated.items()],
                schedule_sha256=digest(schedule), schedule_file_sha256=file_hash(output / "schedule.json"),
                scores_sha256=file_hash(output / "scores.json"),
                source=schedule["provenance"]["source"], input_pins=schedule["provenance"]["inputs"])
        result["costs"] = costs()
        persist()
        save_json(output / "summary.json", result)
        return result

    persist()
    try:
        config, provenance = read_json(config_path, config_pin), read_json(provenance_path, provenance_pin)
        validated = _validate(config, provenance, provenance_pin, fixture_only)
        schedule, factories, referee_factory, budget = _components(config, provenance, validated)
        state.update(config_sha256=json_hash(config), source=provenance["source"], seed=SEED,
                     parameters=config["parameters"], status="playing")
        save_json(output / "schedule.json", schedule)
        persist()
        outside_before = _cpu_usage() - cpu_started
        runner_error = None
        try:
            generated = run_schedule(schedule, output / "matches", budget=budget,
                                     agent_factories=factories, referee_factory=referee_factory)
        except Exception as error:
            runner_error = error
            generated = _generated_records(output / "matches", schedule)
        after_runner = _cpu_usage()
        if (output / "matches/run.json").is_file():
            runner_cost = json.loads((output / "matches/run.json").read_text())
        state.update(status="authorizing_replay", recorded_games=len(generated))
        persist()
        authorization_pin, authorized = authorize_evidence(output, schedule, generated,
                                                          config_pin=config_pin, provenance_pin=provenance_pin)
        state["status"] = "replaying"
        persist()
        replay_started, replay_cpu = time.monotonic(), _cpu_usage()
        referee = None
        try:
            referee = referee_factory() if authorized else None
            saved_schedule, saved_records = read_records(output / "matches", authorized_hashes=authorized,
                                                        empirical_referee=referee)
            if saved_schedule != schedule or saved_records != generated:
                raise ValueError("Independently replayed records differ from the generated matches")
            replayed = saved_records
            state["replay_completed"] = True
        finally:
            if referee is not None:
                referee.close()
            state["replay_costs"] = {"cpu_seconds": _cpu_usage() - replay_cpu,
                                     "wall_seconds": time.monotonic() - replay_started, "gpu_seconds": 0}
        if provenance["source"] != _source_identity():
            raise ValueError("Worker source changed during matches or replay")
        if runner_error is not None:
            raise runner_error
        if costs()["cpu_seconds"] > 4200 or costs()["wall_seconds"] > 5400:
            raise RuntimeError("The match pilot exceeded its outer CPU or wall allowance")
        return finish("completed" if len(replayed) == len(schedule["slots"]) else "incomplete")
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, reason=str(error),
                     recorded_games=len(generated), records_replayed=len(replayed))
        persist()
        save_json(output / "failure.json", dict(state, scope=SCOPE))
        if not (output / "summary.json").exists():
            finish("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--provenance-sha256", required=True)
    parser.add_argument("--fixture-only", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run_matches(arguments.config, arguments.config_sha256, arguments.provenance,
        arguments.provenance_sha256, arguments.output, fixture_only=arguments.fixture_only), indent=2))
