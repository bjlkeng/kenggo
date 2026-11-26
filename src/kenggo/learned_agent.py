"""Pinned CPU population-policy inference and raw match sampling."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import random
import re

import torch

from kenggo.checkpoints import VERSION as CHECKPOINT_VERSION, file_hash, json_hash, read_json
from kenggo.continuation_data import RECIPE as CONTINUATION_RECIPE, validate_lineage
from kenggo.policy_recipe import RECIPE as POLICY_RECIPE
from kenggo.encoding import Context, INPUT_KEYS, SCALARS, apply_normalizer, encode
from kenggo.go import Game, PASS, RULES
from kenggo.kgs_source import RANKS
from kenggo.learner import ModelConfig, PopulationModel
from kenggo.rules_replay import dump_game, load_game


POLICY = "learned_population_raw_v1"
CPU_THREADS = 1
EXPANDED_RECIPE = "expanded_e2_training_v1"
KOMI_INFERENCE_CONTRACT = "komi_comparison_v1"


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class LearnedArtifacts:
    checkpoint_manifest: str
    checkpoint_sha256: str
    config_path: str
    config_sha256: str
    provenance_sha256: str
    normalizer_path: str
    normalizer_sha256: str
    examples_sha256: str
    model_config: ModelConfig = ModelConfig()

    def __post_init__(self):
        if not isinstance(self.model_config, ModelConfig):
            raise ValueError("Inference requires an explicit model configuration")
        for name in ("checkpoint_sha256", "config_sha256", "provenance_sha256",
                     "normalizer_sha256", "examples_sha256"):
            if not _hash(getattr(self, name)):
                raise ValueError(f"Inference requires an independent SHA-256 pin: {name}")
        for name in ("checkpoint_manifest", "config_path", "normalizer_path"):
            if not isinstance(getattr(self, name), (str, Path)) or not str(getattr(self, name)):
                raise ValueError(f"Inference requires an artifact path: {name}")


def _normalizer(artifacts):
    config = read_json(artifacts.config_path, artifacts.config_sha256)
    if (config["version"] != "experiment_config_v1"
            or ModelConfig(**config["parameters"]["model"]) != artifacts.model_config
            or config["inputs"]["normalizer"]["sha256"] != artifacts.normalizer_sha256
            or config["inputs"]["examples"]["sha256"] != artifacts.examples_sha256):
        raise ValueError("Inference model and inputs differ from the pinned training configuration")
    normalizer = read_json(artifacts.normalizer_path, artifacts.normalizer_sha256)
    continuation = config["parameters"].get("recipe") in (CONTINUATION_RECIPE, POLICY_RECIPE)
    if continuation:
        _, fitted = validate_lineage(config["inputs"], fixture_only=normalizer["fixture_only"])
        if fitted != normalizer:
            raise ValueError("Continuation normalizer differs from its actual fitting lineage")
    if (not continuation and normalizer["examples_sha256"] != artifacts.examples_sha256
            or type(normalizer["fixture_only"]) is not bool):
        raise ValueError("Normalizer does not bind the expected training examples")
    apply_normalizer(encode(Game.new()), normalizer)
    parameters = config["parameters"]
    training_komi = (13,)
    inference_komi = (13,)
    if parameters.get("recipe") in (EXPANDED_RECIPE, CONTINUATION_RECIPE, POLICY_RECIPE):
        values = parameters.get("training_komi_halfpoints")
        profile = config["inputs"].get("profile", {})
        if (parameters.get("inference_contract") != KOMI_INFERENCE_CONTRACT
                or not isinstance(values, list) or any(type(value) is not int for value in values)
                or values not in ([13], [1, 13])
                or not isinstance(profile, dict) or set(profile) != {"path", "sha256"}
                or not isinstance(profile["path"], str) or not profile["path"]
                or not _hash(profile["sha256"])):
            raise ValueError("Expanded inference requires the pinned komi comparison training contract")
        training_komi, inference_komi = tuple(values), (1, 13)
    elif "inference_contract" in parameters or "training_komi_halfpoints" in parameters:
        raise ValueError("Komi comparison inference requires the expanded training recipe")
    return normalizer, training_komi, inference_komi


def _weights(artifacts, model):
    path = Path(artifacts.checkpoint_manifest)
    manifest = read_json(path, artifacts.checkpoint_sha256)
    fields = {"version", "file", "sha256", "step", "config_sha256", "provenance_sha256"}
    if (not isinstance(manifest, dict) or set(manifest) != fields
            or manifest["version"] != CHECKPOINT_VERSION or manifest["file"] != "state.pt"
            or not _hash(manifest["sha256"]) or type(manifest["step"]) is not int
            or manifest["step"] < 0 or manifest["config_sha256"] != artifacts.config_sha256
            or manifest["provenance_sha256"] != artifacts.provenance_sha256):
        raise ValueError("Checkpoint does not belong to the expected configuration and provenance")
    raw = (path.parent / "state.pt").read_bytes()
    if sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint state differs from its manifest")
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    fields = {"version", "step", "progress", "config_sha256", "provenance_sha256", "model",
              "module_training", "component_types", "optimizer", "scheduler", "scaler", "rng"}
    if (not isinstance(payload, dict) or set(payload) != fields
            or any(payload[key] != manifest[key] for key in
                   ("version", "step", "config_sha256", "provenance_sha256"))
            or type(payload["step"]) is not int or not isinstance(payload["progress"], dict)
            or not isinstance(payload["component_types"], dict)
            or payload["component_types"].get("model") != "kenggo.learner.PopulationModel"):
        raise ValueError("Checkpoint payload, model type and manifest disagree")
    state, expected = payload["model"], model.state_dict()
    if (not isinstance(state, dict) or state.keys() != expected.keys()
            or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                   or value.layout != torch.strided or value.shape != expected[key].shape
                   or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
                   for key, value in state.items())):
        raise ValueError("Checkpoint model weights have invalid structure, dtype or values")
    modes = payload["module_training"]
    if (not isinstance(modes, dict) or modes.keys() != dict(model.named_modules()).keys()
            or any(type(mode) is not bool for mode in modes.values())):
        raise ValueError("Checkpoint module structure differs from the configured model")
    model.load_state_dict(state, strict=True)


class LearnedPolicy:
    def __init__(self, artifacts):
        if not isinstance(artifacts, LearnedArtifacts):
            raise ValueError("A pinned learned-policy artifact bundle is required")
        normalizer, training_komi, inference_komi = _normalizer(artifacts)
        torch.set_num_threads(CPU_THREADS)
        # Construction initializes parameters; preserve the caller's training RNG.
        with torch.random.fork_rng(devices=[], device_type="cpu"), torch.device("cpu"):
            model = PopulationModel(artifacts.model_config).float()
        _weights(artifacts, model)
        model.eval().requires_grad_(False)
        self.artifacts, self._model, self._normalizer = artifacts, model, normalizer
        self.training_komi_halfpoints, self.inference_komi_halfpoints = training_komi, inference_komi
        self._closed = False

    @classmethod
    def load(cls, artifacts):
        return cls(artifacts)

    @torch.inference_mode()
    def predict_batch(self, inputs, *, normalizer_sha256):
        """Predict already normalized shared CPU tensors; the caller supplies their pin."""
        if self._closed:
            raise ValueError("Learned policy is closed")
        if (normalizer_sha256 != self.artifacts.normalizer_sha256
                or not isinstance(inputs, dict) or set(inputs) != set(INPUT_KEYS)
                or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                       for value in inputs.values())):
            raise ValueError("Prediction requires shared CPU inputs with the pinned normalizer")
        scalars, valid = inputs["scalars"], inputs["scalar_valid"]
        if (scalars.ndim != 2 or scalars.shape[1] != len(SCALARS)
                or valid.shape != scalars.shape or valid.dtype != torch.bool
                or not valid[:, :2].all()
                or not ((scalars[:, 0] == 0) | (scalars[:, 0] == 1)).all()):
            raise ValueError(self._komi_error())
        supported = torch.zeros_like(scalars[:, 1], dtype=torch.bool)
        for halfpoints in self.inference_komi_halfpoints:
            supported |= scalars[:, 1] == (1 - 2 * scalars[:, 0]) * (halfpoints / 2)
        if not supported.all():
            raise ValueError(self._komi_error())
        logits, value = self._model(inputs)
        legal = inputs["legal"]
        if (logits.shape != legal.shape or value.shape != legal.shape[:1]
                or logits.dtype != torch.float32 or value.dtype != torch.float32
                or not torch.isfinite(logits[legal]).all()
                or not torch.isneginf(logits[~legal]).all() or not torch.isfinite(value).all()
                or (value.abs() > 1).any()):
            raise ValueError("Learned policy returned invalid legal logits or mover values")
        return logits, value

    def _komi_error(self):
        values = " or ".join(str(value / 2) for value in self.inference_komi_halfpoints)
        return f"The learned checkpoint supports only signed {values}-point komi"

    @torch.inference_mode()
    def predict_raw_batch(self, inputs, *, unknown_ranks=False):
        """Normalize shared raw CPU observations with this checkpoint's training fit."""
        if (type(unknown_ranks) is not bool or not isinstance(inputs, dict)
                or set(inputs) != set(INPUT_KEYS)
                or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                       or value.layout != torch.strided for value in inputs.values())):
            raise ValueError("Raw prediction requires shared CPU input tensors without labels")
        normalized = apply_normalizer({key: value.detach().numpy() for key, value in inputs.items()},
                                      self._normalizer)
        tensors = {key: torch.from_numpy(value) for key, value in normalized.items()}
        if unknown_ranks:
            tensors["ranks"].zero_()
            tensors["rank_valid"].zero_()
        return self.predict_batch(tensors, normalizer_sha256=self.artifacts.normalizer_sha256)

    def predict(self, game, context=Context()):
        """Encode a verified full-history position with a zero-length model history."""
        if isinstance(game, Game) and game.komi_halfpoints not in self.inference_komi_halfpoints:
            raise ValueError(self._komi_error())
        if (not isinstance(game, Game) or game.status != "playing"
                or not isinstance(context, Context) or load_game(dump_game(game)) != game):
            raise ValueError("Prediction requires a playable ordinary game with intact history")
        row = apply_normalizer(encode(game, context, history=0), self._normalizer)
        inputs = {key: torch.from_numpy(value[None]) for key, value in row.items()}
        logits, value = self.predict_batch(inputs, normalizer_sha256=self.artifacts.normalizer_sha256)
        return logits[0], float(value[0])

    def close(self):
        self._closed = True


def _implementation_pin():
    directory = Path(__file__).parent
    return json_hash({name: file_hash(directory / name) for name in
                      ("learned_agent.py", "learner.py", "encoding.py", "go.py",
                       "rules_replay.py", "checkpoints.py", "kgs_source.py")})


def learned_agent_profile(id, artifacts, rank, *, source_revision, panel="validation"):
    from kenggo.matches import AgentProfile

    if (not isinstance(artifacts, LearnedArtifacts) or rank is not None and rank not in RANKS
            or not artifacts.model_config.rank_conditioned and rank is not None):
        raise ValueError("A pinned model and a supported or explicitly unknown conditioning rank are required")
    return AgentProfile.from_dict({
        "id": id, "family": "learned_population", "policy": POLICY, "panel": panel,
        "context_id": "untimed_missing_clock_unknown_rank_v1",
        "context": {"timed": False, "clock_inputs": "missing", "opponent_rank": "unknown"},
        "device": "cpu", "cpu_threads": CPU_THREADS, "human_rank": None, "conditioning_rank": rank,
        "source_revision": source_revision, "checkpoint_sha256": artifacts.checkpoint_sha256,
        "pins": {"checkpoint": artifacts.checkpoint_sha256, "config": artifacts.config_sha256,
                 "provenance": artifacts.provenance_sha256, "normalizer": artifacts.normalizer_sha256,
                 "examples": artifacts.examples_sha256, "model": json_hash(asdict(artifacts.model_config)),
                 "implementation": _implementation_pin()},
        "model": asdict(artifacts.model_config), "rules": RULES, "komi_halfpoints": 13,
        "sampling": "legal_policy_temperature_1", "pass": "raw_policy", "resignation": "never",
        "history": 0, "legality_history": "full", "value_view": "mover_recorded_outcome",
        "ending": "exact_first_optional_pinned_empirical_v2"})


@dataclass(frozen=True)
class LearnedMatchFactory:
    artifacts: LearnedArtifacts
    source_revision: str

    def __call__(self, profile, seed):
        from kenggo.matches import InfrastructureError

        try:
            expected = learned_agent_profile(profile["id"], self.artifacts, profile["conditioning_rank"],
                source_revision=self.source_revision, panel=profile["panel"]).data
            if ("weight" in profile and (type(profile["weight"]) not in (int, float)
                    or not math.isfinite(profile["weight"]) or not 0 < profile["weight"] <= 1)
                    or {key: value for key, value in profile.items() if key != "weight"} != expected
                    or type(seed) is not int or seed < 0):
                raise ValueError("Scheduled learned profile or seed differs from its pins")
            client = LearnedPolicy.load(self.artifacts)
            return _LearnedAgent(client, profile["conditioning_rank"], seed)
        except Exception as error:
            raise InfrastructureError(str(error)) from error


class _LearnedAgent:
    def __init__(self, client, rank, seed):
        self.client, self.rank, self.rng = client, rank, random.Random(seed)

    def choose(self, game):
        from kenggo.matches import Decision

        context = (Context(black_rank=self.rank) if game.position.to_play == "B"
                   else Context(white_rank=self.rank))
        logits, _ = self.client.predict(game, context)
        probabilities = logits.double().softmax(dim=0).tolist()
        action = self.rng.choices(range(PASS + 1), weights=probabilities, k=1)[0]
        return Decision(action, 1, 0.0)

    def close(self):
        self.client.close()
