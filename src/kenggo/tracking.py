"""Local run evidence, W&B tracking, and explicit checkpoint-based continuation."""

from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import time
import uuid

from kenggo.checkpoints import (VERSION as CHECKPOINT_VERSION, file_hash, json_bytes, json_hash, load_checkpoint,
                                read_json, save_checkpoint, save_json)


VERSION = "experiment_run_v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def capture_provenance(repository, inputs):
    repository = Path(repository).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], timeout=10)
    if git("ls-files", "--others", "--exclude-standard").strip():
        raise ValueError("Stage intended new source files before capturing provenance")
    checked = json.loads(json_bytes(inputs))
    if not {"data", "split", "rules"} <= checked.keys():
        raise ValueError("Run provenance requires data, split and rules pins")
    for ref in checked.values():
        path = repository / ref["path"]
        if file_hash(path) != ref["sha256"]:
            raise ValueError(f"Run input differs from its pin: {ref['path']}")
    patch = git("diff", "--binary", "HEAD")
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    provenance = {"version": "experiment_provenance_v1", "inputs": checked,
                  "source": {"commit": git("rev-parse", "HEAD").decode().strip(),
                             "patch_sha256": sha256(patch).hexdigest()},
                  "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                  "packages": dict(sorted(packages.items()))}}
    return provenance, patch


class WandbSink:
    def __init__(self, directory, config, provenance, run_id, *, mode, resume):
        import wandb
        self.mode = mode
        self.run = wandb.init(entity=config["entity"], project=config["project"], id=run_id,
            group=config["comparison"], job_type=config["purpose"], name=f"{config['arm']}-{run_id}",
            mode=mode, resume="must" if resume else "never", dir=str(directory), save_code=False,
            config={"experiment": config, "provenance": provenance},
            settings=wandb.Settings(base_url=config["base_url"], console="off", disable_git=True,
                disable_code=True, x_disable_stats=True, init_timeout=30, finish_timeout=45,
                finish_timeout_raises=True, x_graphql_timeout_seconds=15, x_graphql_retry_max=1,
                x_file_transfer_timeout_seconds=30, x_file_transfer_retry_max=1,
                x_file_stream_timeout_seconds=15, x_file_stream_retry_max=1))
        self.run.define_metric("train_step")
        self.run.define_metric("*", step_metric="train_step")
        self.reference = {"entity": config["entity"], "project": config["project"], "id": run_id,
                          "mode": mode, "url": self.run.url if mode == "online" else None}

    def log(self, step, metrics, attempt):
        self.run.log({"train_step": step, "attempt": attempt, **metrics})

    def artifact(self, name, files, metadata):
        import wandb
        artifact = wandb.Artifact(name, type="checkpoint", metadata=metadata)
        for label, path in files.items():
            artifact.add_file(str(path), name=label)
        self.run.log_artifact(artifact)
        if self.mode == "offline":
            return {"mode": "offline", "name": name, "synced": False}
        artifact.wait(timeout=45)
        return {"mode": "online", "name": artifact.qualified_name, "digest": artifact.digest,
                "synced": True}

    def finish(self, status, summary):
        self.run.summary.update(summary)
        self.run.finish(exit_code=0 if status == "completed" else 1)


def validate_config(config):
    if (set(config) != {"version", "entity", "project", "base_url", "comparison", "arm", "purpose",
                       "seed", "parameters", "inputs", "budget_id"}
            or config["version"] != "experiment_config_v1"
            or config["base_url"] != "https://api.wandb.ai"
            or type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32
            or not isinstance(config["parameters"], dict) or not isinstance(config["inputs"], dict)):
        raise ValueError("Invalid experiment configuration")
    for key in ("entity", "project", "comparison", "arm", "purpose", "budget_id"):
        if not isinstance(config[key], str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", config[key]) is None:
            raise ValueError(f"Invalid experiment {key}")
    json_bytes(config)
    limit = config["parameters"].get("max_artifact_bytes")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Artifact allowance must be a positive byte count")


class Experiment:
    def __init__(self, directory, config, provenance, patch=b"", *, mode="online", resume=False,
                 sink_factory=WandbSink):
        validate_config(config)
        if mode not in {"online", "offline"} or resume and mode != "online":
            raise ValueError("Resume needs online W&B; offline runs remain explicitly unsynced")
        if (provenance.get("version") != "experiment_provenance_v1"
                or provenance.get("inputs") != config["inputs"]
                or provenance.get("source", {}).get("patch_sha256") != sha256(patch).hexdigest()):
            raise ValueError("Provenance does not match the configured inputs and source patch")
        self.config = json.loads(json_bytes(config))
        self.provenance = json.loads(json_bytes(provenance))
        self.config_sha256, self.provenance_sha256 = json_hash(config), json_hash(provenance)
        self.directory = Path(directory)
        self.sink, self.closed, self.needs_restore = None, False, False
        self.started, self.cpu_started = time.monotonic(), time.process_time()
        if not resume:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = (self.directory / ".writer.lock").open("ab")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        try:
            if resume:
                read_json(self.directory / "config.json", self.config_sha256)
                read_json(self.directory / "provenance.json", self.provenance_sha256)
                if file_hash(self.directory / "source.patch") != sha256(patch).hexdigest():
                    raise ValueError("Saved source patch changed")
                self.state = json.loads((self.directory / "run.json").read_text())
                if (self.state["version"] != VERSION or self.state["status"] == "completed"
                        or self.state["config_sha256"] != self.config_sha256
                        or self.state["provenance_sha256"] != self.provenance_sha256
                        or self.state["wandb"]["mode"] != "online"):
                    raise ValueError("Run cannot resume under this configuration")
                if self.state["status"] == "finishing":
                    self.state["attempts"][-1].update(status="interrupted",
                        requested_status=self.state["requested_status"],
                        reason="Process ended before tracking shutdown was confirmed")
                elif self.state["attempts"][-1]["status"] == "running":
                    self.state["attempts"][-1].update(status="interrupted", reason="Process ended without finish")
                self.needs_restore = bool(self.state["checkpoints"])
            else:
                save_json(self.directory / "config.json", config)
                save_json(self.directory / "provenance.json", provenance)
                with (self.directory / "source.patch").open("xb") as stream:
                    stream.write(patch)
                self.state = {"version": VERSION, "id": uuid.uuid4().hex,
                    "config_sha256": self.config_sha256, "provenance_sha256": self.provenance_sha256,
                    "status": "running", "wandb": {"mode": mode, "url": None},
                    "attempts": [], "checkpoints": [], "artifact_bytes_attempted": 0}
            self.attempt = {"number": len(self.state["attempts"]) + 1, "start_utc": utc_now(),
                            "status": "running", "last_step": None}
            self.state["attempts"].append(self.attempt)
            self.state.update(status="running", tracking_finished=False)
            self.state.pop("requested_status", None)
            self._write()
            self.sink = sink_factory(self.directory, self.config, self.provenance, self.state["id"],
                                     mode=mode, resume=resume)
            self.state["wandb"] = self.sink.reference
            self._write()
        except BaseException as error:
            if hasattr(self, "attempt"):
                self.attempt.update(status="failed", reason=f"Tracking initialization: {type(error).__name__}")
                self.state["status"] = "failed"
                self._costs()
                self._write()
            self.lock.close()
            self.closed = True
            raise

    def _write(self):
        save_json(self.directory / "run.json", self.state, replace=True)

    def _require_open(self):
        if self.closed:
            raise RuntimeError("Experiment is closed")

    def log(self, step, metrics):
        self._require_open()
        if self.needs_restore:
            raise ValueError("Restore a pinned checkpoint before logging a resumed run")
        if (type(step) is not int or step < 0 or not isinstance(metrics, dict)
                or {"train_step", "attempt"} & metrics.keys()
                or self.attempt["last_step"] is not None and step < self.attempt["last_step"]):
            raise ValueError("Metrics need a nondecreasing step and no reserved keys")
        entry = {"step": step, "metrics": metrics, "attempt": self.attempt["number"], "utc": utc_now()}
        encoded = json_bytes(entry)
        with (self.directory / "metrics.jsonl").open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self.attempt["last_step"] = step
        self._write()
        self.sink.log(step, metrics, self.attempt["number"])

    def checkpoint(self, model, optimizer, *, step, progress, scheduler=None, scaler=None):
        self._require_open()
        if self.needs_restore:
            raise ValueError("Restore before saving a resumed checkpoint")
        ref = save_checkpoint(self.directory / "checkpoints", model, optimizer, step=step, progress=progress,
            config_sha256=self.config_sha256, provenance_sha256=self.provenance_sha256,
            scheduler=scheduler, scaler=scaler)
        return self._publish_checkpoint(ref, step)

    def register_checkpoint(self, ref):
        """Publish an owned worker's saved checkpoint without restoring training state."""
        self._require_open()
        if self.needs_restore:
            raise ValueError("Restore before registering a resumed checkpoint")
        path = Path(ref["path"]).resolve()
        root = self.directory.resolve()
        blob = (path.parent / "state.pt").resolve()
        if not path.is_relative_to(root) or not blob.is_relative_to(root):
            raise ValueError("A worker checkpoint must stay inside its owning run")
        manifest = read_json(path, ref["sha256"])
        if (set(manifest) != {"version", "file", "sha256", "step", "config_sha256", "provenance_sha256"}
                or manifest["version"] != CHECKPOINT_VERSION or manifest["file"] != "state.pt"
                or type(manifest["step"]) is not int or manifest["step"] < 0
                or manifest["config_sha256"] != self.config_sha256
                or manifest["provenance_sha256"] != self.provenance_sha256
                or file_hash(blob) != manifest["sha256"]):
            raise ValueError("Worker checkpoint differs from its configuration, provenance or saved state")
        if any(entry["sha256"] == ref["sha256"] for entry in self.state["checkpoints"]):
            raise ValueError("Checkpoint is already registered")
        return self._publish_checkpoint({"path": str(path), "sha256": ref["sha256"]}, manifest["step"])

    def _publish_checkpoint(self, ref, step):
        path = Path(ref["path"]).resolve()
        entry = {"path": str(path.relative_to(self.directory.resolve())), "sha256": ref["sha256"],
                 "step": step, "attempt": self.attempt["number"], "wandb": None}
        self.state["checkpoints"].append(entry)
        self._write()
        files = {"state.pt": path.parent / "state.pt", "manifest.json": path,
                 **{name: self.directory / name for name in ("config.json", "provenance.json", "source.patch")}}
        total = self.state["artifact_bytes_attempted"] + sum(file.stat().st_size for file in files.values())
        limit = self.config["parameters"].get("max_artifact_bytes")
        if limit is not None and total > limit:
            raise ValueError("Artifact upload would exceed the configured byte allowance")
        self.state["artifact_bytes_attempted"] = total
        self._write()
        entry["wandb"] = self.sink.artifact(f"{self.state['id']}-step-{step}-{path.parent.name[-8:]}",
            files, {"checkpoint_sha256": ref["sha256"], "config_sha256": self.config_sha256,
                    "provenance_sha256": self.provenance_sha256, "step": step})
        self._write()
        return ref

    def restore(self, ref, model, optimizer, *, scheduler=None, scaler=None):
        self._require_open()
        if not any(row["sha256"] == ref["sha256"] for row in self.state["checkpoints"]):
            raise ValueError("Checkpoint is not recorded in this run")
        result = load_checkpoint(ref["path"], ref["sha256"], model, optimizer,
            config_sha256=self.config_sha256, provenance_sha256=self.provenance_sha256,
            scheduler=scheduler, scaler=scaler)
        self.attempt["restored_checkpoint"] = ref["sha256"]
        self.attempt["last_step"] = result["step"]
        self.needs_restore = False
        self._write()
        return result

    def _costs(self):
        self.attempt.update(stop_utc=utc_now(), cpu_seconds=time.process_time() - self.cpu_started,
            wall_seconds=time.monotonic() - self.started,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            cost_scope="Calling-process CPU and elapsed time from initialization through the body; tracking shutdown recorded separately. GPU allocations belong to the budget ledger.")

    def finish(self, status="completed", *, reason=None, summary=None):
        self._require_open()
        if status not in {"completed", "partial", "failed", "cancelled"} or status != "completed" and not reason:
            raise ValueError("Finish requires an explicit outcome and a reason when incomplete")
        if self.needs_restore and status == "completed":
            raise ValueError("A resumed run needs its checkpoint restored before completion")
        if summary is not None and (not isinstance(summary, dict)
                or any(not isinstance(key, str) or key in {"outcome", "reason", "cost_scope"}
                       or key.startswith(("attempts/", "costs/")) for key in summary)):
            raise ValueError("Summary must be an object without reserved tracking keys")
        json_bytes(summary or {})
        self.attempt.update(status=status, reason=reason)
        self.state.update(status="finishing", requested_status=status, summary=summary or {})
        self._costs()
        self._write()
        shutdown, cpu = time.monotonic(), time.process_time()
        remote_summary = {"outcome": status, "reason": reason, **(summary or {}),
                          "attempts/count": len(self.state["attempts"]),
                          "cost_scope": "Recorded attempt costs only; missing interrupted-process costs and final tracking shutdown are not included."}
        for attempt in self.state["attempts"]:
            for key, value in attempt.items():
                remote_summary[f"attempts/{attempt['number']}/{key}"] = value
        remote_summary["costs/process_cpu_seconds"] = sum(
            a.get("cpu_seconds", 0) + a.get("tracking_shutdown_cpu_seconds", 0) for a in self.state["attempts"])
        remote_summary["costs/wall_seconds"] = sum(
            a.get("wall_seconds", 0) + a.get("tracking_shutdown_wall_seconds", 0) for a in self.state["attempts"])
        remote_summary["costs/incomplete_attempts"] = sum(
            "cpu_seconds" not in a or "wall_seconds" not in a
            or a is not self.attempt and ("tracking_shutdown_cpu_seconds" not in a
                                         or "tracking_shutdown_wall_seconds" not in a)
            for a in self.state["attempts"])
        try:
            self.sink.finish(status, remote_summary)
            self.state.update(status=status, tracking_finished=True)
        except BaseException as error:
            self.state.update(status="failed", tracking_finished=False)
            self.attempt.update(status="failed", tracking_error=type(error).__name__)
            raise
        finally:
            self.attempt["tracking_shutdown_wall_seconds"] = time.monotonic() - shutdown
            self.attempt["tracking_shutdown_cpu_seconds"] = time.process_time() - cpu
            try:
                self._write()
            finally:
                self.closed = True
                self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if not self.closed:
            self.finish("failed" if error is not None else "completed",
                        reason=type(error).__name__ if error is not None else None)
        return False
