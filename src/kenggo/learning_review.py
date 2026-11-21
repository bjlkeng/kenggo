"""Offline learning curves and run history from saved review evidence."""

from copy import deepcopy
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "tools/learning_review.html"
DEFAULT_HISTORY = ROOT / "configs/e2_review_history_v1.json"
MAX_BYTES = 64 * 1024**2
DATA_PATTERN = re.compile(
    r'<script id="(?:pilot-data|expanded-data)" type="application/json">(.*?)</script>', re.S)


def read_review(reference):
    path = (ROOT / reference["path"]).resolve()
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Saved review exceeds the 64 MiB limit")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Saved review differs from its pinned hash")
    page = raw.decode("utf-8")
    matches = DATA_PATTERN.findall(page)
    if len(matches) != 1:
        raise ValueError("Expected one saved BASE or expanded review payload")
    data = json.loads(matches[0])
    if not isinstance(data.get("training"), dict) or not isinstance(data.get("evaluations"), dict):
        raise ValueError("The saved page lacks training and evaluation evidence")
    return data, page


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _checkpoint_metrics(report):
    result, identities = [], {}
    for record in report.get("validation_metrics", []):
        step, epoch, lifetime = record.get("sampler_step"), record.get("round"), record.get("lifetime_updates")
        if (type(step) is not int or step < 0 or type(epoch) is not int or epoch < 0
                or type(lifetime) is not int or lifetime < step
                or not result and (step != 0 or epoch != 0)
                or result and (step <= result[-1]["step"] or epoch <= result[-1]["epoch"]
                               or lifetime - step != result[0]["lifetime_updates"])
                or step > report.get("completed_updates", step)):
            raise ValueError("Checkpoint curves require epoch zero and increasing stage/lifetime cursors")
        point = {"step": step, "epoch": epoch, "lifetime_updates": lifetime}
        for name in ("training_monitor", "validation"):
            values = record.get(name, {})
            if any(not _finite(values.get(key)) for key in ("nll", "top1", "value_mse")):
                raise ValueError("Checkpoint diagnostics require recorded finite NLL, agreement and value MSE")
            if (not 0 <= values["top1"] <= 1 or values["nll"] < 0 or values["value_mse"] < 0
                    or any(type(values.get(key)) is not int or values[key] < 1
                           for key in ("rows", "policy_count", "value_count"))
                    or values["policy_count"] > values["rows"] or values["value_count"] > values["rows"]
                    or not isinstance(values.get("aggregate_ranks"), list) or not values["aggregate_ranks"]
                    or not isinstance(values.get("diagnostic_sha256"), str)):
                raise ValueError("Checkpoint diagnostics need valid counts, ranks and their frozen sample pin")
            identity = (values["diagnostic_sha256"], tuple(values["aggregate_ranks"]), values["rows"],
                        values["policy_count"], values["value_count"])
            if identities.setdefault(name, identity) != identity:
                raise ValueError("A checkpoint curve cannot change its diagnostic sample or masks")
            point[name] = {key: deepcopy(values[key]) for key in (
                "nll", "top1", "value_mse", "rows", "policy_count", "value_count", "aggregate_ranks",
                "diagnostic_sha256", "excluded_rank_rows", "evaluated_rows", "weighting") if key in values}
        point["new_raw_best"] = record.get("new_raw_best") is True
        point["meaningful_improvement"] = record.get("meaningful_improvement") is True
        result.append(point)
    return result


def _training(report):
    points = []
    blocks = report.get("update_metrics", report.get("epoch_metrics", []))
    for block in blocks:
        step = block.get("next_step")
        if step is None:
            continue
        if (type(step) is not int or step < 1 or points and step <= points[-1]["step"]
                or step > report.get("completed_updates", step)):
            raise ValueError("Training curve steps must increase within completed training")
        for key in ("policy_loss", "value_loss"):
            if block.get(key) is not None and not _finite(block[key]):
                raise ValueError("Training losses must be finite or missing")
        points.append({"step": step, "epoch": block.get("epoch"), "updates": block.get("updates"),
            "examples": block.get("presented_examples"), "policy_count": block.get("policy_count"),
            "value_count": block.get("value_count"), "nll": block.get("policy_loss"),
            "value_mse": block.get("value_loss")})
    checkpoints = _checkpoint_metrics(report)
    selected = None
    if checkpoints:
        step = report.get("selected_sampler_step")
        if step is None:
            marked = [point for point in checkpoints if point["new_raw_best"]]
            step = marked[-1]["step"] if marked else None
        selected = next((point for point in checkpoints if point["step"] == step), None)
        if (selected is None or type(step) is not int
                or report.get("selected_round", selected["epoch"]) != selected["epoch"]
                or report.get("selected_lifetime_updates", selected["lifetime_updates"]) != selected["lifetime_updates"]):
            raise ValueError("The selected checkpoint must name a saved checkpoint measurement")
    return {"points": points, "updates": report.get("completed_updates"),
            "presentations": report.get("presented_examples"), "rows": report.get("training_rows"),
            "checkpoint": report.get("final_checkpoint"), "status": report.get("status"),
            "window": report.get("training_window", "epoch" if "epoch_metrics" in report else "update block"),
            "checkpoints": checkpoints, "selected_step": selected["step"] if selected else report.get("completed_updates"),
            "selected_round": selected["epoch"] if selected else None,
            "selected_lifetime_updates": selected["lifetime_updates"] if selected else None}


def _metric(metric):
    result = {key: deepcopy(metric[key]) for key in (
        "status", "estimate", "counts", "masked_moves", "aggregate_ranks", "missing_ranks") if key in metric}
    result["uncertainty"] = {key: deepcopy(metric["uncertainty"][key]) for key in (
        "status", "interval95", "method", "reasons") if key in metric.get("uncertainty", {})}
    return result


def normalize(data, metadata):
    evaluations = {}
    for identity, arms in data["evaluations"].items():
        if identity in ("population_development", "validation_development"):
            identity += "_komi13"
        evaluations[identity] = {}
        for arm, output in arms.items():
            report = output.get("metrics", {})
            scopes = {"reported": report}
            if "review_scope" in report:
                scopes["review_scope"] = report["review_scope"]
            evaluations[identity][arm] = {
                scope: {"ranks": value.get("aggregate_ranks", []),
                        "rows": value.get("expected_outputs", output.get("rows")),
                        "excluded_rows": value.get("excluded_rows", 0),
                        "metrics": {key: _metric(value.get("overall", {}).get("metrics", {}).get(key, {}))
                                    for key in ("nll", "top1", "value_mse")}}
                for scope, value in scopes.items()}
    return {"id": metadata.get("id", data["bundle_id"]),
        "title": metadata.get("title", data.get("title", "Current saved experiment")),
        "short": metadata.get("short", "Current"),
        "description": metadata.get("description", "Saved training and final development evaluation."),
        "notes": metadata.get("notes", []), "source": metadata.get("source", data.get("input")),
        "bundle_id": data["bundle_id"], "current": False,
        "default_scope": data.get("default_scope", "reported").replace("all_ranks", "reported"),
        "training": {arm: _training(report) for arm, report in data["training"].items()},
        "evaluations": evaluations}


def build(data, catalog_path=None):
    history, references, current = [], [], {}
    catalog_ref = None
    if catalog_path is not None:
        path = Path(catalog_path)
        raw = path.read_bytes()
        catalog = json.loads(raw)
        if catalog.get("schema") != "e2_review_history_v1":
            raise ValueError("Unknown review-history catalog")
        entries = catalog["runs"]
        if len({item["id"] for item in entries}) != len(entries):
            raise ValueError("Review-history IDs must be unique")
        catalog_ref = {"path": str(path.resolve()), "sha256": sha256(raw).hexdigest()}
        for entry in entries:
            if entry["bundle_id"] == data["bundle_id"]:
                current = entry
                break
            previous, _ = read_review(entry["source"])
            if previous["bundle_id"] != entry["bundle_id"]:
                raise ValueError("Catalog entry names a different experiment")
            history.append(normalize(previous, entry))
            references.append(entry["source"])
    selected = normalize(data, current)
    selected["current"] = True
    return {"schema": "learning_review_v1", "runs": history + [selected],
            "catalog": catalog_ref, "history_sources": references,
            "history_status": "available" if history else "No earlier review supplied for this run."}


def insert(page, learning):
    if 'id="learning-review"' in page:
        raise ValueError("This page already contains a learning review")
    encoded = json.dumps(learning, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = TEMPLATE.read_text()
    if template.count("__LEARNING_DATA_JSON__") != 1:
        raise ValueError("Expected one learning-review data marker")
    fragment = template.replace("__LEARNING_DATA_JSON__", encoded)
    if not re.search(r"<main\b", page):
        raise ValueError("Review page has no main content")
    page = re.sub(r"<main\b", lambda _: fragment + "\n<main", page, count=1)
    if len(page.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Review page exceeds its 64 MiB limit")
    return page


def preserve_links(page, source):
    """Keep relative evidence links working when a sealed page is copied elsewhere."""
    def replace(match):
        value = html.unescape(match[1])
        if not value or value.startswith("#") or urlsplit(value).scheme:
            return match[0]
        return 'href="' + html.escape(urljoin(source.as_uri(), value), quote=True) + '"'
    return re.sub(r'href="([^"]*)"', replace, page)


def source_pins():
    return [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in (Path(__file__), TEMPLATE)]
