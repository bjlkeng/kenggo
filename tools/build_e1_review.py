"""Export the frozen E1 manual checks as one offline HTML page."""

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np

from kenggo.canonical_data import digest
from kenggo.encoding import SCALARS
from kenggo.examples import (file_hash, load_corpus, load_dataset_profile, read_pinned_json,
                             read_shards, replay_record)
from kenggo.go import PASS
from kenggo.kgs_source import RANKS


ROOT = Path(__file__).resolve().parents[1]
READINESS = "fe062804cfb1952715daaed055de85e9d26c7d9fd32f6a64092dc97eff3a1a2a"
EXAMPLE_ROLES = ("train", "population_development", "validation_development")


def pinned(reference, *, json_data=True):
    path = ROOT / reference["path"]
    if json_data:
        return read_pinned_json(path, reference["sha256"], max_bytes=16 * 1024**2)
    if file_hash(path) != reference["sha256"]:
        raise ValueError(f"Changed artifact: {reference['path']}")
    return path


def frames(states):
    """Store verified board changes, not a second Go implementation in JavaScript."""
    previous = "." * PASS
    result = []
    for state in states:
        board = "".join(state["board"])
        if len(board) != PASS or set(board) - set(".BW"):
            raise ValueError("Invalid replay board")
        changes = [[i, cell] for i, cell in enumerate(board) if cell != previous[i]]
        result.append([changes, state["to_play"], state["captures"]["B"],
                       state["captures"]["W"], state["pass_streak"], state.get("encore_phase", 0)])
        previous = board
    return result


def source_frames(game):
    return frames({"board": p.rows, "to_play": p.to_play,
                   "captures": {"B": p.captured_black, "W": p.captured_white},
                   "pass_streak": p.pass_streak} for p in game.positions)


def native_cleanup(binary, game, run):
    moves = [[m.color, m.action] for m in game.moves] + run["cleanup"]
    request = {"mode": "history", "board": list(game.positions[0].rows), "to_play": "B",
               "moves": [{"color": c, "action": a} for c, a in moves]}
    process = subprocess.run([str(binary)], input=json.dumps(request) + "\n", text=True,
                             capture_output=True, check=True, timeout=10)
    result = json.loads(process.stdout)
    steps = result.get("steps", [])
    if (not result.get("ok") or len(steps) != len(moves)
            or any(not step.get("accepted") or [step["color"], step["action"]] != move
                   for step, move in zip(steps, moves))):
        raise ValueError("Native cleanup replay failed")
    states = [result["initial"]] + [step["state"] for step in steps]
    for state, position in zip(states, game.positions):
        if (state["board"] != list(position.rows) or state["to_play"] != position.to_play
                or state["captures"] != {"B": position.captured_black, "W": position.captured_white}):
            raise ValueError("Native replay differs from the original source")
    if result["final"] != run["terminal"]:
        raise ValueError("Saved cleanup terminal differs from pinned native replay")
    return {"visits": run["visits"], "moves": run["cleanup"],
            "frames": frames(states[len(game.moves):]), "terminal": run["terminal"]}


def selection(records, partitions, diagnostics):
    wanted = {row["source_url"] for row in diagnostics["evidence"]}
    for role in EXAMPLE_ROLES:
        wanted.update(row["source_url"] for row in sorted(partitions[role], key=lambda r: r["source_url"])[:2])
    allowed = {row["source_url"] for role in EXAMPLE_ROLES for row in partitions[role]}
    targets = set()
    for key in sorted(wanted & allowed):
        record = records[key]
        if record.get("sealed_final", False):
            raise PermissionError("Sealed source cannot enter manual inspection")
        count = len(record["actions"])
        indices = {0, 1, count // 2, count - 1}
        indices.update(i for i, move in enumerate(record["actions"]) if move["action"] == PASS)
        targets.update((key, i) for i in indices if 0 <= i < count)
    return targets


def stored_example(arrays, index, meta, record, game):
    if record.get("sealed_final", False) or meta["role"] not in EXAMPLE_ROLES:
        raise PermissionError("Only training and development examples may be inspected")
    ply = meta["ply"]
    position = game.positions[ply]
    item = {key: value[index].tolist() for key, value in arrays.items()}
    board = np.asarray(item.pop("board"))
    if np.any(board[0] & board[1]):
        raise ValueError("Stored board planes overlap")
    mover = meta["mover"]
    other = "W" if mover == "B" else "B"
    cells = np.where(board[0], mover, np.where(board[1], other, "."))
    stored = "".join(cells.flat)
    action = item["action"]
    if (stored != "".join(position.board) or mover != position.to_play
            or action != record["actions"][ply]["action"] or not item["legal"][action]
            or item["value_valid"] and (not record["recorded_value_supported"]
                or item["value"] != record["recorded_y_black"] * (1 if mover == "B" else -1))):
        raise ValueError("Stored example differs from its source replay or labels")
    item.update(board=stored, ko=np.flatnonzero(board[2]).tolist(),
                legal="".join("1" if flag else "0" for flag in item["legal"]), metadata=meta)
    return item


def collect_examples(directory, pin, targets, records, games):
    output = []
    for role in EXAMPLE_ROLES:
        for arrays, metadata in read_shards(directory, pin, role):
            for index, meta in enumerate(metadata):
                key = meta["source_url"]
                if (key, meta["ply"]) in targets:
                    output.append(stored_example(arrays, index, meta, records[key], games[key]))
    if {(x["metadata"]["source_url"], x["metadata"]["ply"]) for x in output} != targets:
        raise ValueError("A selected stored example is missing")
    return output


def render(payload):
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    directory = Path(__file__).with_name("e1_review")
    page = (directory / "index.html").read_text()
    return page.replace("/* REVIEW_CSS */", (directory / "review.css").read_text()).replace(
        "/* REVIEW_JS */", (directory / "review.js").read_text()).replace("REVIEW_DATA", serialized)


def build(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"Keep the existing review export; choose a new output: {output}")
    readiness = pinned({"path": "manifests/e1_readiness_v2.json", "sha256": READINESS})
    population = pinned(readiness["population"])
    artifacts = population["artifacts"]
    profile_ref = population["inputs"]["example_profile"]
    profile = load_dataset_profile(ROOT / profile_ref["path"], profile_ref["sha256"])
    canonical = pinned(artifacts["canonical"], json_data=False)
    split = ROOT / artifacts["split_manifest"]["path"]
    records, _, partitions = load_corpus(canonical, profile.canonical_sha256, split.parent,
                                         profile.split_sha256, profile=profile)
    if any(record.get("sealed_final") for record in records.values()):
        raise PermissionError("Sealed record returned by nonfinal reader")
    membership = {row["source_url"]: row for rows in partitions.values() for row in rows}
    ref = readiness["empirical_referee"]
    comparison = pinned(ref["comparison"])
    catalog = pinned(ref["authorized_evidence_catalog"])
    native = pinned(readiness["validation"]["native_reference"], json_data=False)
    diagnostics = pinned(artifacts["diagnostics"])
    example_manifest = pinned(artifacts["examples_manifest"])
    if (example_manifest["canonical_sha256"] != profile.canonical_sha256
            or example_manifest["split_sha256"] != profile.split_sha256):
        raise ValueError("Examples do not belong to this population")
    games, summaries = {}, []
    for key, record in sorted(records.items()):
        if time.monotonic() - started > 300:
            raise TimeoutError("Review export exceeded its 300-second budget")
        gate, meta = record["eligibility"], record["metadata"]
        row = {"id": key, "metadata": meta, "archive": record["archive"], "eligibility": gate,
               "role": membership[key]["role"], "exclusions": membership[key]["exclusions"],
               "source": record["source"], "recorded_result": record.get("recorded_result"),
               "value_supported": record["recorded_value_supported"]}
        if gate["training_eligible"] and not record["structural_exclusions"]:
            game = replay_record(record)
            games[key] = game
            row.update(frames=source_frames(game), moves=[[m.color, m.action] for m in game.moves])
        summaries.append(row)
    by_id = {row["id"]: row for row in summaries}
    checked_archives = set()
    for key in games:
        source = records[key]["source"]
        archive = ROOT / source["archive_path"]
        if str(archive) not in checked_archives:
            if file_hash(archive) != source["archive_sha256"]:
                raise ValueError("Source archive changed")
            checked_archives.add(str(archive))
        with zipfile.ZipFile(archive) as bundle:
            if bundle.getinfo(source["zip_member"]).file_size > 1024**2:
                raise ValueError("Source SGF exceeds 1 MiB")
            raw = bundle.read(source["zip_member"])
        if sha256(raw).hexdigest() != source["sgf_sha256"]:
            raise ValueError("Source SGF changed")
        by_id[key]["sgf_base64"] = base64.b64encode(raw).decode()
    endings = []
    for row in comparison["endings"]:
        if time.monotonic() - started > 300:
            raise TimeoutError("Review export exceeded its 300-second budget")
        key = row["source_url"]
        if key not in games:
            raise ValueError("Referee case is not an admitted nonfinal source")
        path = f"{comparison['directory']}/ending-{row['case']:03d}.json"
        saved = pinned({"path": path, "sha256": row["artifact_sha256"]})
        decision = saved["decision"]
        evidence = json.loads(decision["evidence"])
        if (saved["source_url"] != key or digest(evidence["original"]) != games[key].snapshot
                or sha256(decision["evidence"].encode()).hexdigest() != row["evidence_sha256"]
                or row["evidence_sha256"] not in catalog["evidence_sha256"]
                or evidence["profile_sha256"] != comparison["profile_sha256"]):
            raise ValueError("Saved referee evidence is not authorized for this source")
        runs = [native_cleanup(native, games[key], run) for run in evidence["runs"]]
        endings.append({**row, "runs": runs})
    for row in comparison["passes"]:
        if row["source_url"] not in games:
            raise ValueError("Pass case is not an admitted nonfinal source")
    targets = selection(records, partitions, diagnostics)
    examples = collect_examples((ROOT / artifacts["examples_manifest"]["path"]).parent,
                                artifacts["examples_manifest"]["sha256"], targets, records, games)
    payload = {"version": "e1_manual_review_v1", "readiness_sha256": READINESS,
               "scope": population["scope"], "games": summaries, "endings": endings,
               "passes": comparison["passes"], "examples": examples,
               "scalars": list(SCALARS), "ranks": RANKS, "deferred": readiness["deferred"],
               "counts": {"admitted": len(games), "source_records": len(records),
                          "endings": len(endings), "examples": len(examples)},
               "recorded_validation": {"tests": readiness["validation"]["tests_passed"],
                                       "skips": readiness["validation"]["tests_skipped"]}}
    payload["bundle_id"] = digest(payload)
    page = render(payload)
    if len(page.encode()) > 32 * 1024**2 or time.monotonic() - started > 300:
        raise ValueError("Review export exceeds the byte or wall-time budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(page)
    print(json.dumps({"path": str(output.resolve()), "bytes": output.stat().st_size,
                      "counts": payload["counts"], "bundle_id": payload["bundle_id"],
                      "wall_seconds": round(time.monotonic() - started, 2),
                      "http_requests": 0, "gpu_seconds": 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "runs/e1-manual-review/index.html")
    args = parser.parse_args()
    build(args.output)
