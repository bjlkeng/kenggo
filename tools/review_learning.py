"""Add learning curves and history to a new copy of a pinned review page."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from kenggo.learning_review import (DEFAULT_HISTORY, ROOT, build, insert, preserve_links,
                                   read_review, source_pins)


def export(catalog_path, run_id, output):
    started, cpu = time.monotonic(), time.process_time()
    catalog_path, output = Path(catalog_path), Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or output.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("Review output and manifest paths must be unused")
    catalog = json.loads(catalog_path.read_text())
    entry = next(item for item in catalog["runs"] if item["id"] == run_id)
    data, page = read_review(entry["source"])
    if data["bundle_id"] != entry["bundle_id"]:
        raise ValueError("Current page differs from its catalog experiment")
    learning = build(data, catalog_path)
    learning["original_page_url"] = (ROOT / entry["source"]["path"]).resolve().as_uri()
    page = insert(preserve_links(page, (ROOT / entry["source"]["path"]).resolve()), learning)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(page)
    manifest = {"schema": "learning_review_export_v1", "source": entry["source"],
        "html": str(output.resolve()), "html_sha256": sha256(output.read_bytes()).hexdigest(),
        "catalog": learning["catalog"], "history_sources": learning["history_sources"],
        "code": source_pins() + [{"path": "tools/review_learning.py", "sha256": sha256(Path(__file__).read_bytes()).hexdigest()}],
        "runs": [run["id"] for run in learning["runs"]],
        "training_points": {run["id"]: {arm: len(item["points"]) for arm, item in run["training"].items()}
                            for run in learning["runs"]},
        "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started,
        "inference_calls": 0, "checkpoint_reads": 0, "raw_dataset_reads": 0, "gpu_seconds": 0}
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run ID in the review history catalog")
    parser.add_argument("output", type=Path, help="Unused output HTML path")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    print(json.dumps(export(args.history, args.run_id, args.output), indent=2))
