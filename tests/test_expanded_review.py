"""Offline review checks using small saved-result fixtures, without model or dataset reads."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from kenggo.checkpoints import file_hash, save_json


SPEC = importlib.util.spec_from_file_location("expanded_review", Path(__file__).resolve().parents[1] / "tools/review_expanded.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return {"path": str(path), "sha256": save_json(path, value, replace=True)}


def metric(value):
    return {"status": "available", "estimate": {"status": "finite", "value": value},
        "counts": {"moves": 2, "games": 2, "accounts": 2}, "masked_moves": 0,
        "uncertainty": {"status": "inconclusive", "interval95": None,
                        "reasons": ["fewer_than_30_accounts"]}}


def metrics(delta=False):
    group = {"metrics": {"nll": metric(-.04 if delta else 2.1), "top1": metric(.2),
                         "value_mse": metric(.3)}, "value_calibration": {"model": {"bins": []}}}
    return {"kind": "candidate_minus_control" if delta else "single_model", "expected_outputs": 2,
            "overall": group, "by_rank": {"5k": group}, "by_phase": {"opening": group},
            "by_clock": {"10-20": group}}


def fixture(root):
    run = root / "expanded-synthetic"
    run.mkdir()
    plan_ref = ref(root / "plan.json", {"purpose": "expanded-pilot", "seed": 20261118,
                                       "parameters": {"scope": "synthetic fixture"}})
    stages, common, predicted = {}, {}, {}
    entries = [{"run_id": "earlier-pilot", "gpu_seconds": 1246.177333, "cpu_seconds": 1., "wall_seconds": 1.}]
    for name in review.STAGES:
        directory, summary = run / name, {"status": "completed"}
        worker = directory / "worker"
        worker.mkdir(parents=True)
        config = {"inputs": {"raw_data_never_opened": {"path": str(root / "missing-dataset.parquet"),
                                                        "sha256": "a" * 64}}, "parameters": {}}
        if name == "preparation":
            selected = []
            for identity in review.SETS:
                role, komi = identity.rsplit("_komi", 1)
                spec = {"id": identity, "role": role, "komi_halfpoints": int(komi),
                        "source_ids": ["https://example.invalid/" + identity + suffix for suffix in ("-a", "-z")],
                        "aggregate_ranks": ["5k"]}
                manifest = ref(worker / identity / "manifest.json", {"role": role,
                    "source_ids": list(reversed(spec["source_ids"])),
                    "rows": [{"row_key": identity + "-0"}, {"row_key": identity + "-1"}]})
                item = {"spec": spec, "rows": 2, "manifest": manifest, "coverage": {"games": 2, "accounts": 4}}
                common[identity] = item
                selected.append(item)
            summary.update(common_sets=selected, selection=ref(worker / "selection.json",
                {"schema": "expanded_selection_v1", "target_or_prediction_selection": False, "sets": selected}))
        elif name in review.ARMS or name.startswith("probe-"):
            arm = name.removeprefix("probe-")
            probe = name.startswith("probe-")
            updates = 256 if probe else 8
            summary.update(training_rows=1000000 if arm.startswith("baseline") else 2000000,
                completed_updates=updates, presented_examples=updates * 128,
                initial_tensor_sha256="a" * 64, row_keys_sha256="b" * 64 if arm.startswith("baseline") else "c" * 64,
                checkpoint_selection="fixed_final_update", final_checkpoint={"path": "never-open-checkpoint/manifest.json"},
                update_metrics=[{"phase": "warmup", "updates": 6, "seconds_per_update": .02, "order_sha256": "d" * 64}]
                    + [{"phase": "measured", "updates": 50 if probe else 2, "seconds_per_update": .01,
                        "order_sha256": "e" * 64} for _ in range(5 if probe else 1)])
        elif name.startswith("evaluation-"):
            arm, reports = name.removeprefix("evaluation-"), []
            for identity, common_item in common.items():
                if arm == "human-reference" and common_item["spec"]["komi_halfpoints"] != 13:
                    continue
                artifacts = {}
                for kind, value in (("metrics", metrics()), ("sampling", {"samples_per_position": 32}),
                                    ("timing", {"scope": "synthetic", "measurements": []})):
                    path = worker / identity / (kind + ".json")
                    artifact = ref(path, value)
                    artifacts[path.name] = {"path": str(path.relative_to(worker)), "sha256": artifact["sha256"]}
                prediction_path = worker / identity / "predictions.npz"
                prediction_path.write_bytes(b"Only the saved hash is read; this fixture is not an array file.")
                artifacts["predictions.npz"] = {"path": str(prediction_path.relative_to(worker)),
                                                "sha256": file_hash(prediction_path)}
                predicted[identity, arm] = {"path": str(prediction_path), "sha256": file_hash(prediction_path)}
                reports.append({"status": "completed", "set": identity, "variant": "ordinary", "rows": 2,
                    "common_set_sha256": common_item["manifest"]["sha256"], "artifacts": artifacts,
                    "out_of_training_support_komi_halfpoints": [1] if arm.startswith("baseline") and identity.endswith("komi1") else []})
            summary["reports"] = reports
        elif name.startswith("matches-"):
            slots = [{"id": f"slot-{index}", "arm": arm, "profile_id": "weak_uniform" if offset < 2 else "capture_first",
                      "candidate_color": "B" if offset % 2 == 0 else "W"}
                     for index, (arm, offset) in enumerate((arm, offset)
                         for arm in ("conditioned", "unconditioned", "human_sl") for offset in range(4))]
            schedule = ref(worker / "schedule.json", {"slots": slots})
            records = {}
            for slot in (slots[0], slots[4], slots[8]):
                identity = slot["id"]
                sgf = worker / "matches" / (identity + ".sgf")
                sgf.parent.mkdir(exist_ok=True)
                sgf.write_text("(;GM[1]SZ[19]KM[6.5];B[aa];W[ba];B[])")
                record = {"slot": slot, "state": {"origin": "empty", "initial": {"board": ["." * 19] * 19,
                    "to_play": "B", "captured_black": 0, "captured_white": 0}, "moves": [["B", 0], ["W", 1], ["B", 361]]},
                    "actual_actions": 3, "sgf_file": sgf.name, "sgf_sha256": file_hash(sgf), "result": "U",
                    "status": "truncated", "reason": "synthetic cap", "behavior": {"unresolved": True},
                    "candidate_profile": {"conditioning_rank": "5k"}, "censored": True,
                    "candidate_passes": [{"move_index": 2, "verdict": "unjudgeable"}]}
                record_pin = ref(worker / "matches" / (identity + ".json"), record)["sha256"]
                records[identity] = {"record_sha256": record_pin, "sgf_sha256": file_hash(sgf)}
            authorization = ref(worker / "evidence_authorization.json",
                {"schedule_file_sha256": schedule["sha256"], "records": records})
            summary.update(status="incomplete", scheduled_games=12, recorded_games=3, records_replayed=3,
                schedule_file_sha256=schedule["sha256"], evidence_authorization_sha256=authorization["sha256"],
                unverified_slots=[], missing_slots=[slot["id"] for slot in slots if slot["id"] not in records])
        elif name == "analysis":
            comparisons, specs = {}, []
            for identity in review.SETS:
                contrasts = {}
                for first, second in ((review.ARMS[0], review.ARMS[1]), (review.ARMS[2], review.ARMS[3]),
                                      (review.ARMS[2], review.ARMS[0]), (review.ARMS[3], review.ARMS[1])):
                    key = first + "_minus_" + second
                    report = metrics(delta=True)
                    contrasts[key] = {"report": ref(worker / (identity + "-" + key + ".json"), report),
                                      "overall": report["overall"]}
                comparisons[identity] = contrasts
                mapping = {}
                for arm in review.ARMS:
                    key = identity + "-" + arm
                    config["inputs"][key] = predicted[identity, arm]
                    mapping[arm] = key
                specs.append({"id": identity, "prediction_inputs": mapping})
            summary["comparisons"] = comparisons
            config["parameters"]["common_sets"] = specs
        config_ref = ref(directory / "config.json", config)
        provenance_ref = ref(directory / "provenance.json", {"inputs": config["inputs"]})
        summary.update(config_file_sha256=config_ref["sha256"], provenance_sha256=provenance_ref["sha256"])
        stages[name] = {"directory": str(directory), "config": config_ref, "provenance": provenance_ref,
            "summary": ref(worker / "summary.json", summary), "wandb_url": "https://forge.coreweave.com/wandb/example/run",
            "job": {"status": "completed", "costs": {"wall_seconds": 2., "cpu_seconds": 1.}}}
        entries.append({"run_id": run.name + "-" + name,
                        "gpu_seconds": 5000. if name in review.ARMS else 100. if name.startswith("probe-") else 0.,
                        "cpu_seconds": 1., "wall_seconds": 2.,
                        "outcome": "completed"})
    charged = sum(entry["gpu_seconds"] for entry in entries)
    main = {"schema": "expanded_e2_pilot_v1", "status": "completed", "promotion": "not_promoted", "plan": plan_ref,
        "stages": stages, "matched_training": True,
        "training_decision": ref(run / "frozen-training.json", {"updates": 8, "probe_weights_reused": False}),
        "costs": {"elapsed_seconds": 40., "cpu_seconds": 30., "output_bytes": 10000,
            "ledger": {"charged_gpu_seconds": charged, "remaining_gpu_seconds": 86400 - charged,
                       "budget_breached": False, "state": {"entries": entries, "reservations": {}}}}}
    ref(run / "summary.json", main)
    return run, main


class ExpandedReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.main = fixture(self.root)

    def test_completed_results_retain_all_arms_sets_slots_costs_and_display_frames(self):
        prepared = json.loads(Path(self.main["stages"]["preparation"]["summary"]["path"]).read_text())
        common = prepared["common_sets"][0]
        manifest_path = Path(common["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source_ids"], list(reversed(common["spec"]["source_ids"])))
        data = review.collect(self.run)
        self.assertEqual(file_hash(manifest_path), common["manifest"]["sha256"])
        self.assertTrue(data["matched_training"])
        self.assertEqual(len(data["training"]), 4)
        self.assertEqual(len(data["games"]), 24)
        self.assertEqual(sum(game["worker_verified"] for game in data["games"]), 6)
        for identity, reports in data["evaluations"].items():
            self.assertEqual(len(reports), 5 if identity.endswith("komi13") else 4)
            self.assertEqual(reports[review.ARMS[0]]["metrics"]["overall"]["metrics"]["nll"]["uncertainty"]["status"], "inconclusive")
        game = data["games"][0]
        self.assertEqual(game["frames"], [[[], "B", 0, 0, 0], [[[0, "B"]], "W", 0, 0, 0],
                                         [[[1, "W"]], "B", 0, 0, 0], [[], "W", 0, 0, 1]])
        self.assertEqual(game["passes"][0]["verdict"], "unjudgeable")
        self.assertAlmostEqual(data["costs"]["prior_gpu_seconds"], 1246.177333)
        self.assertEqual(data["costs"]["run_gpu_seconds"], 20400)
        self.assertEqual(data["costs"]["gpu_seconds_by_stage"][review.ARMS[0]], 5000)
        self.assertEqual(data["costs"]["gpu_seconds_by_stage"]["probe-" + review.ARMS[0]], 100)
        self.assertFalse((self.root / "missing-dataset.parquet").exists())

    def test_duplicate_common_sources_are_rejected_even_when_membership_matches(self):
        main = copy.deepcopy(self.main)
        stage = main["stages"]["preparation"]
        path = Path(stage["summary"]["path"])
        prepared = json.loads(path.read_text())
        common = prepared["common_sets"][0]
        manifest_path = Path(common["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text())
        manifest["source_ids"].append(manifest["source_ids"][0])
        common["manifest"] = ref(manifest_path, manifest)
        stage["summary"] = ref(path, prepared)
        ref(self.run / "summary.json", main)
        with self.assertRaisesRegex(ValueError, "Frozen common set differs"):
            review.collect(self.run)

    def test_changed_plan_report_prediction_or_sgf_pins_are_rejected(self):
        paths = [self.root / "plan.json", self.run / "analysis" / "worker" / (review.SETS[0] + "-" + review.ARMS[0] + "_minus_" + review.ARMS[1] + ".json"),
                 self.run / ("evaluation-" + review.ARMS[0]) / "worker" / review.SETS[0] / "predictions.npz",
                 self.run / "matches-baseline" / "worker" / "matches" / "slot-0.sgf"]
        for path in paths:
            old = path.read_bytes()
            try:
                path.write_bytes(b"changed")
                with self.subTest(path=path), self.assertRaises(ValueError):
                    review.collect(self.run)
            finally:
                path.write_bytes(old)

    def test_display_allows_the_white_candidates_last_reply_at_action_801(self):
        state = {"origin": "empty", "initial": {"board": ["." * 19] * 19,
            "to_play": "B", "captured_black": 0, "captured_white": 0},
            "moves": [["B" if index % 2 == 0 else "W", 361] for index in range(801)]}
        self.assertEqual(len(review._display_frames(state, 801)), 802)
        state["moves"].append(["W", 361])
        with self.assertRaises(ValueError):
            review._display_frames(state, 802)

    def test_missing_top_or_stage_status_is_not_treated_as_completion(self):
        changed = copy.deepcopy(self.main)
        del changed["status"]
        ref(self.run / "summary.json", changed)
        with self.assertRaisesRegex(ValueError, "explicit status"):
            review.collect(self.run)
        changed = copy.deepcopy(self.main)
        stage = changed["stages"][review.ARMS[0]]
        path = Path(stage["summary"]["path"])
        summary = json.loads(path.read_text())
        del summary["status"]
        stage["summary"] = ref(path, summary)
        ref(self.run / "summary.json", changed)
        with self.assertRaisesRegex(ValueError, "explicit status"):
            review.collect(self.run)

    def test_failure_exports_missing_metrics_and_checked_partial_stage_without_inventing_results(self):
        failed = copy.deepcopy(self.main)
        failed.update(status="incomplete", reason="Synthetic worker failure", stages={"preparation": self.main["stages"]["preparation"]})
        failed.pop("training_decision")
        name = review.ARMS[0]
        artifact = ref(self.run / name / "worker" / "training_failure.json",
                       {"status": "failed", "completed_updates": 2, "reason": "Synthetic interrupted update"})
        failed["failed_stage"] = {"name": name, "directory": str(self.run / name),
                                  "artifacts": {"worker/training_failure.json": artifact}}
        path = self.run / "failure.json"
        ref(path, failed)
        result = review.collect(path)
        self.assertFalse(result["matched_training"])
        self.assertEqual(result["training"], {})
        self.assertEqual(result["games"], [])
        self.assertTrue(all(not value for value in result["evaluations"].values()))
        self.assertEqual(next(stage for stage in result["stages"] if stage["name"] == name)["status"], "failed")
        self.assertEqual(result["failure"]["artifacts"]["worker/training_failure.json"]["completed_updates"], 2)

    def test_export_escapes_script_data_and_refuses_to_overwrite_outputs(self):
        data = review.collect(self.run)
        data["reason"] = "</script><img src=x onerror=alert(1)>\u2028\u2029"
        page = review.render(data)
        self.assertNotIn(data["reason"], page)
        self.assertIn("\\u003c/script>", page)
        payload = page.split('<script id="expanded-data" type="application/json">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["reason"], data["reason"])
        self.assertEqual(page.count("<script "), 2)
        self.assertNotIn("__EXPANDED_DATA_JSON__", page)
        self.assertIn("connect-src 'none'", page)
        self.assertNotIn("innerHTML", page)
        output = self.root / "review.html"
        manifest = review.export(self.run, output)
        self.assertEqual(manifest["html_sha256"], file_hash(output))
        self.assertEqual((manifest["raw_dataset_reads"], manifest["checkpoint_reads"], manifest["inference_calls"]), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "unused"):
            review.export(self.run, output)

    def test_page_script_selectors_replay_notes_and_downloads_without_a_browser(self):
        node = shutil.which("node")
        if node is None:
            candidates = sorted((Path.home() / ".nvm/versions/node").glob("*/bin/node"))
            node = str(candidates[-1]) if candidates else None
        if node is None:
            self.skipTest("A local Node runtime is unavailable; browser QA remains separate")
        data = review.collect(self.run)
        page = review.render(data)
        script = page.split('<script id="expanded-script">', 1)[1].split("</script>", 1)[0]
        ids = re.findall(r'\bid="([^"]+)"', page.split('<script id="expanded-data"', 1)[0])
        harness = r'''
const assert = require('node:assert/strict');
class Element {
 constructor(tag){this.tagName=tag;this.children=[];this.attributes={};this.listeners={};this._value=undefined;this.textContent='';this.disabled=false;this.nodeType=1;}
 setAttribute(k,v){this.attributes[k]=v;if(k==='value')this._value=v;}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];if(this.tagName==='select')this._value=undefined;}
 addEventListener(name,handler){this.listeners[name]=handler;}
 get value(){return this._value??(this.tagName==='select'?this.children[0]?.value??'':'');}
 set value(value){this._value=String(value);}
 click(){this.listeners.click?.();}
 remove(){}
}
const elements=Object.fromEntries(IDS.map(id=>[id,new Element(id.includes('select')||['game-cohort','game-arm'].includes(id)?'select':'div')]));
elements['expanded-data']=new Element('script');elements['expanded-data'].textContent=JSON.stringify(DATA);
elements['game-cohort'].value='all';elements['game-arm'].value='all';elements['breakdown-select'].value='overall';
const document={getElementById:id=>{assert.ok(elements[id],id);return elements[id];},createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),body:new Element('body')};
const localStorage={data:new Map(),getItem(key){return this.data.get(key)??null;},setItem(key,value){this.data.set(key,value);}};
let downloads=0;URL.createObjectURL=()=>{downloads++;return 'blob:synthetic';};URL.revokeObjectURL=()=>{};
DATA.evaluations[DATA.sets[0]]['human-reference'].metrics.overall.metrics.nll.estimate={status:'positive_infinity',value:null};
Object.values(DATA.paired[DATA.sets[0]])[0].overall.metrics.nll.estimate={status:'negative_infinity',value:null};
DATA.evaluations[DATA.sets[0]]['combined-unconditioned'].metrics.overall.metrics.nll.estimate={status:'undefined',value:null};
elements['expanded-data'].textContent=JSON.stringify(DATA);
SCRIPT
const textOf=node=>[node.textContent,...node.children.map(textOf)].join(' ');
assert.ok(textOf(elements['metric-cards']).includes('+∞'));
assert.ok(textOf(elements['metric-cards']).includes('Undefined'));
assert.ok(textOf(elements['metric-table']).includes('+∞'));
assert.ok(textOf(elements['comparison-chart']).includes('+∞ · outside finite scale'));
assert.ok(textOf(elements['paired-table']).includes('−∞'));
const trainingText=textOf(elements['training-table']),probeText=textOf(elements['probe-table']);
assert.ok(trainingText.includes('1.389 h'));
assert.ok(!trainingText.includes('0.028 h'));
assert.ok(probeText.includes('0.028 h'));
assert.equal(elements['set-select'].children.length,4);
assert.equal(elements['game-select'].children.length,24);
elements['game-arm'].value='conditioned';elements['game-arm'].listeners.change();assert.equal(elements['game-select'].children.length,8);
elements['game-cohort'].value='baseline';elements['game-cohort'].listeners.change();assert.equal(elements['game-select'].children.length,4);
elements['next-move'].click();assert.equal(elements['move-slider'].value,'1');
elements['game-notes'].value='Inspect move 1';elements['game-notes'].listeners.input();assert.equal(localStorage.data.size,1);
elements['download-sgf'].click();elements['download-notes'].click();assert.equal(downloads,2);
elements['set-select'].value=DATA.sets[1];elements['set-select'].listeners.change();
assert.ok(elements['set-context'].textContent.includes('outside training support'));
elements['breakdown-select'].value='by_rank';elements['breakdown-select'].listeners.change();
assert.ok(elements['metric-table'].children.length);
assert.ok(elements['stage-table'].children.length);
console.log('selectors, replay, local notes, downloads, and breakdowns passed');
'''.replace("IDS", json.dumps(ids), 1).replace("SCRIPT", script, 1)
        harness = "const DATA = " + json.dumps(data) + ";\n" + harness
        path = self.root / "ui-check.cjs"
        path.write_text(harness)
        result = subprocess.run([node, str(path)], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
