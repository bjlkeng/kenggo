"""Offline owner readout from saved progressive-training evidence."""

from copy import deepcopy
import html
import json
from pathlib import Path

from kenggo.checkpoints import save_json
from kenggo.continuation_data import read, reference
from kenggo.learning_review import DEFAULT_HISTORY, build, insert


def readout(summary_ref):
    summary = read(summary_ref)
    training, evaluations, descriptions, tiers = {}, {}, [], []
    for tier in summary["tiers"]:
        number = tier["tier"]
        descriptions.append(f"Round {number}: {tier['previous_rows']:,} old fitting moves; "
                            f"{tier.get('training_rows', 0):,} larger-set moves. "
                            "Both arms continued the same saved parent with equal update opportunities.")
        for arm in ("control", "candidate"):
            stages = [item for item in tier["training"] if item["arm"] == arm]
            measured = []
            last = None
            for item in stages:
                worker = read(item["stage"]["summary"])
                if worker.get("progress"):
                    last = worker
                for commit in worker.get("commits", []):
                    measured.append(read(read(commit)["metrics"]))
            if not last or not measured:
                continue
            selected = tier.get("selected", {}).get(arm, last.get("best_checkpoint"))
            selected_metrics = None
            for item in stages:
                worker = read(item["stage"]["summary"])
                for ref in worker.get("commits", []):
                    commit = read(ref)
                    if selected and commit["checkpoint"]["sha256"] == selected["sha256"]:
                        selected_metrics = read(commit["metrics"])
            if selected_metrics is None:
                raise ValueError("The selected checkpoint lacks its committed validation measurement")
            progress = last["progress"]
            interruption = tier.get("incomplete_interval", {})
            status = interruption.get("status", "partial") if interruption.get("arm") == arm else stages[-1]["stage"]["status"]
            training[f"tier{number}-{arm}"] = {
                "status": status, "training_rows": last["training_rows"],
                "completed_updates": progress["sampler_step"],
                "presented_examples": progress["sampler_step"] * 128,
                "epoch_metrics": progress["epoch_metrics"], "validation_metrics": measured,
                "training_window": "matched interval (one larger-data epoch)",
                "final_checkpoint": selected,
                "selected_sampler_step": selected_metrics["sampler_step"],
                "selected_round": selected_metrics["round"],
                "selected_lifetime_updates": selected_metrics["lifetime_updates"]}
        for arm, ref in tier["assessments"].items():
            for identity, report_ref in read(ref)["reports"].items():
                report = read(report_ref)
                scoped = deepcopy(report["report"])
                scoped["excluded_rows"] = report["excluded_rank_rows"]
                full = deepcopy(report["all_rank_report"])
                full["review_scope"] = scoped
                evaluations.setdefault(identity, {})[f"tier{number}-{arm}"] = {
                    "metrics": full, "rows": report["evaluated_rows"], "evidence": report_ref}
        tiers.append({key: tier[key] for key in ("tier", "target_rows", "previous_rows", "training_rows",
            "status", "training_stop", "completed_intervals", "incomplete_interval", "comparison_status",
            "decision", "comparison", "dataset") if key in tier})
        tiers[-1]["attempts"] = [{"arm": item["arm"], "epoch": item["epoch"],
            "status": item["stage"]["status"], "summary": item["stage"]["summary"],
            "reason": item["stage"].get("job", {}).get("reason")}
            for item in tier["training"]]
        if "comparison" in tier:
            tiers[-1]["paired_metrics"] = {
                opponent: {identity: {name: {"status": metric["status"], "estimate": metric["estimate"],
                    "interval95": metric["uncertainty"].get("interval95"),
                    "uncertainty_status": metric["uncertainty"]["status"]}
                    for name, metric in report["metrics"].items()}
                    for identity, report in reports.items()}
                for opponent, reports in read(tier["comparison"])["comparisons"].items()}
    return {"bundle_id": summary_ref["sha256"], "input": summary_ref,
            "title": "More data and validation-based stopping", "default_scope": "review_scope",
            "description": "Progressive supervised training with an old-data control at each larger data tier.",
            "notes": descriptions + [
                "Stopping-validation games differ between data tiers. Fixed comparison games stay the same.",
                "Training-monitor curves use a small fixed sample from the original fitting set; online losses use all presented batches.",
                "This study reuses the network architecture and adds training. Earlier Human SL results remain reference context.",
                "Final and confirmation targets remain closed. No new complete-game panel or rank qualification was run."],
            "training": training, "evaluations": evaluations, "tiers": tiers,
            "status": summary["status"], "reason": summary["reason"], "costs": summary["costs"],
            "deadline_utc": summary["deadline_utc"], "failed_attempts": summary.get("failed_attempts", [])}


def export(summary_ref, output, *, history_path=DEFAULT_HISTORY):
    data = readout(summary_ref)
    learning = build(data, history_path)
    learning["runs"][-1].update(description=data["description"], notes=data["notes"])
    esc = html.escape
    sections = []
    for tier in data["tiers"]:
        decision = tier.get("decision", {})
        sections.append(f"<li><strong>Round {tier['tier']}</strong>: {tier['previous_rows']:,} → "
            f"{tier.get('training_rows', 0):,} fitting moves. {esc(tier.get('training_stop', tier['status']).replace('_', ' '))}. "
            + esc("; ".join(decision.get("reasons", ["The fixed comparison is not yet available."]))) + "</li>")
        if "incomplete_interval" in tier:
            interrupted = tier["incomplete_interval"]
            outcome = ("The fixed comparison uses the last completed pair."
                       if tier.get("comparison_status") == "completed" else "The fixed comparison did not finish.")
            sections.append(f"<li><strong>Interrupted interval {interrupted['epoch']} · "
                f"{esc(interrupted['arm'])}:</strong> {esc(interrupted['reason'])}. "
                f"{outcome} The unfinished attempt remains in the evidence.</li>")
    costs = data["costs"]
    gpu = costs["ledger"]["charged_gpu_seconds"]
    raw = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E2 · More data and validation</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#192f35;background:#f6f8f6;margin:0}}main,header{{max-width:1100px;margin:auto;padding:24px}}h1{{font-size:30px;line-height:1.2}}section{{background:white;padding:20px;border:1px solid #d6dfdc;border-radius:10px;margin:20px 0}}a{{color:#185e57;overflow-wrap:anywhere}}textarea{{box-sizing:border-box;width:100%;min-height:130px;font:inherit;padding:10px}}button{{font:inherit;padding:10px 15px;cursor:pointer}}li{{margin:9px 0}}code{{overflow-wrap:anywhere}}
</style><header><p>E2 · Human move prediction · kenggo-ydo.16</p><h1>More data, with validation during training</h1>
<p>Each round compares a model trained on more games with a model that continues on the old data. Both start from the same checkpoint. Separate validation games choose when to stop.</p>
<p><strong>Status:</strong> {esc(data['status'].replace('_', ' '))}. {esc(data['reason'] or 'Work is still running.')}</p>
<section><h2>Overall result</h2><ul>{''.join(sections) or '<li>No complete training round is available yet.</li>'}</ul>
<p>Used {costs['elapsed_seconds']/3600:.2f} elapsed hours, {costs['cpu_seconds']/3600:.2f} CPU hours and {gpu/3600:.2f} allocated GPU hours. The single deadline is {esc(data['deadline_utc'])}.</p>
<p>Better move prediction does not establish stronger or complete play. Earlier premature passing and the rank/value tradeoff remain open in E2.</p>
<p><a href="#review-actions">Go to the three review steps</a></p></section></header>
<main><section id="review-actions"><h2>What to review</h2><ol><li><strong>Check the curves for .16.</strong> Compare training-monitor and stopping-validation losses, and inspect the selected checkpoint. This shows whether training leveled off or the budget stopped it.</li>
<li><strong>Compare each larger-data model with its old-data control and parent.</strong> Use the fixed 12k–1d development metrics and the paired evidence below. This separates added-data benefit from extra training.</li>
<li><strong>Record a next-step preference for E2.</strong> If data no longer helps, inspect whether training or validation loss has stalled. Poor fitting suggests optimization or capacity; a growing gap suggests overfitting or data mismatch. Passing and outcome prediction still need their own investigation.</li></ol>
<p><a href="{esc(str(Path(summary_ref['path']).resolve()), quote=True)}">Saved experiment summary and evidence paths</a></p>
<details><summary>Paired comparisons and saved run details</summary><pre id="details" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details></section>
<section><h2>Your notes</h2><p>These notes stay in this browser until you download them.</p><textarea id="notes" aria-label="Review notes"></textarea><p><button id="download">Download notes</button></p></section></main>
<script id="expanded-data" type="application/json">{raw}</script><script>
const data=JSON.parse(document.getElementById('expanded-data').textContent);
document.getElementById('details').textContent=JSON.stringify({{tiers:data.tiers,failed_attempts:data.failed_attempts}},null,2);
const notes=document.getElementById('notes'),key='e2-progressive-notes-'+data.bundle_id;
try{{notes.value=localStorage.getItem(key)||''}}catch(e){{}}
notes.addEventListener('input',()=>{{try{{localStorage.setItem(key,notes.value)}}catch(e){{}}}});
document.getElementById('download').addEventListener('click',()=>{{const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify({{bead:'kenggo-ydo.16',evidence:data.input,notes:notes.value}},null,2)],{{type:'application/json'}}));link.download='e2-progressive-review-notes.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}});
</script></html>'''
    output = Path(output)
    if output.exists():
        raise FileExistsError("Publish a new review version; preserve saved pages")
    output.write_text(insert(page, learning))
    save_json(output.with_suffix(".json"), {"source": summary_ref, "page": reference(output),
        "history": learning["catalog"], "history_sources": learning["history_sources"]})
    return reference(output)
