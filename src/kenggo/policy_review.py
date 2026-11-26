"""Offline review of saved policy-only tuning and fixed comparisons."""

from copy import deepcopy
import html
import json
import math
from pathlib import Path

from kenggo.checkpoints import save_json
from kenggo.continuation_data import path, read, reference
from kenggo.learning_review import DEFAULT_HISTORY, build, insert, source_pins


VALUE_NOTE = ("Value loss weight is zero in this study. The inherited value head is not trained "
              "for outcome prediction during these trials; its outputs can change through shared features. Value metrics "
              "are diagnostic only: they do not select, guard, or promote a policy.")


def _take(value, keys):
    return {key: deepcopy(value[key]) for key in keys if key in value}


def _metric(value):
    result = _take(value, ("status", "estimate", "counts", "masked_moves", "aggregate_ranks", "missing_ranks"))
    if "uncertainty" in value:
        result["uncertainty"] = _take(value["uncertainty"], ("status", "interval95", "method", "reasons"))
    return result


def _report(value):
    result = _take(value, ("aggregate_ranks", "expected_outputs", "excluded_rows"))
    result["overall"] = {"metrics": {key: _metric(metric)
        for key, metric in value.get("overall", {}).get("metrics", {}).items()
        if key in ("nll", "top1", "value_mse")}}
    return result


def _epoch(value):
    if "value_loss_weight" in value and value["value_loss_weight"] != 0:
        raise ValueError("Policy-only review cannot describe nonzero value-loss training")
    result = _take(value, ("round", "sampler_step", "lifetime_updates", "new_raw_best",
        "meaningful_improvement", "learning_rate_used", "learning_rate_next", "value_loss_weight"))
    for name in ("training_monitor", "validation"):
        result[name] = _take(value.get(name, {}), ("nll", "top1", "value_mse", "rows", "policy_count",
            "value_count", "aggregate_ranks", "diagnostic_sha256", "excluded_rank_rows",
            "evaluated_rows", "weighting"))
    return result


def _presentations(progress):
    blocks = progress.get("epoch_metrics", [])
    counts = [block.get("presented_examples") for block in blocks]
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    partial = progress.get("partial_round", {}).get("presented_examples")
    if partial is None:
        if progress.get("sampler_step") != (blocks[-1].get("next_step") if blocks else 0):
            return None
        partial = 0
    return sum(counts) + partial if type(partial) is int and partial >= 0 else None


def _trial(trial):
    attempts, measurements, checkpoints = [], {}, {}
    last, latest_best, training_rows, uncertain = {}, None, None, 0
    for item in trial.get("training", []):
        stage = item["stage"]
        worker = read(stage["summary"]) if stage.get("summary") else {}
        progress = worker.get("progress") or {}
        if progress:
            last = worker
        if worker.get("training_rows") is not None:
            training_rows = worker["training_rows"]
        elif training_rows is None and stage.get("config"):
            training_rows = read(stage["config"])["parameters"].get("training_rows")
        unknown_updates = worker.get("uncertain_updates", 0)
        if type(unknown_updates) is not int or unknown_updates < 0:
            raise ValueError("Uncertain update attempts require a nonnegative count")
        uncertain += unknown_updates
        attempt = _take(stage, ("status", "summary", "summary_kind", "artifacts"))
        attempt.update(epoch=item.get("epoch"),
                       reason=stage.get("job", {}).get("reason") or stage.get("reason") or worker.get("reason"),
                       uncertain_updates=unknown_updates,
                       progress=_take(progress, ("sampler_step", "completed_rounds", "partial_round")))
        attempts.append(attempt)
        for commit_ref in worker.get("commits", []):
            commit = read(commit_ref)
            point = _epoch(read(commit["metrics"]))
            epoch = point["round"]
            if epoch in measurements and measurements[epoch] != point:
                raise ValueError("A trial has conflicting committed epoch measurements")
            measurements[epoch] = point
            checkpoints[commit["checkpoint"]["sha256"]] = point
            latest_best = commit.get("best_checkpoint") or latest_best
        latest_best = worker.get("best_checkpoint") or latest_best
    measured = [measurements[key] for key in sorted(measurements)]
    selected = trial.get("selected") or latest_best
    selected_metrics = checkpoints.get(selected["sha256"]) if selected else None
    if measured and selected_metrics is None:
        raise ValueError("The selected checkpoint lacks its committed validation measurement")
    progress = last.get("progress") or {}
    details = _take(trial, ("id", "parameters", "status", "stop_reason"))
    details.update(attempts=attempts, selected=selected, training_rows=training_rows,
        uncertain_update_attempts=uncertain,
        completed_updates=progress.get("sampler_step"), presented_examples=_presentations(progress),
        committed_epochs=len([point for point in measured if point["round"] > 0]),
        learning_rates=[_take(point, ("round", "sampler_step", "learning_rate_used", "learning_rate_next",
                                    "value_loss_weight")) for point in measured])
    if not measured:
        details["measurement_status"] = "No committed checkpoint measurements are available."
        return details, None
    details.update(selected_round=selected_metrics["round"], selected_sampler_step=selected_metrics["sampler_step"],
        selected_lifetime_updates=selected_metrics["lifetime_updates"], measurement_status="committed")
    committed_steps = {point["sampler_step"] for point in measured}
    online = [_take(block, ("epoch", "next_step", "updates", "presented_examples", "policy_count",
                           "value_count", "policy_loss", "value_loss"))
              for block in progress.get("epoch_metrics", []) if block.get("next_step") in committed_steps]
    training = {"status": trial.get("status"), "training_rows": details["training_rows"],
        "completed_updates": details["completed_updates"], "presented_examples": details["presented_examples"],
        "epoch_metrics": online, "validation_metrics": measured, "training_window": "training interval",
        "final_checkpoint": selected, **_take(details, ("selected_sampler_step", "selected_round",
                                                       "selected_lifetime_updates"))}
    return details, training


def _preparation(ref):
    if not ref:
        return None
    saved = read(ref)
    result = {key: deepcopy(value) for key, value in saved.items()
              if key not in ("train_monitor", "epoch_validation", "assessments")}
    result["evidence"] = ref
    if saved.get("selection"):
        selection = saved["selection"]
        if set(selection) == {"path", "sha256"}:
            result["selection_evidence"] = selection
            selection = read(selection)
        result["selection"] = _take(selection, ("schema", "seed", "per_cell", "ranks", "komi_halfpoints",
            "source_ids_sha256", "unique_games", "expected_rows", "excluded_rank_rows", "available_games",
            "available_rows", "accounts", "maximum_games_per_account", "selection_uses_targets_or_predictions"))
        result["selection"]["cells"] = []
        for cell in selection.get("cells", []):
            compact = _take(cell, ("rank", "komi_halfpoints", "available_games", "available_mover_accounts",
                                  "target_games", "selected_games", "below_requested_quota"))
            if "selected_mover_accounts" in cell:
                compact["selected_mover_account_count"] = len(cell["selected_mover_accounts"])
            result["selection"]["cells"].append(compact)
    diagnostics = _take(saved, ("train_monitor", "epoch_validation"))
    diagnostics.update(saved.get("assessments", {}))
    result["diagnostics"] = {}
    for name, diagnostic_ref in diagnostics.items():
        if not diagnostic_ref:
            continue
        diagnostic = read(diagnostic_ref)
        record = _take(diagnostic, ("purpose", "role", "aggregate_ranks", "source_ids", "normalizer_sha256"))
        record.update(evidence=diagnostic_ref, rows=len(diagnostic["rows"]))
        result["diagnostics"][name] = record
    return result


def _comparison(ref):
    if not ref:
        return None
    saved = read(ref)
    result = _take(saved, ("schema", "status", "selected_trial_id", "baseline_trial_id", "notes", "reason"))
    result.update(evidence=ref, comparisons={})
    for trial, opponents in saved.get("comparisons", {}).items():
        result["comparisons"][trial] = {}
        for opponent, scopes in opponents.items():
            result["comparisons"][trial][opponent] = {}
            for scope, report in scopes.items():
                record = _take(report, ("kind", "aggregate_ranks", "rows", "excluded_rows", "uncertainty_method"))
                record["metrics"] = {key: _metric(value) for key, value in report.get("metrics", {}).items()
                                     if key in ("nll", "top1", "value_mse")}
                result["comparisons"][trial][opponent][scope] = record
    return result


def readout(summary_ref):
    summary = read(summary_ref)
    if summary.get("schema") != "e2_policy_study_v1":
        raise ValueError("Expected a policy-study summary")
    trials, training, evaluations, assessments = [], {}, {}, {}
    for trial in summary.get("trials", []):
        details, measured = _trial(trial)
        if any(prior["id"] == details["id"] for prior in trials):
            raise ValueError("Policy trial IDs must be unique")
        trials.append(details)
        if measured is not None:
            training[trial["id"]] = measured
    for name, assessment_ref in summary.get("assessments", {}).items():
        assessment = read(assessment_ref)
        assessments[name] = {"evidence": assessment_ref,
            **_take(assessment, ("status", "reason", "scope", "checkpoint", "reports"))}
        for scope, report_ref in assessment.get("reports", {}).items():
            report = read(report_ref)
            scoped, full = _report(report["report"]), _report(report["all_rank_report"])
            scoped["excluded_rows"] = report["excluded_rank_rows"]
            full["review_scope"] = scoped
            evaluations.setdefault(scope, {})[name] = {
                "metrics": full, "rows": report["evaluated_rows"], "evidence": report_ref}
    return {"bundle_id": summary_ref["sha256"], "input": summary_ref,
        "title": "Policy-only tuning on the existing training data", "default_scope": "review_scope",
        "description": "Continue the saved parent on the existing four-million-move training set, "
                       "with a larger fixed stopping-validation sample and policy-only optimization.",
        "notes": [VALUE_NOTE,
            "Training-monitor and stopping-validation curves use separate fixed whole-game samples. "
            "The fixed development comparisons use another saved sample; final and confirmation targets remain closed.",
            "Online losses average changing models and learning batches. Checkpoint metrics use fixed models, "
            "equal rank/game-side/valid-position weighting and saved policy/outcome masks.",
            "Earlier runs retain their own samples and training amounts. Their differences are historical context; "
            "only the paired comparisons below measure a difference on the same positions.",
            "Every new trial uses policy-only loss. There is no equally trained joint-loss control; "
            "differences from the parent combine further training, loss and optimizer changes.",
            "This exploratory study does not establish complete-game behavior, playing strength, or a human rank."],
        **_take(summary, ("status", "reason", "plan", "parent_checkpoint", "launch_utc", "deadline_utc",
                          "limits", "costs", "costs_scope", "selected_trial_id", "failed_attempts")),
        "training": training, "evaluations": evaluations, "trials": trials, "assessments": assessments,
        "preparation": _preparation(summary.get("preparation")),
        "comparison": _comparison(summary.get("comparison"))}


def _number(value, digits=4):
    return f"{value:,.{digits}f}" if type(value) in (int, float) and math.isfinite(value) else "Not recorded"


def _text(value):
    return html.escape(str(value)) if value is not None else "Not recorded"


def _table(headers, rows):
    return '<div class="table"><table><thead><tr>' + "".join(
        f'<th scope="col">{_text(header)}</th>' for header in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def _trial_page(trial):
    params = trial.get("parameters", {})
    settings = "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in params.items()) or "Settings not recorded"
    if trial["measurement_status"] == "committed":
        checkpoint = (f"Selected interval {trial['selected_round']} at {trial['selected_sampler_step']:,} trial updates. "
                      f"{trial['committed_epochs']} training intervals have committed measurements.")
    else:
        checkpoint = trial["measurement_status"] + " No curve point or validated checkpoint selection is inferred."
    rates = [[point.get("round"), point.get("sampler_step"), point.get("learning_rate_used"),
              point.get("learning_rate_next")] for point in trial["learning_rates"]]
    attempts = [[item.get("epoch"), item.get("status"), item.get("reason"),
                 item["progress"].get("sampler_step")] for item in trial["attempts"]]
    return (f"<article><h3>{_text(trial['id'])}</h3><p>{_text(settings)}.</p>"
        f"<p>Continues the same saved parent; {_number(trial['training_rows'], 0)} fitting moves. "
        f"Recorded training: {_number(trial['completed_updates'], 0)} trial updates and "
        f"{_number(trial['presented_examples'], 0)} move presentations. Status: {_text(trial.get('status'))}. "
        f"Update attempts with an uncertain outcome: {_number(trial['uncertain_update_attempts'], 0)}. "
        f"Stop reason: {_text(trial.get('stop_reason'))}.</p><p>{_text(checkpoint)}</p>"
        + (_table(["Interval", "Trial update", "Learning rate used", "Learning rate next"], rates)
           if rates else "<p>Learning rates at committed checkpoints are not recorded.</p>")
        + (_table(["Attempt interval", "Status", "Reason", "Recorded trial update"], attempts)
           if attempts else "<p>No training attempt is recorded for this trial.</p>") + "</article>")


def _comparison_page(comparison):
    rows = []
    for trial, opponents in (comparison or {}).get("comparisons", {}).items():
        for opponent, scopes in opponents.items():
            for scope, report in scopes.items():
                for name, metric in report.get("metrics", {}).items():
                    estimate, uncertainty = metric.get("estimate", {}), metric.get("uncertainty", {})
                    value = (_number(estimate.get("value")) if metric.get("status") == "available"
                             and estimate.get("status") == "finite" else "Unavailable")
                    interval = uncertainty.get("interval95")
                    interval_text = (f"[{_number(interval[0])}, {_number(interval[1])}]"
                                     if uncertainty.get("status") == "available" and interval else
                                     uncertainty.get("status", "Unavailable"))
                    support = (f"Ranks: {', '.join(report.get('aggregate_ranks', [])) or 'Not recorded'}; "
                               f"counts: {json.dumps(metric.get('counts', {}))}; "
                               f"masked moves: {metric.get('masked_moves', 'Not recorded')}; "
                               f"excluded rows: {report.get('excluded_rows', 'Not recorded')}; "
                               f"missing ranks: {', '.join(metric.get('missing_ranks', [])) or 'None recorded'}")
                    if uncertainty.get("reasons"):
                        interval_text += ": " + ", ".join(uncertainty["reasons"])
                    rows.append([trial, opponent, scope, name + (" · diagnostic only" if name == "value_mse" else ""),
                                 value, interval_text, support])
    heading = (f"<p>Comparison status: {_text((comparison or {}).get('status'))}. "
               f"Baseline trial: {_text((comparison or {}).get('baseline_trial_id'))}.</p>")
    return heading + (_table(["Trial", "Reference", "Fixed sample", "Metric", "Trial minus reference",
                             "95% interval / support", "Sample and masks"], rows) if rows else
                      "<p>No fixed comparison measurements are available.</p>")


def _preparation_page(preparation):
    if not preparation:
        return "<p>Preparation evidence is unavailable.</p>"
    rows = [[name, record.get("role"), len(record["source_ids"]) if "source_ids" in record else None,
             record.get("rows"), ", ".join(record.get("aggregate_ranks", [])) or "Not recorded"]
            for name, record in preparation["diagnostics"].items()]
    page = (_table(["Fixed sample", "Data role", "Whole games", "Positions", "Review ranks"], rows)
            if rows else "<p>No prepared diagnostic samples are recorded.</p>")
    cells = preparation.get("selection", {}).get("cells", [])
    if cells:
        page += "<h3>Stopping-validation coverage</h3>" + _table(
            ["Rank", "Komi", "Available games", "Selected games", "Selected mover accounts", "Below requested quota"],
            [[cell.get("rank"), cell["komi_halfpoints"] / 2 if "komi_halfpoints" in cell else None,
              cell.get("available_games"), cell.get("selected_games"), cell.get("selected_mover_account_count"),
              cell.get("below_requested_quota")] for cell in cells])
    return page


def export(summary_ref, output, *, history_path=DEFAULT_HISTORY):
    output = Path(output)
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("Publish a new review version; preserve saved pages and manifests")
    data = readout(summary_ref)
    history_ref = None
    if history_path is DEFAULT_HISTORY:
        history_ref = read(data["plan"])["inputs"]["review_history"]
        read(history_ref)
        history_path = path(history_ref)
    learning = build(data, history_path)
    if history_ref is not None and learning["catalog"]["sha256"] != history_ref["sha256"]:
        raise ValueError("Review history differs from the study's pinned catalog")
    learning["runs"][-1].update(description=data["description"], notes=data["notes"])
    costs = data.get("costs", {})
    def hours(seconds):
        return _number(seconds / 3600, 2) if type(seconds) in (int, float) else "Not recorded"
    trained = any((trial["completed_updates"] or 0) > 0 for trial in data["trials"])
    training_status = ("Saved committed checkpoints appear in the curves; unfinished attempts remain listed below."
        if data["training"] else "No committed training or validation curve is available.")
    if not trained:
        training_status += " No new training updates are recorded; epoch-zero evaluation is not new training."
    selected = data.get("selected_trial_id")
    cost_scope = (f"<p><strong>Cost scope:</strong> {_text(data['costs_scope'])}. "
        f"<a href=\"{_text(str(path(summary_ref).parent / 'summary.json'))}\">Final study cost record, including publication</a>.</p>"
        if data.get("costs_scope") else "")
    raw = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E2 · Policy-only tuning</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#192f35;background:#f6f8f6;margin:0}}main,header{{max-width:1100px;margin:auto;padding:24px}}h1{{font-size:30px;line-height:1.2}}section,article{{background:white;padding:20px;border:1px solid #d6dfdc;border-radius:10px;margin:20px 0}}a{{color:#185e57;overflow-wrap:anywhere}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d6dfdc;text-align:left;vertical-align:top}}th{{background:#f3f7f5}}textarea{{box-sizing:border-box;width:100%;min-height:130px;font:inherit;padding:10px}}button{{font:inherit;padding:10px 15px;cursor:pointer}}li{{margin:9px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto}}
</style><header><p>E2 · Human move prediction · kenggo-ydo.19</p><h1>Policy-only tuning on the existing data</h1>
<p>{_text(data['description'])}</p><p><strong>Status:</strong> {_text(data.get('status'))}. {_text(data.get('reason'))}</p>
<section><h2>Overall result</h2><p>Selected trial: {_text(selected) if selected else 'No trial selected in the saved summary'}.</p>
<p>{_text(training_status)}</p><p>{_text(VALUE_NOTE)}</p>
<p>Recorded usage: {hours(costs.get('elapsed_seconds'))} elapsed hours, {hours(costs.get('cpu_seconds'))} CPU hours and {hours(costs.get('ledger', {}).get('charged_gpu_seconds'))} allocated GPU hours. Deadline: {_text(data.get('deadline_utc'))}.</p>{cost_scope}
<p>Move prediction is separate from playing strength and complete-game behavior. No new rank qualification or complete-game panel is claimed.</p>
<p><a href="#review-actions">Go to the three review steps</a> · <a href="#trials">Inspect every trial</a> · <a href="#comparisons">Inspect fixed comparisons</a></p></section></header>
<main><section id="review-actions"><h2>What to review</h2><ol>
<li><strong>Check policy learning for kenggo-ydo.19.</strong> Compare training-monitor and stopping-validation NLL and move agreement, the selected interval, and saved learning rates. This shows whether the trial improved, leveled off, or stopped before finishing.</li>
<li><strong>Check fixed comparisons for E2.</strong> Compare each trial with the parent and baseline on the saved common positions. Inspect rank coverage, masks and uncertainty; use earlier runs as context when their samples differ. Value results are diagnostic only.</li>
<li><strong>Record the next step for kenggo-ydo.19.</strong> Judge whether policy tuning is worth continuing from the saved policy evidence and cost. Record partial work and remaining complete-game limits; a better move predictor is not a qualified playing opponent.</li></ol>
<p><a href="{_text(str(path(summary_ref)))}">Saved study summary and evidence paths</a></p></section>
<section id="trials"><h2>Every trial and its training amount</h2>{''.join(_trial_page(trial) for trial in data['trials']) or '<p>No trial is recorded. Training curves and new model metrics are not applicable.</p>'}
<p>A recovery checkpoint without a committed validation measurement is not a new curve point or a validated checkpoint choice. Missing measurements stay unavailable.</p></section>
<section id="comparisons"><h2>Fixed development comparisons</h2><p>Differences are trial minus reference on common positions. Lower NLL and higher move agreement are better. Agreement differences use proportions. Value MSE is diagnostic only.</p>
{_comparison_page(data['comparison'])}</section>
<section><h2>Data, limits and incomplete work</h2><p>Checkpoint metrics keep their recorded rank scope and masks. Training-monitor games come from fitting data; stopping validation and fixed comparisons have separate whole-game roles. Outcome diagnostics exclude masked results. The saved evidence below retains sample identities, coverage and failures.</p>
{_preparation_page(data['preparation'])}
<details><summary>Preparation, budget, failed attempts and comparison notes</summary><pre id="details"></pre></details></section>
<section><h2>Your notes</h2><p>These notes stay in this browser until you download them.</p><textarea id="notes" aria-label="Review notes"></textarea><p><button id="download">Download notes</button></p></section></main>
<script id="expanded-data" type="application/json">{raw}</script><script>
const data=JSON.parse(document.getElementById('expanded-data').textContent);
document.getElementById('details').textContent=JSON.stringify({{preparation:data.preparation,assessments:data.assessments,limits:data.limits,costs:data.costs,failed_attempts:data.failed_attempts,comparison:data.comparison}},null,2);
const notes=document.getElementById('notes'),key='e2-policy-notes-'+data.bundle_id;
try{{notes.value=localStorage.getItem(key)||''}}catch(e){{}}
notes.addEventListener('input',()=>{{try{{localStorage.setItem(key,notes.value)}}catch(e){{}}}});
document.getElementById('download').addEventListener('click',()=>{{const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify({{bead:'kenggo-ydo.19',evidence:data.input,notes:notes.value}},null,2)],{{type:'application/json'}}));link.download='e2-policy-review-notes.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}});
</script></html>'''
    output.write_text(insert(page, learning))
    save_json(manifest, {"source": summary_ref, "page": reference(output), "history": learning["catalog"],
                         "history_sources": learning["history_sources"], "code": [reference(__file__), *source_pins()]})
    return reference(output)
