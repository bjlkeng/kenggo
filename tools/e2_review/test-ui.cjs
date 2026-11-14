"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "template.html"), "utf8");
const script = html.match(/<script id="review-script">([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
assert.equal(html.split("__PILOT_DATA_JSON__").length, 2);
assert.doesNotMatch(script, /innerHTML|XMLHttpRequest|\bfetch\s*\(/);
assert.match(html, /connect-src 'none'/);
assert.match(html, /html, body \{ margin: 0; padding: 0; background: transparent; \}/);

class Element {
  constructor(tag, document) {
    this.tagName = tag.toLowerCase(); this.ownerDocument = document;
    this.nodeType = 1; this.children = []; this.attrs = {}; this.listeners = {};
    this._text = ""; this._value = ""; this.disabled = false;
  }
  setAttribute(key, value) { this.attrs[key] = String(value); if (key === "value") this._value = String(value); }
  getAttribute(key) { return this.attrs[key]; }
  get id() { return this.attrs.id; }
  set id(value) { this.attrs.id = value; }
  get className() { return this.attrs.class || ""; }
  set className(value) { this.attrs.class = value; }
  get value() { return this._value; }
  set value(value) { this._value = String(value); }
  get childElementCount() { return this.children.length; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  set textContent(text) { this._text = String(text); this.children = []; }
  append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
  replaceChildren(...children) { for (const child of this.children) child.parent = null; this._text = ""; this.children = []; this.append(...children); }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  dispatch(name) { for (const callback of this.listeners[name] || []) callback({target: this}); }
  click() {
    if (this.disabled) return;
    if (this.tagName === "a") this.ownerDocument.downloads.push({name: this.attrs.download, url: this.attrs.href, blob: this.ownerDocument.blobs.get(this.attrs.href)});
    this.dispatch("click");
  }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  find(predicate) { if (predicate(this)) return this; for (const child of this.children) { const found = child.find(predicate); if (found) return found; } return null; }
  all(predicate) { return [...(predicate(this) ? [this] : []), ...this.children.flatMap(child => child.all(predicate))]; }
}
class Document {
  constructor() { this.body = new Element("body", this); this.blobs = new Map(); this.downloads = []; }
  createElement(tag) { return new Element(tag, this); }
  createElementNS(namespace, tag) { return this.createElement(tag); }
  getElementById(id) { return this.body.find(element => element.id === id); }
}
function storage(seed = new Map(), fail = false) {
  return {
    seed,
    getItem(key) { if (fail) throw Error("storage denied"); return seed.get(key) ?? null; },
    setItem(key, value) { if (fail) throw Error("storage denied"); seed.set(key, value); },
    removeItem(key) { if (fail) throw Error("storage denied"); seed.delete(key); }
  };
}
function launch(payload, saved = storage()) {
  const document = new Document();
  for (const [tag, id] of [["h1", "title"], ["p", "summary"], ["button", "export-notes"], ["span", "storage-status"], ["span", "review-progress"], ["nav", "navigation"], ["main", "content"], ["footer", "footer"], ["script", "pilot-data"]]) {
    const item = document.createElement(tag); item.id = id; document.body.append(item);
  }
  document.getElementById("pilot-data").textContent = typeof payload === "string" ? payload : JSON.stringify(payload);
  let sequence = 0;
  const context = vm.createContext({document, localStorage: saved, Blob,
    URL: {createObjectURL(blob) { const url = `blob:${++sequence}`; document.blobs.set(url, blob); return url; }, revokeObjectURL() {}},
    setTimeout(callback) { callback(); }
  });
  vm.runInContext(script, context, {timeout: 5000});
  return {
    document, saved,
    get(id) { const value = document.getElementById(id); assert.ok(value, `Expected element ${id}`); return value; },
    text(id = "content") { return this.get(id).textContent; },
    change(id, value) { const item = this.get(id); item.value = value; item.dispatch("change"); },
    input(id, value) { const item = this.get(id); item.value = value; item.dispatch("input"); },
    click(id) { this.get(id).click(); },
    clickText(text) { const item = document.body.find(element => element.tagName === "button" && element.textContent === text); assert.ok(item, `Expected button ${text}`); item.click(); }
  };
}

const counts = {moves: 12, games: 3, game_sides: 5, mover_accounts: 6, accounts: 7};
function metric(value, status = "finite") {
  return {status: "available", estimate: {status, value}, counts, masked_moves: 0, uncertainty: {status: "inconclusive", interval95: null, reasons: ["single_seed_exploratory"]}};
}
function calibration(points = [], overrides = {}) {
  const edges = [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1];
  const bins = edges.slice(0, -1).map((lower, index) => ({lower, upper: edges[index + 1], upper_inclusive: index === 9, raw_count: 0, weighted_mass: 0, mean_prediction: {status: "undefined", value: null}, mean_result: {status: "undefined", value: null}, counts: {moves: 0, games: 0, game_sides: 0, mover_accounts: 0, accounts: 0, insufficient: true}}));
  for (const [index, prediction, result, rawCount, weight] of points) bins[index] = {...bins[index], raw_count: rawCount, weighted_mass: weight, mean_prediction: {status: "finite", value: prediction}, mean_result: {status: "finite", value: result}, counts: {...counts, moves: rawCount, insufficient: true}};
  return {status: "available", aggregate_ranks: ["5k"], missing_ranks: [], weighting: "frozen_value_row_weights", bins, ...overrides};
}
const group = {counts, metrics: {nll: metric(3.4), top1: metric(0), value_mse: metric(.18)}, value_calibration: {model: calibration([[0, -1, -1, 3, .3], [5, 0, 1, 2, .2], [9, 1, 0, 5, .5]])}};
const unavailable = {...group, metrics: Object.fromEntries(["nll", "top1", "value_mse"].map(key => [key, {...metric(null, "undefined"), status: "unavailable", missing_ranks: ["5k"]}])), value_calibration: {model: {...group.value_calibration.model, status: "unavailable", missing_ranks: ["10k"]}}};
const rankGroup = {...group, value_calibration: {model: calibration([[2, -.5, .25, 5, 1]])}};
const phaseGroup = {...group, value_calibration: {model: calibration([[0, -.9, .5, 3, 1]])}};
const clockGroup = {...group, value_calibration: {model: calibration()}};
const categoryNames = ["pass", "capture", "self_atari", "atari", "connection", "other"];
const categories = {
  status: "available", counts, model: Object.fromEntries(categoryNames.map((name, index) => [name, index === 0 ? .5 : .1])),
  recorded: Object.fromEntries(categoryNames.map((name, index) => [name, index === 0 ? .25 : .15])),
  total_variation: .25, uncertainty: {status: "inconclusive", interval95: null, reasons: ["confidence_interval_not_estimated"]}
};
const report = {
  metrics: {overall: group, by_rank: {"5k": rankGroup, "10k": unavailable}, by_phase: {"1-60": phaseGroup}, by_clock: {missing: clockGroup}},
  sampling: {draws_per_position: 32, temperature: 1, overall: {unique_legal_actions: 20, pairwise_disagreement: .8, raw_pass_mass: 0, sampled_pass_fraction: 0}, by_rank: {"5k": {unique_legal_actions: 18, pairwise_disagreement: .7, raw_pass_mass: .02, sampled_pass_fraction: .01}}, move_categories: {categories: categoryNames, overall: categories, by_rank: {"5k": categories, "10k": {...categories, status: "unavailable", missing_ranks: ["10k"], model: {}, recorded: {}, total_variation: null}}, by_phase: {"1-60": categories}, by_clock: {missing: categories}, joint_cells: []}, scope: "Synthetic diagnostics only"}
};
const fixture = {
  bundle_id: "synthetic-ui-test", title: "Synthetic test <script>alert(1)</script>", summary: "Synthetic assertions only; never a published review fixture.",
  limitations: ["Small synthetic test"], decision: {prompt: "Choose the next action", options: [{id: "defer", label: "Defer", detail: "Collect evidence first"}, {id: "investigate", label: "Investigate", detail: "Review the failures"}]},
  roles: [{id: "population_development", label: "Population development"}, {id: "validation_development", label: "Validation development"}],
  arms: [{id: "conditioned", label: "Conditioned"}, {id: "unconditioned", label: "Unconditioned"}, {id: "human-reference", label: "Human SL"}, {id: "conditioned_unknown_ranks", label: "Unknown-rank diagnostic"}],
  evaluations: {population_development: {conditioned: {...report, metrics: {...report.metrics, overall: {...group, value_calibration: {model: calibration([[1, -.75, -.5, 4, 1]])}}}}}, validation_development: {conditioned: report, unconditioned: {...report, metrics: {...report.metrics, overall: {...group, value_calibration: {model: calibration([[9, .9, -.5, 4, 1]])}}}}, "human-reference": {...report, metrics: {...report.metrics, overall: {...group, metrics: {...group.metrics, nll: metric(null, "positive_infinity")}, value_calibration: null}}}, conditioned_unknown_ranks: {...report, metrics: {...report.metrics, overall: {...group, metrics: {...group.metrics, nll: metric(null, "negative_infinity"), value_mse: metric(null, "undefined")}}}}}},
  paired: {validation_development: {conditioned_minus_unconditioned: {overall: {metrics: {nll: metric(-.25)}}}, "conditioned_minus_human-reference": {overall: {metrics: {nll: metric(.3)}}}}},
  training: {conditioned: {epoch_metrics: [{epoch: 1, policy_loss: 5, value_loss: .6}, {epoch: 2, policy_loss: null, value_loss: .4}, {epoch: 3, policy_loss: 3, value_loss: .2}], completed_updates: 9, presented_examples: 72, costs: {gpu_seconds: 1, cpu_seconds: 2, wall_seconds: 3}}},
  costs: {gpu_seconds: 1, cpu_seconds: null, wall_seconds: 3, remaining_gpu_seconds: 10},
  games: [
    {id: "case-1", arm: "conditioned", opponent: "synthetic", candidate_color: "B", conditioning_rank: "5k", result: "U", status: "truncated", reason: "administrative_cap", censored: true, actual_actions: 3, behavior: {runtime_failure: false, cap: true, unresolved: true, dispute: null, premature_pass: null, length_tail: false}, moves: [["B", 0], ["W", 1], ["B", 361]], frames: [[[], "B", 0, 0, 0], [[[0, "B"]], "W", 0, 0, 0], [[[0, "."], [1, "W"]], "B", 1, 0, 0], [[], "W", 1, 0, 1]], passes: [{ply: 2, move_index: 2, verdict: "unjudgeable"}], sgf: "(;GM[1]SZ[19];B[aa];W[ba];B[])"},
    {id: "case-2", arm: "unconditioned", opponent: "synthetic", candidate_color: "W", conditioning_rank: null, result: "L", status: "runtime_failure", reason: "synthetic_failure", censored: true, actual_actions: 0, behavior: {runtime_failure: true}, moves: [], frames: null, passes: [], sgf: "(;GM[1]SZ[19])"}
  ],
  evidence: [{label: "Synthetic", path: "synthetic/<fixture>.json", sha256: "a".repeat(64), url: "https://example.invalid/evidence"}]
};

(async () => {
  const app = launch(fixture);
  assert.equal(app.get("title").textContent, fixture.title);
  assert.equal(app.document.body.all(element => element.tagName === "script").length, 1);
  assert.match(app.text(), /positive infinity/);
  assert.match(app.text(), /negative infinity/);
  assert.match(app.text(), /Undefined/);
  assert.match(app.text(), /0%/);
  assert.match(app.text(), /No finite estimates|Bar scale/);
  assert.match(app.text("paired-result"), /Conditioned − Unconditioned-0.25/);
  assert.match(app.text("paired-result"), /Conditioned − Human SL\+0.3/);
  const calibrationPoints = () => app.get("calibration-chart").all(element => element.tagName === "circle");
  const calibrationRows = () => app.get("calibration-table").find(element => element.tagName === "tbody").children;
  assert.equal(calibrationPoints().length, 3, "Empty bins are absent points, not zeros");
  assert.deepEqual(calibrationPoints().map(point => [point.attrs.cx, point.attrs.cy]), [["15%", "240"], ["55.5%", "36"], ["96%", "138"]]);
  assert.equal(app.get("calibration-chart").attrs.height, "300");
  assert.deepEqual([app.get("calibration-identity").attrs.x1, app.get("calibration-identity").attrs.y1, app.get("calibration-identity").attrs.x2, app.get("calibration-identity").attrs.y2], ["15%", "240", "96%", "36"]);
  assert.equal(calibrationRows().length, 10);
  assert.equal(calibrationRows()[0].children[0].textContent, "[-1, -0.8)");
  assert.equal(calibrationRows()[9].children[0].textContent, "[0.8, 1]");
  assert.equal(calibrationRows()[0].children[2].textContent, "0.3");
  assert.equal(calibrationRows()[1].children[3].textContent, "Undefined");
  assert.match(app.text("calibration-result"), /mover's perspective/);
  assert.match(app.text("calibration-result"), /not raw win probabilities/);
  app.change("calibration-arm", "unconditioned");
  assert.equal(calibrationPoints().length, 1);
  assert.ok(Math.abs(parseFloat(calibrationPoints()[0].attrs.cx) - 91.95) < 1e-10);
  assert.equal(calibrationPoints()[0].attrs.cy, "189");
  app.change("calibration-arm", "human-reference");
  assert.match(app.text("calibration-result"), /No value-calibration report/);
  assert.equal(app.document.getElementById("calibration-chart"), null);
  app.change("calibration-arm", "conditioned");
  app.change("dimension-filter", "rank");
  assert.equal(calibrationPoints().length, 1);
  assert.deepEqual([calibrationPoints()[0].attrs.cx, calibrationPoints()[0].attrs.cy], ["35.25%", "112.5"]);
  app.change("group-filter", "10k");
  assert.equal(calibrationPoints().length, 0, "An unavailable aggregate must not plot finite diagnostic means");
  assert.equal(calibrationRows().length, 10, "Unavailable aggregate bins remain inspectable");
  assert.match(app.text("calibration-result"), /Missing aggregate ranks: 10k/);
  assert.match(app.text("calibration-empty"), /Calibration is unavailable/);
  assert.match(app.text("paired-result"), /overall, regardless|overall population, regardless/);
  assert.match(app.text("paired-result"), /-0.25/);
  assert.match(app.text(), /Unavailable/);
  assert.match(app.text("sampling-result"), /Not available/);
  assert.match(app.text("sampling-result"), /Missing aggregate ranks: 10k/);
  app.change("role-filter", "population_development");
  assert.match(app.text(), /Not reported/);
  app.change("dimension-filter", "overall");
  assert.equal(calibrationPoints().length, 1);
  assert.deepEqual([calibrationPoints()[0].attrs.cx, calibrationPoints()[0].attrs.cy], ["25.125%", "189"]);
  app.change("dimension-filter", "phase");
  assert.equal(calibrationPoints().length, 1);
  assert.ok(Math.abs(parseFloat(calibrationPoints()[0].attrs.cx) - 19.05) < 1e-10);
  assert.equal(calibrationPoints()[0].attrs.cy, "87");
  assert.match(app.text("sampling-result"), /no reported breakdown/);
  app.change("dimension-filter", "clock");
  assert.equal(calibrationPoints().length, 0);
  assert.equal(calibrationRows().length, 10);
  assert.match(app.text("calibration-empty"), /No eligible bins/);

  const invalidCalibration = structuredClone(fixture);
  const invalidBins = invalidCalibration.evaluations.validation_development.conditioned.metrics.overall.value_calibration.model.bins;
  invalidBins[0].mean_result = {status: "positive_infinity", value: null};
  invalidBins[5].mean_prediction = {status: "finite", value: null};
  invalidBins[9].weighted_mass = null;
  const invalidCalibrationApp = launch(invalidCalibration);
  assert.equal(invalidCalibrationApp.get("calibration-chart").all(element => element.tagName === "circle").length, 0);
  assert.match(invalidCalibrationApp.text("calibration-result"), /positive infinity/);
  assert.match(invalidCalibrationApp.text("calibration-result"), /Not available/);
  const preciseCalibration = structuredClone(fixture);
  const preciseBin = preciseCalibration.evaluations.validation_development.conditioned.metrics.overall.value_calibration.model.bins[0];
  preciseBin.mean_prediction.value = -.9876543210987654;
  preciseBin.mean_result.value = .1234567890123456;
  preciseBin.weighted_mass = .30000000000000004;
  const preciseApp = launch(preciseCalibration);
  assert.match(preciseApp.text("calibration-table"), /-0\.9876543210987654/);
  assert.match(preciseApp.text("calibration-table"), /0\.1234567890123456/);
  assert.match(preciseApp.text("calibration-table"), /0\.30000000000000004/);
  const missingBins = structuredClone(fixture);
  missingBins.evaluations.validation_development.conditioned.metrics.overall.value_calibration.model.bins = [];
  const missingBinsApp = launch(missingBins);
  assert.match(missingBinsApp.text("calibration-result"), /No calibration bins were supplied/);
  assert.equal(missingBinsApp.get("calibration-table").find(element => element.tagName === "tbody").children.length, 0);

  app.click("view-training");
  assert.match(app.text(), /Not reported/);
  assert.match(app.text(), /Not available/);
  const policyChart = app.get("training-result").find(element => element.tagName === "svg" && element.attrs["aria-label"].startsWith("Policy loss"));
  assert.equal(policyChart.all(element => element.tagName === "circle").length, 2);
  assert.equal(policyChart.all(element => element.tagName === "line").length, 2, "A missing epoch must break the line; only the two grid lines remain");

  app.click("view-games");
  assert.match(app.text("replay-status"), /Move 0 of 3/);
  assert.match(app.text(), /Conditioning rank input5k/);
  assert.match(app.text(), /U · unresolved/);
  assert.match(app.text(), /Censored lengthYes/);
  app.click("replay-next");
  assert.match(app.text("replay-status"), /Black A19 · candidate/);
  app.click("replay-next");
  assert.match(app.text("replay-status"), /White B19/);
  assert.match(app.text("replay-metadata"), /Captured black stones: 1/);
  assert.equal(app.get("board").all(element => element.tagName === "circle" && element.attrs.fill === "#202721").length, 0);
  assert.equal(app.get("board").all(element => element.tagName === "circle" && element.attrs.fill === "#fffdf6").length, 1);
  app.click("replay-end");
  assert.match(app.text("replay-status"), /Move 3 of 3 · Black pass · candidate/);
  assert.equal(app.get("replay-next").disabled, true);
  app.clickText("Before");
  assert.match(app.text("replay-status"), /Move 2 of 3/);
  app.clickText("After");
  assert.match(app.text("replay-status"), /Move 3 of 3/);
  app.input("replay-slider", 1);
  assert.match(app.text("replay-status"), /Move 1 of 3/);
  app.input("note-text-game:case-1", "A19 checked <script>literal annotation</script>");
  app.change("note-status-game:case-1", "checked");
  assert.match(app.text("review-progress"), /1 of 2 games checked/);
  app.click("download-sgf");
  assert.equal(await app.document.downloads.at(-1).blob.text(), fixture.games[0].sgf);
  app.click("next-game");
  assert.match(app.text(), /Replay unavailable/);
  assert.match(app.text(), /L · candidate loss/);
  assert.match(app.text(), /Conditioning rank inputNone/);
  assert.equal(app.get("next-game").disabled, true);
  app.click("previous-game");
  assert.equal(app.get("note-text-game:case-1").value, "A19 checked <script>literal annotation</script>");
  app.change("game-arm", "unconditioned");
  assert.equal(app.get("game-select").children.length, 1);

  app.click("view-decision");
  const coverageFixture = structuredClone(fixture);
  coverageFixture.decision.coverage_audit = {status: "partial", summary: "Only a fixed prefix was attempted", gpu_seconds: 0, network_requests: 0,
    roles: [{label: "Training", candidates: 100, attempted: 40, source_admitted: 0, distinct_handles: null, passes: 0}],
    scope: "Provisional only; no legal replay or profile qualification"};
  const coverageApp = launch(coverageFixture); coverageApp.click("view-decision");
  assert.match(coverageApp.text("coverage-audit"), /partial/);
  assert.match(coverageApp.text("coverage-audit"), /Training100400Not available0/);
  assert.match(coverageApp.text("coverage-audit"), /no legal replay or profile qualification/);
  assert.match(coverageApp.text("coverage-audit"), /GPU seconds0New requests0/);
  app.get("decision-defer").checked = true; app.get("decision-defer").dispatch("change");
  app.input("decision-text", "Keep the evidence limited.");
  app.click("export-notes");
  const exported = JSON.parse(await app.document.downloads.at(-1).blob.text());
  assert.equal(exported.bundle_id, fixture.bundle_id);
  assert.equal(exported.decision.choice, "defer");
  assert.equal(exported.decision.text, "Keep the evidence limited.");
  assert.equal(exported.notes["game:case-1"].status, "checked");
  const reload = launch(fixture, app.saved);
  reload.click("view-games");
  assert.equal(reload.get("note-status-game:case-1").value, "checked");
  reload.click("view-decision");
  assert.equal(reload.get("decision-defer").checked, true);
  assert.equal(reload.get("decision-text").value, "Keep the evidence limited.");

  app.click("view-evidence");
  assert.equal(app.document.body.all(element => element.tagName === "a").length, 0, "Evidence references do not create network navigation links");
  app.clickText("Download embedded evidence JSON");
  assert.deepEqual(JSON.parse(await app.document.downloads.at(-1).blob.text()), fixture);
  const raw = app.get("content").find(element => element.tagName === "details");
  raw.open = true; raw.dispatch("toggle");
  assert.match(raw.textContent, /Synthetic test <script>alert\(1\)<\/script>/);

  const memory = launch(fixture, storage(new Map(), true));
  assert.match(memory.text("storage-status"), /in memory only/);
  memory.click("view-games"); memory.input("note-text-game:case-1", "Export this memory-only note");
  memory.click("export-notes");
  assert.equal(JSON.parse(await memory.document.downloads.at(-1).blob.text()).notes["game:case-1"].text, "Export this memory-only note");
  assert.match(memory.text("storage-status"), /in memory only/);

  const invalid = launch("__PILOT_DATA_JSON__");
  assert.equal(invalid.get("export-notes").disabled, true);
  assert.match(invalid.text(), /Unable to load the evidence/);
  const malformed = structuredClone(fixture);
  malformed.games[0].frames[1][0][0][0] = 361;
  const malformedApp = launch(malformed); malformedApp.click("view-games");
  assert.match(malformedApp.text(), /Replay unavailable: A supplied board delta is invalid/);
  const empty = launch({bundle_id: "empty", games: [], evaluations: {}, arms: [], roles: []});
  assert.match(empty.text(), /No evaluation report/);
  for (const view of ["training", "games", "decision", "evidence"]) empty.click(`view-${view}`);
  console.log("PASS: synthetic UI interactions, calibration coordinates/filter switches/exact bins, missing/infinite values, frame replay, pass jumps, notes persistence, storage fallback, original downloads, safe text, and empty/malformed bundles.");
  console.log("LIMIT: DOM assertions do not verify browser layout, SVG rendering, or native download behavior.");
})().catch(error => { console.error(error); process.exitCode = 1; });
