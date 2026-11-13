"use strict";
const data = JSON.parse(document.getElementById("review-data").textContent);
const byId = new Map(data.games.map(g => [g.id, g]));
const app = document.getElementById("app");
const storageKey = `kenggo-e1-review:${data.bundle_id}`;
const statuses = {pending: "Not reviewed", checked: "Checked", followup: "Needs follow-up", na: "Not applicable"};
const checkIds = ["endings", "admission", "examples", "scope", "gpu", "delivery"];
const allowedKeys = new Set([
  ...data.endings.map(e => `ending:${e.case}`), ...data.games.map(g => `game:${g.id}`),
  ...data.examples.map(e => `example:${e.metadata.row_key}`), ...checkIds.map(id => `check:${id}`)
]);
let notes = {}, storageOK = true, tab = "endings";
let endingId = data.endings.find(e => e.winner_agrees === false)?.case ?? data.endings[0]?.case;
let gameId = data.games.find(g => g.frames)?.id, exampleId = 0;
let endingFilter = "winner", endingQuery = "", admissionFilter = "admitted", admissionQuery = "";
const frameCache = new WeakMap();

function node(tag, text, attributes = {}) {
  const element = document.createElement(tag);
  if (text !== null && text !== undefined) element.textContent = String(text);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}
function add(parent, ...children) { parent.append(...children); return parent; }
function button(text, fn, secondary = true) {
  const element = node("button", text, secondary ? {class: "secondary"} : {});
  element.onclick = fn; return element;
}
function select(options, value, label) {
  const element = node("select", null, {"aria-label": label});
  for (const [key, text] of Object.entries(options)) element.append(node("option", text, {value: key}));
  element.value = value; return element;
}
function table(headers, rows) {
  const result = node("table"), head = node("tr"), body = node("tbody");
  headers.forEach(text => head.append(node("th", text)));
  rows.forEach(row => { const tr = node("tr"); row.forEach(value => tr.append(node("td", value))); body.append(tr); });
  return add(node("div", null, {class: "tablewrap"}), add(result, add(node("thead"), head), body));
}
function jsonDetails(title, value) {
  return add(node("details"), node("summary", title), node("pre", JSON.stringify(value, null, 2)));
}
function badge(text, kind = "") { return node("span", text, {class: `tag ${kind}`}); }
function heading(title, description) {
  app.append(node("h2", title), node("p", description, {class: "muted"}));
}
function title(g) { return `${g.metadata.PB || "Unknown"} (${g.metadata.BR || "?"}) · ${g.metadata.PW || "Unknown"} (${g.metadata.WR || "?"})`; }
function coordinate(action) { return action === 361 ? "pass" : `${"ABCDEFGHJKLMNOPQRST"[action % 19]}${19 - Math.floor(action / 19)}`; }
function score(halfpoints) { return halfpoints == null ? "Unresolved" : `${halfpoints > 0 ? "W" : "B"} +${Math.abs(halfpoints) / 2}`; }
function noteFlag(key) { return notes[key]?.status === "checked" ? "✓ " : notes[key]?.status === "followup" ? "! " : ""; }

function validateNotes(value) {
  if (!value || Array.isArray(value) || typeof value !== "object") throw Error("Invalid notes object");
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    if (!allowedKeys.has(key) || !item || !Object.hasOwn(statuses, item.status) ||
        typeof item.text !== "string" || item.text.length > 20000 || typeof item.updated !== "string") {
      throw Error("Notes contain an unknown item or invalid field");
    }
    result[key] = {status: item.status, text: item.text, updated: item.updated};
  }
  return result;
}
function saveStatus(message) {
  const status = document.getElementById("save-status");
  status.textContent = storageOK ? message : "Browser storage unavailable. Notes are in memory; export before closing.";
  status.className = storageOK ? "muted" : "error";
  const values = Object.values(notes);
  document.getElementById("progress").textContent = `${values.filter(n => n.status === "checked").length} checked · ${values.filter(n => n.status === "followup").length} follow-ups`;
}
function readStoredNotes() {
  const stored = validateNotes(JSON.parse(localStorage.getItem(storageKey) || "{}"));
  for (const key of allowedKeys) {
    const value = localStorage.getItem(`${storageKey}:${key}`);
    if (value !== null) Object.assign(stored, validateNotes({[key]: JSON.parse(value)}));
  }
  return stored;
}
function refreshNotes() {
  if (storageOK) {
    try { notes = {...notes, ...readStoredNotes()}; } catch { storageOK = false; }
  }
}
try { notes = readStoredNotes(); }
catch { storageOK = false; }
saveStatus("Notes save in this browser. Export for a portable backup.");

window.addEventListener("storage", event => {
  if (!event.key || !(event.key === storageKey || event.key.startsWith(`${storageKey}:`))) return;
  refreshNotes();
  document.querySelectorAll("[data-review-key]").forEach(editor => {
    const item = notes[editor.dataset.reviewKey];
    if (item) {
      editor.querySelector("textarea").value = item.text;
      editor.querySelector("select").value = item.status;
    }
  });
  saveStatus("Review notes updated from another tab.");
});

function save(key, status, text) {
  notes[key] = {status, text, updated: new Date().toISOString()};
  if (storageOK) {
    try { localStorage.setItem(`${storageKey}:${key}`, JSON.stringify(notes[key])); } catch { storageOK = false; }
  }
  saveStatus("Saved in this browser.");
}
function noteEditor(key, prompt = "What did you find? Record board coordinates, move numbers, and any follow-up.") {
  const card = node("section", null, {class: "card notes", "aria-label": "Review notes", "data-review-key": key});
  const status = select(statuses, notes[key]?.status || "pending", "Review status");
  const text = node("textarea", notes[key]?.text || "", {placeholder: prompt, maxlength: "20000", "aria-label": "Review notes"});
  status.onchange = text.oninput = () => save(key, status.value, text.value);
  return add(card, add(node("div", null, {class: "inline"}), node("h3", "Your review"), status), text,
    node("p", "Annotations only. Export notes to retain or hand off your findings.", {class: "muted"}));
}
function download(name, content, mime = "application/json") {
  const url = URL.createObjectURL(new Blob([content], {type: mime}));
  const anchor = node("a", "", {href: url, download: name});
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
document.getElementById("export").onclick = () => {
  refreshNotes();
  download("kenggo-e1-review-notes.json", JSON.stringify({
    version: "e1_manual_notes_v1", bundle_id: data.bundle_id, exported_at: new Date().toISOString(), notes
  }, null, 2));
};
document.getElementById("import").onchange = async event => {
  try {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 32 * 1024 * 1024) throw Error("Notes file exceeds 32 MiB");
    const imported = JSON.parse(await file.text());
    if (imported.version !== "e1_manual_notes_v1" || imported.bundle_id !== data.bundle_id) throw Error("Notes belong to a different evidence bundle");
    const incoming = validateNotes(imported.notes);
    refreshNotes(); notes = {...notes, ...incoming};
    try {
      for (const [key, value] of Object.entries(incoming)) localStorage.setItem(`${storageKey}:${key}`, JSON.stringify(value));
      storageOK = true;
    } catch { storageOK = false; }
    render(); saveStatus("Imported notes; matching entries replaced, other notes retained.");
  } catch (error) { document.getElementById("save-status").textContent = `Import failed: ${error.message}`; }
  event.target.value = "";
};

function expandFrames(packed) {
  if (frameCache.has(packed)) return frameCache.get(packed);
  const board = Array(361).fill(".");
  const result = packed.map(frame => {
    frame[0].forEach(([index, cell]) => { board[index] = cell; });
    return {board: board.join(""), toPlay: frame[1], capturedBlack: frame[2], capturedWhite: frame[3], passes: frame[4], phase: frame[5]};
  });
  frameCache.set(packed, result); return result;
}
function drawBoard(board, {last = null, target = null, legal = null, ko = [], compare = null, plane = "all", mover = "B"} = {}) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 480 480"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "19 by 19 Go board; A19 is top left, T1 bottom right");
  const append = (tag, attrs, text) => {
    const child = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) child.setAttribute(key, String(value));
    if (text != null) child.textContent = text;
    svg.append(child); return child;
  };
  for (let i = 0; i < 19; i++) {
    const p = 33 + i * 23;
    append("line", {x1:33, y1:p, x2:447, y2:p, stroke:"#94733c", "stroke-width":.7});
    append("line", {x1:p, y1:33, x2:p, y2:447, stroke:"#94733c", "stroke-width":.7});
    for (const y of [19, 466]) append("text", {x:p, y, "text-anchor":"middle", "font-size":10, fill:"#624b2d"}, "ABCDEFGHJKLMNOPQRST"[i]);
    for (const x of [16, 465]) append("text", {x, y:p+3, "text-anchor":"middle", "font-size":10, fill:"#624b2d"}, 19-i);
  }
  for (const row of [3,9,15]) for (const col of [3,9,15]) append("circle", {cx:33+col*23, cy:33+row*23, r:2.8, fill:"#624b2d"});
  [...board].forEach((cell, i) => {
    const x = 33 + (i%19)*23, y = 33 + Math.floor(i/19)*23;
    if (legal?.[i] === "1") append("circle", {cx:x, cy:y, r:3, fill:"#246b58", opacity:.55});
    if (compare && compare[i] !== cell) append("rect", {x:x-11, y:y-11, width:22, height:22, fill:"#d97721", opacity:.36});
    const visible = plane === "all" || (plane === "own" && cell === mover) || (plane === "opponent" && cell !== mover);
    if (cell !== "." && visible) append("circle", {cx:x, cy:y, r:10.4, fill:cell === "B" ? "#252923" : "#fffdf4", stroke:cell === "B" ? "#151a17" : "#b5ae98", "stroke-width":1});
    if (i === last) append("circle", {cx:x, cy:y, r:3, fill:cell === "B" ? "#f2dcaa" : "#254f3e"});
    if (ko.includes(i)) append("rect", {x:x-6, y:y-6, width:12, height:12, fill:"none", stroke:"#a23f33", "stroke-width":2});
    if (i === target) append("circle", {cx:x, cy:y, r:8, fill:"none", stroke:"#237268", "stroke-width":2.5});
    const hit = append("circle", {cx:x, cy:y, r:10, fill:"transparent"});
    const tooltip = document.createElementNS(ns, "title"); tooltip.textContent = `${coordinate(i)}: ${cell === "." ? "empty" : cell === "B" ? "Black" : "White"}`; hit.append(tooltip);
  });
  return svg;
}
function player(packed, moves, label, initial, compare = null) {
  const states = expandFrames(packed);
  const box = node("section", null, {class: "boardbox", "aria-label": label});
  const boardSlot = node("div"), metadata = node("div", null, {class: "board-meta"});
  const controls = node("div", null, {class: "controls"}), caption = node("span", null, {class: "move-label"});
  const range = node("input", null, {type:"range", min:0, max:states.length-1, step:1, "aria-label": `${label} move`});
  let current = initial;
  const change = position => {
    current = Math.max(0, Math.min(states.length-1, position)); range.value = current;
    const frame = states[current], move = moves[current-1];
    boardSlot.replaceChildren(drawBoard(frame.board, {last:move?.[1], compare}));
    caption.textContent = `${current} / ${states.length-1}${move ? ` · ${move[0]} ${coordinate(move[1])}` : " · start"}`;
    metadata.textContent = `Next: ${frame.toPlay === "B" ? "Black" : "White"} · Prisoners: Black ${frame.capturedWhite}, White ${frame.capturedBlack} · Pass streak ${frame.passes}${label.includes("Cleanup") ? ` · Native phase ${frame.phase}` : ""}`;
    controls.querySelectorAll("button").forEach((b, i) => { b.disabled = i < 3 ? current === 0 : current === states.length-1; });
  };
  [["Start",()=>change(0)],["−10",()=>change(current-10)],["←",()=>change(current-1)],
    ["→",()=>change(current+1)],["+10",()=>change(current+10)],["End",()=>change(states.length-1)]].forEach(([text,fn])=>controls.append(button(text,fn)));
  range.oninput = () => change(Number(range.value));
  add(box, node("h3", label), boardSlot, metadata, add(controls, range, caption)); change(initial);
  return {element:box, change};
}
function sourceInfo(g) {
  const top = node("div", null, {class: "section-top"});
  const identity = add(node("div"), node("h3", title(g), {class: "source-title"}),
    node("p", `Black / White · ${g.metadata.DT || "Date unknown"} · ${g.role.replaceAll("_", " ")} · Recorded ${g.metadata.RE || "unknown"}`, {class:"muted"}));
  const actions = node("div", null, {class:"inline"});
  if (g.sgf_base64) actions.append(button("Download original SGF", () => {
    const bytes = Uint8Array.from(atob(g.sgf_base64), c => c.charCodeAt(0));
    download(g.id.split("/").pop(), bytes, "application/x-go-sgf");
  }));
  if (/^https:\/\/files\.gokgs\.com\//.test(g.id)) actions.append(node("a", "Source ↗", {href:g.id, target:"_blank", rel:"noopener noreferrer"}));
  return add(top, identity, actions);
}
function queueButton(primary, secondary, active, fn) {
  const b = button("",fn); b.className = active ? "active" : "";
  return add(b, node("strong", primary), node("small", secondary));
}

function renderEndings() {
  heading("Compare source endings and cleanup", "Start with the two winner disagreements. Check groups, seki, prisoners and cleanup changes, then sample the other margin differences.");
  const filter = select({winner:"Winner disagreements", margin:"Any score difference", budgets:"Cleanup budgets disagree on margin", all:"All 108 endings"}, endingFilter, "Ending filter");
  const search = node("input", null, {type:"search", placeholder:"Find a player, date or case…", "aria-label":"Search endings"}); search.value=endingQuery;
  const count=node("span",null,{class:"count"}), queue=node("aside",null,{class:"queue","aria-label":"Ending cases"}), detail=node("div",null,{class:"detail"});
  add(app, add(node("div",null,{class:"toolbar"}), add(node("div",null,{class:"inline"}), filter,search), count), add(node("div",null,{class:"workspace"}),queue,detail));
  function show(row) {
    detail.replaceChildren(); if (!row) {detail.append(node("p","No matching endings.",{class:"empty"}));return;}
    const g=byId.get(row.source_url), original=expandFrames(g.frames).at(-1).board;
    detail.append(sourceInfo(g));
    const results=node("div",null,{class:"result-row"});
    [["Source result",g.metadata.RE],["32-visit cleanup",score(row.native_scores_halfpoints[0])],["128-visit cleanup · selected",score(row.native_scores_halfpoints[1])]].forEach(([label,value])=>results.append(add(node("div"),node("small",label),node("strong",value))));
    detail.append(results,node("div","Cleanup is an empirical interpretation of the ending. Native scores apply to each played cleanup transcript; source labels stay unchanged. Orange squares mark changes from the source ending.",{class:"notice"}));
    const boards=node("div",null,{class:"boards"}), right=node("div");
    boards.append(player(g.frames,g.moves,"Source game",g.moves.length).element,right);
    const budgets=select({"32":"32 visits", "128":"128 visits · selected margin"},"128","Cleanup budget");
    detail.append(add(node("div",null,{class:"toolbar"}),node("span","Saved cleanup budget"),budgets),boards);
    function cleanup() {
      const run=row.runs.find(r=>r.visits===Number(budgets.value));
      right.replaceChildren(player(run.frames,run.moves,"Cleanup on a clone",run.moves.length,original).element);
    }
    budgets.onchange=cleanup;cleanup();
    const passes=data.passes.filter(p=>p.source_url===g.id);
    if(passes.length) {
      const panel=add(node("div",null,{class:"card pass-list"}),node("h3","Audited source passes"));
      panel.append(table(["Before move", "Color", "Verdict", "Gain, 32 / 128 visits"],passes.map(p=>[p.pre_move_index+1,p.color,p.verdict,p.gains_points.map(x=>x==null?"unknown":x.toFixed(3)).join(" / ")])));
      detail.append(panel);
    }
    detail.append(noteEditor(`ending:${row.case}`),jsonDetails("Comparison provenance",{...row,runs:undefined}));
  }
  function update() {
    endingFilter=filter.value;endingQuery=search.value;
    const rows=data.endings.filter(e=>(endingFilter==="all" || endingFilter==="winner"&&e.winner_agrees===false || endingFilter==="margin"&&e.margin_delta_halfpoints!=null&&e.margin_delta_halfpoints!==0 || endingFilter==="budgets"&&e.native_scores_halfpoints[0]!==e.native_scores_halfpoints[1])&&(`${String(e.case).padStart(3,"0")} ${title(byId.get(e.source_url))} ${byId.get(e.source_url).metadata.DT}`).toLowerCase().includes(endingQuery.toLowerCase()));
    if(!rows.some(e=>e.case===endingId)) endingId=rows[0]?.case;
    queue.replaceChildren(); count.textContent=`${rows.length} of ${data.endings.length} endings`;
    rows.forEach(row=>queue.append(queueButton(`${noteFlag(`ending:${row.case}`)}${String(row.case).padStart(3,"0")} · ${title(byId.get(row.source_url))}`,`${byId.get(row.source_url).metadata.RE} → ${score(row.selected_white_minus_black_halfpoints)}`,row.case===endingId,()=>{endingId=row.case;update();})));
    show(rows.find(e=>e.case===endingId));
  }
  filter.onchange=search.oninput=update;update();
}

function renderAdmission() {
  heading("Spot-check source admission", "Search the exposed source population for known bots, aliases or provenance mistakes. Admission is presumed-human screening, not verified identity.");
  const filter=select({admitted:"Admitted sources", excluded:"Excluded sources", all:"All exposed sources"},admissionFilter,"Admission filter");
  const search=node("input",null,{type:"search",placeholder:"Player, rank, date or exclusion…","aria-label":"Search admission"});search.value=admissionQuery;
  const queue=node("aside",null,{class:"queue","aria-label":"Source games"}),detail=node("div",null,{class:"detail"}),count=node("span",null,{class:"count"});
  add(app,add(node("div",null,{class:"toolbar"}),add(node("div",null,{class:"inline"}),filter,search),count),add(node("div",null,{class:"workspace"}),queue,detail));
  function show(g) {
    detail.replaceChildren();if(!g){detail.append(node("p","No matching source games.",{class:"empty"}));return;}
    const e=g.eligibility;
    detail.append(sourceInfo(g),add(node("div",null,{class:"chips"}),badge(e.training_eligible?"Admitted · presumed human":"Excluded",e.training_eligible?"":"warn"),badge("Identity / persona masked")),
      table(["Evidence", "Recorded value"],[["Archive type",g.archive.type],["Rules / komi / board",`${g.metadata.RU} / ${g.metadata.KM} / ${g.metadata.SZ}`],["Time control",`${g.metadata.TM} seconds + ${g.metadata.OT}`],["Source-use basis",e.source_use_basis],["Black / White screening",`${e.black_human_status} / ${e.white_human_status}`],["Admission reasons",e.eligibility_reasons.join(", ")||"No exclusion under the approved policy"],["Split exclusions",g.exclusions.join(", ")||"None"],["Recorded value usable",g.value_supported?"Yes, with mover perspective":"No: retain the value mask"]]));
    detail.append(node("p","The archive filename does not determine colors. The Black / White names above come from the SGF. Flag concrete evidence in your notes; this viewer does not quarantine or relabel games.",{class:"notice"}));
    if(g.frames) detail.append(player(g.frames,g.moves,"Source game",0).element);
    else detail.append(node("p","This excluded source has no validated replay in this export. Inspect the preserved metadata and source evidence.",{class:"muted"}));
    detail.append(noteEditor(`game:${g.id}`),jsonDetails("Source metadata, admission and provenance",{metadata:g.metadata,archive:g.archive,eligibility:e,source:g.source}));
  }
  function update(){
    admissionFilter=filter.value;admissionQuery=search.value;
    const rows=data.games.filter(g=>(admissionFilter==="all"||g.eligibility.training_eligible===(admissionFilter==="admitted"))&&JSON.stringify([g.metadata,g.exclusions,g.eligibility.eligibility_reasons]).toLowerCase().includes(admissionQuery.toLowerCase()));
    if(!rows.some(g=>g.id===gameId))gameId=rows[0]?.id;
    count.textContent=`${rows.length} of ${data.counts.source_records} exposed sources`;queue.replaceChildren();
    rows.forEach(g=>queue.append(queueButton(noteFlag(`game:${g.id}`)+title(g),`${g.metadata.DT||"Date unknown"} · ${g.role.replaceAll("_"," ")}`,g.id===gameId,()=>{gameId=g.id;update();})));
    show(rows.find(g=>g.id===gameId));
  }
  filter.onchange=search.oninput=update;update();
}

function renderExamples(){
  heading("Inspect stored pre-move examples", "Compare the source board with the actual stored input. The policy target is the next human action; values use the mover’s perspective. No board rotation is applied.");
  const queue=node("aside",null,{class:"queue","aria-label":"Stored examples"}),detail=node("div",null,{class:"detail"});
  app.append(node("p",`${data.examples.length} deterministic samples from training and development only. Includes first, middle, final and explicit-pass positions. Stored arrays are shown before normalization.`,{class:"muted"}),add(node("div",null,{class:"workspace"}),queue,detail));
  data.examples.forEach((e,index)=>queue.append(queueButton(`${noteFlag(`example:${e.metadata.row_key}`)}${title(byId.get(e.metadata.source_url))}`,`${e.metadata.role.replaceAll("_"," ")} · before move ${e.metadata.ply+1} · ${coordinate(e.action)}`,index===exampleId,()=>{exampleId=index;render();})));
  const e=data.examples[exampleId];if(!e)return;
  const m=e.metadata,g=byId.get(m.source_url),frame=expandFrames(g.frames)[m.ply];
  detail.append(sourceInfo(g),add(node("div",null,{class:"chips"}),badge(`Mover: ${m.mover==="B"?"Black":"White"}`),badge(`Target: ${coordinate(e.action)}`),badge(e.value_valid?`Value: ${e.value>0?"+1":"−1"} (${e.value>0?"mover wins":"mover loses"})`:"Value masked","warn")));
  const boards=node("div",null,{class:"boards"}),left=node("section",null,{class:"boardbox"}),right=node("section",null,{class:"boardbox"}),stored=node("div");
  left.append(node("h3",`Source · before move ${m.ply+1}`),drawBoard(frame.board,{target:e.action,last:g.moves[m.ply-1]?.[1]}));
  const plane=select({all:"Both stone planes",own:"Mover plane only",opponent:"Opponent plane only"},"all","Stored board plane");
  const legal=node("input",null,{type:"checkbox",id:"show-legal"});legal.checked=true;
  const controls=add(node("div",null,{class:"inline"}),plane,add(node("label"),legal,document.createTextNode(" Show legal moves")));
  const redraw=()=>stored.replaceChildren(drawBoard(e.board,{target:e.action,ko:e.ko,legal:legal.checked?e.legal:null,plane:plane.value,mover:m.mover}));
  plane.onchange=legal.onchange=redraw;redraw();
  right.append(node("h3","Stored mover-relative input"),stored,controls);
  boards.append(left,right);detail.append(boards,node("p",`Green ring: policy target (pass has no intersection). Green dots: legal placements. Red square: ko plane. Pass legal: ${e.legal[361]==="1"?"yes":"no"}. Own/opponent planes change by color; coordinates stay fixed.`,{class:"legend"}));
  detail.append(table(["Field", "Stored value", "Validity / meaning"],[
    ["Mover rank",e.rank_valid[0]?data.ranks[e.ranks[0]-1]:"unknown",m.rank_hidden[0]?"Training fallback mask; source "+m.source_rank_mover:"Source "+m.source_rank_mover],
    ["Opponent rank",e.rank_valid[1]?data.ranks[e.ranks[1]-1]:"unknown",m.rank_hidden[1]?"Training fallback mask; source "+m.source_rank_opponent:"Source "+m.source_rank_opponent],
    ["Policy target",`${e.action} · ${coordinate(e.action)}`,e.policy_valid?"Valid":"Masked"],
    ["Value target",e.value,e.value_valid?"Valid source outcome, mover perspective":"Masked; zero is a placeholder"],
    ["Duration",e.duration,e.duration_valid?"Valid":"Masked; zero is not a measured duration"],
    ["Persona",m.persona_mode,m.identity_valid?"Identity valid":"Identity masked"],
    ["Model history",e.history_actions.length,"Rules replay retains full history"]]));
  detail.append(table(["Scalar input", "Stored value", "Valid"],data.scalars.map((name,i)=>[name,e.scalars[i],e.scalar_valid[i]?"Yes":"No · masked placeholder"])),noteEditor(`example:${m.row_key}`),jsonDetails("Stored arrays and row provenance",e));
}

function renderChecklist(){
  heading("Record your manual sign-off", "E1 is recorded as complete for exploratory E2 work. These annotations help you inspect that evidence; they do not close tasks or change research gates.");
  const items=[
    ["endings","Inspect endings","Review both winner disagreements (044 and 083), then sample other score differences. Check living/dead groups, seki, captured stones and cleanup changes. Note remaining uncertainty; empirical agreement is not a proof."],
    ["admission","Spot-check admission","Check familiar accounts for known bots or aliases and verify source/date/rank/rules fields. Permission is owner-attested. Record concrete corrections for quarantine; there is no requirement to manually relabel all 1,000 candidates."],
    ["examples","Check a few encoded examples","Check board orientation, actual pass actions, mover-view value signs, training rank fallbacks, and missing clock/persona masks. A masked zero is not an observed zero. Final-test targets remain sealed."],
    ["scope","Accept the capability limits","This corpus supports exploratory population/rank baselines. No opponent profile, human rank, timing model or personalization capability is qualified. B must still start from random weights. E5 and E7 retain their qualification/calibration gates."],
    ["gpu","GPU readiness · separate E2 prerequisite","The GPU bead kenggo-ydo.7 remains open in the recorded handoff. This page cannot inspect your current driver or execute CUDA. After resolving host setup, run the synchronized check below; success requires status: ok. A CPU check is insufficient."],
    ["delivery","Check validation, review and delivery evidence",`The E1 record reports ${data.recorded_validation.tests} passing tests with ${data.recorded_validation.skips} skips and pinned native comparisons. Beads records a clean GPT-6-Astra/max review on pass 3. Confirm current task acceptance and source/tracker synchronization separately; this page is a frozen evidence export.`]
  ];
  items.forEach(([id,title,text],index)=>{
    const body=add(node("div"),node("h3",title),node("p",text));
    if(id==="gpu")body.append(node("pre","nvidia-smi\ntimeout 60s .cache/venvs/cuda/bin/python -m kenggo.cuda_smoke"),node("p","Run from the repository root after following docs/gpu_workspace.md. Record the actual output and date in your notes.",{class:"muted"}));
    if(id==="delivery")body.append(node("pre","bd show kenggo-6yr\ngit status --short --branch\ngit rev-parse HEAD origin/main"),node("p","Required checks and the bd sync / source push procedure are in AGENTS.md.",{class:"muted"}));
    body.append(noteEditor(`check:${id}`));app.append(add(node("section",null,{class:"card check-item"}),node("span",index+1,{class:"check-number"}),body));
  });
  app.append(jsonDetails("Export provenance",{bundle_id:data.bundle_id,readiness_sha256:data.readiness_sha256,counts:data.counts,deferred:data.deferred,scope:data.scope}));
}
function render(){
  app.replaceChildren();document.querySelectorAll("nav button").forEach(b=>{b.classList.toggle("active",b.dataset.tab===tab);b.setAttribute("aria-current",b.dataset.tab===tab?"page":"false");});
  ({endings:renderEndings,admission:renderAdmission,examples:renderExamples,checklist:renderChecklist})[tab]();
}
document.querySelectorAll("nav button").forEach(b=>{b.onclick=()=>{tab=b.dataset.tab;render();};});
render();
