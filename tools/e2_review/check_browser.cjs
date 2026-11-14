// Uses an existing Playwright installation; the exported page has no dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

(async () => {
  const html = path.resolve(process.argv[2] || "runs/e2-manual-review-2026-11-14/review-final.html");
  const output = path.resolve(process.argv[3] || "runs/e2-manual-review-2026-11-14/browser");
  await fs.mkdir(output, {recursive:true});
  const browser = await chromium.launch({headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
  const context = await browser.newContext({viewport:{width:728,height:1100},acceptDownloads:true});
  const page = await context.newPage(), errors = [], requests = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {if (/^https?:/.test(request.url())) requests.push(request.url());});
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), "kenggo-e2-browser-"));
  try {
    await page.goto(pathToFileURL(html).href, {timeout:20000});
    const data = await page.locator("#pilot-data").textContent().then(JSON.parse);
    assert.equal(data.games.length,12);
    assert.match(await page.locator("#content").innerText(), /6\.0038/);
    await page.locator("#role-filter").selectOption("population_development");
    assert.match(await page.locator("#content").innerText(), /5\.9532/);
    assert.ok(await page.locator("#calibration-table tbody tr").evaluateAll(rows=>rows.every(row=>row.getBoundingClientRect().height<140)), "Calibration coverage must not squeeze each row into a tall narrow column");
    await page.screenshot({path:path.join(output,"compare-desktop.png"),fullPage:true});
    await page.locator("#view-training").click();
    assert.ok(await page.locator("#training-result svg").count()>=2);
    await page.locator("#view-games").click();
    await page.locator("#replay-end").click();
    assert.match(await page.locator("#replay-status").innerText(), /Move 433 of 433/);
    assert.match(await page.locator("#game-result").innerText(), /cleanup move budget exhausted/);
    await page.screenshot({path:path.join(output,"game-desktop.png"),fullPage:true});
    const firstPass = data.games[0].passes[0].ply;
    await page.getByRole("button",{name:/Before candidate pass/}).first().click();
    assert.equal(Number(await page.locator("#replay-slider").inputValue()),firstPass);
    await page.getByRole("button",{name:/After candidate pass/}).first().click();
    assert.equal(Number(await page.locator("#replay-slider").inputValue()),firstPass+1);
    await page.locator('[id="note-text-game:s00000000"]').fill("Synthetic browser verification; not owner feedback.");
    await page.locator('[id="note-status-game:s00000000"]').selectOption("followup");
    await page.reload();
    await page.locator("#view-games").click();
    assert.equal(await page.locator('[id="note-status-game:s00000000"]').inputValue(),"followup");
    assert.match(await page.locator('[id="note-text-game:s00000000"]').inputValue(),/Synthetic browser/);
    let pending = page.waitForEvent("download");
    await page.locator("#download-sgf").click();
    let download = await pending;
    await download.saveAs(path.join(temporary,"game.sgf"));
    assert.equal(await fs.readFile(path.join(temporary,"game.sgf"),"utf8"),data.games[0].sgf);
    await page.locator("#view-decision").click();
    assert.match(await page.locator("#coverage-audit").innerText(), /2,135 unattempted/);
    await page.locator("#decision-cached_corpus").check();
    await page.locator("#decision-text").fill("Synthetic test choice only.");
    pending = page.waitForEvent("download");
    await page.locator("#export-notes").click();
    download = await pending;
    await download.saveAs(path.join(temporary,"notes.json"));
    const notes = JSON.parse(await fs.readFile(path.join(temporary,"notes.json"),"utf8"));
    assert.equal(notes.bundle_id,data.bundle_id);
    assert.equal(notes.decision.choice,"cached_corpus");
    assert.equal(notes.notes["game:s00000000"].status,"followup");
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await page.screenshot({path:path.join(output,"decision-mobile.png"),fullPage:true});
    await page.locator("#view-compare").click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await page.screenshot({path:path.join(output,"compare-mobile.png"),fullPage:true});
    await page.locator("#view-evidence").click();
    assert.match(await page.locator("#content").innerText(),new RegExp(data.bundle_id));
    assert.deepEqual(errors,[]);
    assert.deepEqual(requests,[]);
    console.log(JSON.stringify({status:"passed",browser:browser.version(),games:data.games.length,
      checks:["actual metrics and charts","original replay and pass jumps","persistent notes","exact SGF download","notes export","mobile overflow","zero page errors","zero HTTP requests"],output}));
  } finally {
    await browser.close();
    await fs.rm(temporary,{recursive:true,force:true});
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
