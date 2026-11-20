// Run with the workspace's existing Playwright installation; the page is offline.
const fs = require("node:fs/promises");
const path = require("node:path");
const {pathToFileURL,fileURLToPath} = require("node:url");
const {chromium} = require("playwright");

(async () => {
  const input = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
  await fs.mkdir(output, {recursive:true});
  const checks = [], errors = [], requests = [];
  const result = {status:"incomplete", checks, errors, external_requests:requests};
  const ok = (name, passed) => { checks.push({name, passed:!!passed}); if (!passed) throw Error(name); };
  const display = (m,key) => {
    if(m?.status!=="available") return "Unavailable";
    if(m.estimate?.status==="positive_infinity") return "+∞";
    if(m.estimate?.status!=="finite") return "Unavailable";
    return key==="top1" ? (m.estimate.value*100).toLocaleString("en-US",{maximumFractionDigits:2})+"%" : m.estimate.value.toLocaleString("en-US",{maximumFractionDigits:4});
  };
  let browser;
  try {
    browser = await chromium.launch({headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
      args:["--disable-gpu","--disable-background-networking"]});
    const context = await browser.newContext({offline:true,acceptDownloads:true});
    const page = await context.newPage();
    page.on("pageerror", e=>errors.push(e.message));
    page.on("request", r=>{if(/^https?:/.test(r.url())) requests.push(r.url());});
    await page.goto(pathToFileURL(input).href);
    const data = JSON.parse(await page.locator("#learning-data").textContent());
    const current = data.runs.find(r=>r.current);
    ok("Current run selected", await page.locator("#lr-run").inputValue()===current.id);
    for(const name of ["assessment.json","final-resources.json"]) {
      const href=await page.locator(`a[href$="/${name}"]`).getAttribute("href");
      ok(`Preserved evidence link / ${name}`,href.startsWith("file:") && (await fs.stat(fileURLToPath(href))).isFile());
    }
    for(const width of [1000,390]) {
      await page.setViewportSize({width,height:900});
      for(const run of data.runs) {
        await page.locator("#lr-run").selectOption(run.id);
        ok(`${width}/${run.id}/description`, (await page.locator("#lr-description").innerText()).includes(run.description));
        const arms = await page.locator("#lr-model option").evaluateAll(ns=>ns.map(n=>n.value));
        for(const arm of arms) {
          await page.locator("#lr-model").selectOption(arm);
          const training = run.training[arm];
          ok(`${width}/${run.id}/${arm}/training count`, await page.locator("#lr-points tbody tr").count()===(training?.points.length??0));
          for(const set of Object.keys(run.evaluations)) {
            await page.locator("#lr-set").selectOption(set);
            for(const scope of ["default","reported"]) {
              await page.locator("#lr-scope").selectOption(scope);
              const r=run.evaluations[set][arm]?.[scope==="reported"?"reported":run.default_scope];
              const cells=await page.locator("#lr-split-metrics tbody tr").nth(1).locator("td").allTextContents();
              for(const [i,key] of ["nll","top1","value_mse"].entries())
                ok(`${width}/${run.id}/${arm}/${set}/${scope}/${key}`, cells[i+1].startsWith(display(r?.metrics[key],key)));
              const expectedDots=["nll","value_mse"].filter(k=>r?.metrics[k]?.status==="available"&&r.metrics[k].estimate.status==="finite"&&typeof training?.updates==="number").length;
              ok(`${width}/${run.id}/${arm}/${set}/${scope}/validation markers`,await page.locator("#lr-policy-chart circle, #lr-value-chart circle").count()===expectedDots);
            }
          }
        }
      }
      await page.locator("#lr-run").selectOption(current.id);
      await page.locator("#lr-model").selectOption("combined-conditioned");
      await page.locator("#lr-set").selectOption("validation_development_komi13");
      await page.locator("#lr-scope").selectOption("default");
      for(const key of ["nll","top1","value_mse"]) {
        await page.locator("#lr-history-metric").selectOption(key);
        const rows=await page.locator("#lr-history-table tbody tr").count();
        ok(`${width}/${key}/history chart and table`, rows>0 && await page.locator("#lr-history-chart > text").count()===rows*2);
      }
      await page.locator("#lr-history-metric").selectOption("nll");
      ok(`${width}/no page overflow`,await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      ok(`${width}/validation limitation`,(await page.locator("#lr-curve-note").innerText()).includes("No intermediate validation curve"));
      ok(`${width}/no invented training agreement`,(await page.locator("#lr-split-metrics tbody tr").first().innerText()).includes("Not recorded"));
      ok(`${width}/three run descriptions`,await page.locator("#lr-run-notes .lr-run").count()===data.runs.length);
      await page.locator("#learning-review").screenshot({path:path.join(output,`learning-${width}.png`)});
      const original=JSON.parse(await page.locator("#expanded-data").textContent());
      for(const game of original.games) {
        await page.locator("#game-select").selectOption(game.id);
        await page.locator("#last-move").click();
        ok(`${width}/${game.id}/game end`,Number(await page.locator("#move-slider").inputValue())===game.moves.length);
        if(game.passes.length) {
          await page.locator("#pass-list button").first().click();
          ok(`${width}/${game.id}/pass jump`,Number(await page.locator("#move-slider").inputValue())===game.passes[0].move_index+1);
        }
      }
    }
    await page.locator("#general-notes").fill("Automated check; not owner feedback.");
    const pending=page.waitForEvent("download");await page.locator("#download-notes").click();
    const download=await pending;await download.saveAs(path.join(output,"notes-test.json"));
    const notes=JSON.parse(await fs.readFile(path.join(output,"notes-test.json"),"utf8"));
    ok("Notes download",notes.general==="Automated check; not owner feedback."&&notes.feedback_sent===false);
    await page.evaluate(()=>localStorage.clear());await page.reload();
    ok("Automated notes cleared",await page.locator("#general-notes").inputValue()==="");
    ok("No JavaScript errors",errors.length===0);ok("No external requests",requests.length===0);
    result.status="passed";
  } catch(error) {
    result.status="failed";result.failure=String(error);process.exitCode=1;
  } finally {
    await browser?.close();await fs.writeFile(path.join(output,"browser-checks.json"),JSON.stringify(result,null,2)+"\n");
    console.log(JSON.stringify({status:result.status,checks:checks.length,errors,requests,failure:result.failure}));
  }
})();
