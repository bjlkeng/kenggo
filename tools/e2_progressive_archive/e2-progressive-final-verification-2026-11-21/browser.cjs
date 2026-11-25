'use strict';
const fs=require('fs');
const path=require('path');
const {pathToFileURL,fileURLToPath}=require('url');
const {chromium}=require('playwright');
const OUT=process.env.E2_QA_OUTPUT||__dirname;
if(!path.resolve(OUT).startsWith(path.resolve(__dirname)))throw Error('QA output outside allowed directory');
const evidence={schema:'e2_offline_browser_verification_v1',status:'running',pages:[],networkRequests:[],errors:[],noServer:true,notesCleared:false};
function save(){fs.writeFileSync(path.join(OUT,'browser-evidence.json'),JSON.stringify(evidence,null,2)+'\n');}
function insist(value,message){if(!value)throw Error(message);}
async function values(page,id){if(await page.locator(id).isDisabled())return [];return page.locator(id+' option').evaluateAll(xs=>xs.map(x=>x.value));}
(async()=>{
 let browser;
 try{
  browser=await chromium.launch({executablePath:'/home/bojji/.cache/ms-playwright/chromium-1187/chrome-linux/chrome',headless:true,args:['--no-sandbox','--disable-background-networking','--disable-component-update','--no-first-run','--disable-extensions','--disable-gpu','--use-gl=disabled','--enable-logging=stderr']});
  for(const [pi,filename] of process.argv.slice(2).entries()){
   const result={path:filename,viewports:[],localEvidenceLinks:[],unvisitedExternalLinks:[]};evidence.pages.push(result);save();
   for(const width of [1000,390]){
    const context=await browser.newContext({viewport:{width,height:850},acceptDownloads:true,offline:true});
    const page=await context.newPage();
    page.on('pageerror',error=>evidence.errors.push({filename,width,message:error.message}));
    page.on('console',msg=>{if(msg.type()==='error')evidence.errors.push({filename,width,message:msg.text()});});
    page.on('request',req=>{if(/^https?:/i.test(req.url()))evidence.networkRequests.push(req.url());});
    await context.route('**/*',route=>/^https?:/i.test(route.request().url())?route.abort():route.continue());
    await page.goto(pathToFileURL(filename).href,{waitUntil:'load'});
    await page.locator('#learning-data').waitFor({state:'attached'});
    const data=await page.locator('#learning-data').evaluate(n=>JSON.parse(n.textContent));
    const item={width,runCount:data.runs.length,selections:0,curveChecks:[],screenshots:[],notesDownloaded:false};result.viewports.push(item);
    insist(data.runs.length>=2,'Prior run history is absent');
    for(const runid of await values(page,'#lr-run')){
     await page.selectOption('#lr-run',runid);
     const run=data.runs.find(x=>x.id===runid);
     insist(run.description&&run.notes,'Run description/limitations absent');
     for(const arm of await values(page,'#lr-model')){
      await page.selectOption('#lr-model',arm);
      const training=run.training[arm];
      for(const set of await values(page,'#lr-set')){
       await page.selectOption('#lr-set',set);
       for(const rank of await values(page,'#lr-scope')){
        await page.selectOption('#lr-scope',rank);item.selections++;
        insist((await page.locator('#lr-description').innerText()).length>0,'Run selector did not render description');
        const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);
        insist(!overflow,`Horizontal page overflow at ${width}/${runid}/${arm}/${set}/${rank}`);
       }
      }
      if(training?.checkpoints?.length){
       const selected=await page.locator('#lr-policy-chart [data-selected-step]').getAttribute('data-selected-step');
       insist(Number(selected)===training.selected_step,'Selected checkpoint marker differs from saved data');
       insist(await page.locator('#lr-policy-chart [data-series="training-monitor"]').count()>0 && await page.locator('#lr-policy-chart [data-series="stopping-validation"]').count()>0,'Separate fixed curves absent');
       if(training.points.length)insist(await page.locator('#lr-policy-chart [data-series="online-training"]').count()>0,'Online curve absent');
       insist((await page.locator('#lr-definitions').innerText()).includes('Online losses'),'Online loss definition absent');
       insist((await page.locator('#lr-training-facts').innerText()).includes(training.status.replaceAll('_',' ')),'Training status absent');
       item.curveChecks.push({run:runid,arm,selectedStep:Number(selected),checkpoints:training.checkpoints.length,onlinePoints:training.points.length});
      }
     }
    }
    for(const metric of await values(page,'#lr-history-metric')){
     await page.selectOption('#lr-history-metric',metric);
     insist((await page.locator('#lr-history-table').innerText()).includes('Run / model'),'History metric table absent');
    }
    await page.selectOption('#lr-run',data.runs.at(-1).id);
    await page.selectOption('#lr-scope','default');
    const text=await page.locator('body').innerText();
    insist(/status:/i.test(text),'Overall partial/failure/completion status absent');
    insist(/premature/i.test(text)&&/0.5/i.test(text),'Earlier passing/value concerns absent');
    insist(/masked/i.test(text)&&/rank/i.test(text),'Masks/rank description absent');
    const links=await page.locator('a[href]').evaluateAll(xs=>xs.map(x=>x.href));
    for(const url of links){
     if(url.startsWith('file:')){
      const local=fileURLToPath(new URL(url));
      if(local!==filename){insist(fs.existsSync(local),'Broken evidence link: '+local);result.localEvidenceLinks.push(local);}
     }else if(/^https?:/.test(url))result.unvisitedExternalLinks.push(url);
    }
    await page.locator('#notes').fill('AUTOMATED QA ONLY — REMOVE');
    const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();
    const download=await downloadPromise;const notePath=path.join(OUT,`qa-notes-page${pi}-${width}.json`);await download.saveAs(notePath);
    const note=JSON.parse(fs.readFileSync(notePath,'utf8'));insist(note.notes==='AUTOMATED QA ONLY — REMOVE','Notes download contents differ');
    await page.locator('#notes').fill('');await page.evaluate(()=>localStorage.clear());
    insist((await page.locator('#notes').inputValue())==='','QA notes not cleared');item.notesDownloaded=true;
    const shot=path.join(OUT,`review-page${pi}-${width}.png`);await page.screenshot({path:shot,fullPage:true});item.screenshots.push(shot);
    await context.close();save();
   }
  }
  insist(evidence.networkRequests.length===0,'Page attempted network access');
  insist(evidence.errors.length===0,'Browser script/console errors');
  evidence.notesCleared=true;evidence.status='passed';
 }catch(error){evidence.status='failed';evidence.failure=String(error.stack||error);process.exitCode=1;}
 finally{if(browser)await browser.close();save();}
})();
