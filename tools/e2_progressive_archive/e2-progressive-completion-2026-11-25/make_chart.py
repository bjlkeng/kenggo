"""Draw a small in-thread chart directly from the saved owner readout."""
import html
import json
from pathlib import Path

out=Path('/home/bojji/code/kenggo/runs/e2-progressive-completion-2026-11-25')
data=json.loads((out/'readout.json').read_text())
colors={'control':'var(--chart-1)','candidate':'var(--chart-2)'}
labels={'control':'2M data','candidate':'4M data'}

def curves(metric,title):
    values=[point[split][metric] for points in data['curves'].values() for point in points for split in ('training','validation')]
    low=0 if metric=='value_mse' else 1.6
    high=1.6 if metric=='value_mse' else 2.8
    x=lambda epoch:48+epoch*60
    y=lambda value:190-(value-low)/(high-low)*155
    content=[]
    for i in range(5):
        v=low+(high-low)*i/4;yy=y(v)
        content.append(f'<line x1="48" x2="408" y1="{yy:.2f}" y2="{yy:.2f}" class="grid"/><text x="41" y="{yy+4:.2f}" text-anchor="end">{v:.1f}</text>')
    for epoch in range(7):content.append(f'<text x="{x(epoch)}" y="210" text-anchor="middle">{epoch}</text>')
    for arm,points in data['curves'].items():
        for split in ('training','validation'):
            path=' '.join(f'{x(p["round"])},{y(p[split][metric]):.3f}' for p in points)
            dash=' stroke-dasharray="6 4"' if split=='validation' else ''
            content.append(f'<polyline points="{path}" fill="none" stroke="{colors[arm]}" stroke-width="2.5"{dash}/>')
            if split=='validation':
                for point in points:
                    if point['selected']:content.append(f'<circle cx="{x(point["round"])}" cy="{y(point[split][metric]):.3f}" r="4.5" fill="{colors[arm]}"><title>{labels[arm]} selected interval {point["round"]}</title></circle>')
    return f'<div class="panel"><h3>{title}</h3><p>Fixed training sample and separate stopping validation</p><svg viewBox="0 0 440 235" role="img" aria-label="{html.escape(title)} by training interval">'+''.join(content)+'<text x="230" y="230" text-anchor="middle">Completed training interval</text></svg></div>'

def bars(metric,title,factor,maximum,suffix):
    primary='validation_development_komi13'
    content=[]
    for i,(arm,label) in enumerate([('parent','Previous 2M'),('control','2M continued'),('candidate','4M continued')]):
        value=data['fixed'][arm][primary][metric]*factor;y=50+i*57;w=value/maximum*210
        color='var(--chart-3)' if arm=='parent' else colors[arm]
        content.append(f'<text x="5" y="{y+18}">{label}</text><rect x="118" y="{y}" width="{w:.3f}" height="29" rx="3" fill="{color}"/><text x="{125+w:.3f}" y="{y+19}">{value:.2f}{suffix}</text>')
    return f'<div class="panel"><h3>{title}</h3><p>Same unseen-account 6.5-komi development positions</p><svg viewBox="0 0 430 235" role="img" aria-label="{html.escape(title)} for the three selected models">'+''.join(content)+'</svg></div>'

page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E2 saved training and validation results</title><style>
:root{--foreground:#172e36;--muted-foreground:#607078;--border:#ccd6d8;--chart-1:#337fc4;--chart-2:#1a9776;--chart-3:#987bb2;--font-sans:system-ui,sans-serif}body{margin:0;color:var(--foreground);font:14px/1.45 var(--font-sans)}h3{margin:0;font-size:16px}p{margin:5px 0 12px;color:var(--muted-foreground)}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.panel{min-width:0}svg{width:100%;height:235px;display:block}svg text{font:13px var(--font-sans);fill:var(--foreground)}.grid{stroke:var(--border);stroke-width:.7}.legend{display:flex;gap:15px;flex-wrap:wrap;margin:6px 0 16px}.swatch{display:inline-block;width:22px;border-top:3px solid;vertical-align:middle;margin-right:6px}a{color:var(--chart-1);overflow-wrap:anywhere}details{margin-top:12px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:5px;text-align:right;border-bottom:1px solid var(--border)}th:first-child,td:first-child{text-align:left}@media(max-width:520px){.plots{grid-template-columns:1fr}}
</style><body><div class="legend"><span><i class="swatch" style="border-color:var(--chart-1)"></i>2M data</span><span><i class="swatch" style="border-color:var(--chart-2)"></i>4M data</span><span>Solid: training sample · dashed: validation · dot: selected checkpoint</span></div><div class="plots">'''
page+=curves('nll','Move loss (NLL) · lower is better')+curves('value_mse','Winner-prediction error · lower is better')
page+=bars('top1','Move matching · higher is better',100,50,'%')+bars('value_mse','Fixed value error · lower is better',1,1.6,'')
page+='''</div><p>Curves use saved checkpoints, a fixed training sample and 52 separate stopping-validation games. The fixed comparison below them uses 127 other whole games: 27,268 in-scope policy rows and 26,150 valid value labels. All panels use 12k–1d ranks with equal-rank/game-side weighting. These development results are from one training seed.</p>'''
page+=f'<p><a href="{out / "review.html"}">Full review: earlier runs, all fixed comparisons, recovery and limitations</a> · <a href="{out / "readout.json"}">Saved measurements</a></p>'
page+='<details><summary>Exact fixed-comparison values</summary><table><tr><th>Model</th><th>NLL</th><th>Matching</th><th>Value error</th></tr>'
for arm,label in [('parent','Previous 2M'),('control','2M continued'),('candidate','4M continued')]:
    m=data['fixed'][arm]['validation_development_komi13'];page+=f'<tr><td>{label}</td><td>{m["nll"]:.4f}</td><td>{100*m["top1"]:.2f}%</td><td>{m["value_mse"]:.4f}</td></tr>'
page+='</table></details></body></html>'
(out/'charts.html').write_text(page)
print(out/'charts.html')
