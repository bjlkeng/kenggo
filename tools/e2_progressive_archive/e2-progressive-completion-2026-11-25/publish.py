"""Publish a new owner page and a compact chart from saved, verified measurements."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path('/home/bojji/code/kenggo')
OUT = ROOT / 'runs/e2-progressive-completion-2026-11-25'
OLD = ROOT / 'runs/e2-progressive-validation-verification-2026-11-24'
PROD = ROOT / 'runs/e2-progressive-validation-resume-2026-11-24'


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(pin):
    path = ROOT / pin['path']
    if ref(path)['sha256'] != pin['sha256']:
        raise ValueError('Saved input changed: ' + str(path))
    return json.loads(path.read_text())


def main():
    if (OUT / 'review.html').exists():
        raise ValueError('Publish another version; preserve the existing page')
    previous = {'path': str(OLD / 'verification.json'), 'sha256': 'fd62bfe7e46f5fc72d3274f4b989c27f0b38df162adf13e615f10c6a12a51429'}
    old_page = {'path': str(OLD / 'review.html'), 'sha256': '70f2ed92f825d30e4b7426ffa3ca6d9b46d69287d1eb64ba9dad48b103a5e813'}
    verified = read(previous)
    fixed = next(item['evidence'] for item in verified['checks'] if item['name'] == 'Fixed comparisons tier 1' and item['status'] == 'passed')
    correction_ref = ref(OUT / 'verification.json')
    correction = read(correction_ref)
    if correction['status'] != 'passed':
        raise ValueError('Composite verification remains incomplete')
    summary_ref = ref(PROD / 'summary.json')
    summary = read(summary_ref)
    tier = summary['tiers'][0]
    comparison = read(tier['comparison'])
    curves = {}
    for arm in ('control', 'candidate'):
        curves[arm] = []
        for item in tier['training']:
            if item['arm'] != arm:
                continue
            worker = read(item['stage']['summary'])
            for committed in worker['commits']:
                commit = read(committed)
                metrics = read(commit['metrics'])
                curves[arm].append({'round': metrics['round'], 'new_updates': metrics['sampler_step'],
                    'lifetime_updates': metrics['lifetime_updates'], 'metrics': commit['metrics'],
                    'selected': commit['checkpoint']['sha256'] == tier['selected'][arm]['sha256'],
                    'training': {key: metrics['training_monitor'][key] for key in ('nll','top1','value_mse','policy_count','value_count')},
                    'validation': {key: metrics['validation'][key] for key in ('nll','top1','value_mse','policy_count','value_count')}})
    charge = read(ref(OUT / 'resource-charge.json'))
    compact = {'schema':'e2_progressive_owner_readout_v1', 'beads':['kenggo-ydo.16','kenggo-ydo.17','kenggo-ydo.18'],
        'production': summary_ref, 'verification': previous, 'correction': correction_ref,
        'result':'More data improved move prediction. Outcome-value errors exceeded the preset regression limit in two scopes, so further data growth stopped.',
        'method':'12k–1d mover ranks; equal ranks, equal game sides within rank, equal valid moves within side. Policy and value masks are separate. One seed; fixed development comparison, not final testing.',
        'training_rows':{'control':2000000,'candidate':3999997},'training_games':{'control':9135,'candidate':18126},
        'selected':tier['selected'],'curves':curves,
        'fixed':{arm:fixed[arm] for arm in ('parent','control','candidate')},
        'paired':{opponent:{scope:{name:{'estimate':metric['estimate']['value'],'interval95':metric['uncertainty'].get('interval95'),'status':metric['uncertainty']['status']}
                for name,metric in report['metrics'].items()} for scope,report in scopes.items()} for opponent,scopes in comparison['comparisons'].items()},
        'decision':tier['decision'],'cost_charge':charge,
        'review_status':'Saved checks complete. Separate independent completion review is recorded in the versioned manifest.',
        'limits':['No promotion, rank claim or new complete-game qualification. Earlier premature passing remains open.',
                  'Validation selected the checkpoints; the fixed development comparisons guided growth. Confirmation and final targets remain closed.',
                  'The 4M fitting audit is reused by hash. Only the selected supplemental games were independently replayed and re-encoded in the correction.']}
    (OUT / 'readout.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n')
    primary='validation_development_komi13'
    labels={'parent':'Previous 2M model','control':'2M continued','candidate':'4M continued'}
    def table():
        rows=''.join(f'<tr><th>{labels[arm]}</th><td>{fixed[arm][primary]["nll"]:.4f}</td><td>{100*fixed[arm][primary]["top1"]:.2f}%</td><td>{fixed[arm][primary]["value_mse"]:.4f}</td></tr>' for arm in labels)
        return '<div class="completion-table"><table><thead><tr><th>Selected model</th><th>Move loss ↓</th><th>Move matching ↑</th><th>Value error ↓</th></tr></thead><tbody>'+rows+'</tbody></table></div>'
    actions='''<ol><li><strong>.16 — Check the training curves.</strong> In the learning section, choose the 4M candidate and compare training with stopping validation. The retained candidate is interval 3; the 2M control is interval 1. This checks that the final weights were selected from validation rather than the last training step.</li>
<li><strong>.16 — Read the fixed comparison and value warnings.</strong> More data helped move prediction, but two value-error increases blocked another doubling. This explains the stop while time remained.</li>
<li><strong>.17 and .18 — Review the recovery note, then record an E2 next-step preference.</strong> The original allowance, failed attempts and protected holdouts were preserved; seven new validation-only games repaired missing coverage. I recommend diagnosing value accuracy before another data expansion. This is a proposed next experiment, not an automatic launch.</li></ol>'''
    reg_rows=''
    for scope,label in [('population_development_komi1','Population development, 0.5 komi'),('validation_development_komi13','Unseen accounts, 6.5 komi')]:
        m=compact['paired']['parent'][scope]['value_mse']; lo,hi=m['interval95']
        reg_rows+=f'<tr><th>{label}</th><td>+{m["estimate"]:.4f}</td><td>[+{lo:.4f}, +{hi:.4f}]</td></tr>'
    section=f'''<section id="completion-readout"><h2>More data helped move prediction; value accuracy stopped further growth</h2>
<p>We doubled fitting data from 2,000,000 to 3,999,997 moves in 18,126 complete games. Both models completed six matched intervals, or 187,500 new updates and 24 million move presentations each. A separate 52-game validation sample selected the checkpoints.</p>
<p><strong>Status:</strong> production and saved-evidence checks finished. E2 readiness remains open. The preset stopping rule fired; the seven-day budget was not exhausted.</p>
<h3>Fixed comparison: unseen accounts, 6.5 komi</h3><p>All three models use the same 27,268 in-scope positions from 127 complete games. Value error uses 26,150 valid winner labels. Move loss is negative log-likelihood (NLL); value error is mean squared error (MSE) in predicting the eventual winner from the mover’s view.</p>{table()}
<p>The 4M model reduces NLL by 0.1015 against the equally trained 2M procedure (paired 95% interval: 0.0763–0.1268). Against the previous model, the reduction is 0.0976 (0.0640–0.1311). These clear the declared useful reduction of 0.02. Selected checkpoints differ: 4M uses 93,750 new updates; 2M uses 31,250, from the same six opportunities.</p>
<h3>Why another doubling did not run</h3><p>Value error increased against the previous model in two fixed scopes. Each interval’s lower bound exceeds the allowed +0.02. Higher value error is worse. This is a separate outcome from improved human-move prediction.</p>
<div class="completion-table"><table><thead><tr><th>Scope</th><th>Value-error change</th><th>Paired 95% interval</th></tr></thead><tbody>{reg_rows}</tbody></table></div>
<p>{html.escape(compact['method'])} The comparisons account for shared games and accounts; they do not include variation across training seeds. The wider comparison retains all 539 games / 120,450 positions, with 111,591 positions in scope. No new Human SL evaluation or complete-game panel ran.</p>
<h3>What was completed</h3><ul><li><strong>.16 — Data and training:</strong> one doubled dataset, six paired intervals, selected checkpoints and all 12 fixed reports completed. No second data tier was started.</li><li><strong>.17 — Recovery:</strong> reused the saved 4M dataset after the interrupted supervisor. Every failed attempt and cumulative charge remains recorded; no allowance reset.</li><li><strong>.18 — Validation coverage:</strong> seven usable validation-only games supplied a missing 11k/6.5-komi group. Two enter the new 52-game stopping sample; all seven IDs and duplicate fingerprints stay barred from fitting.</li></ul>
<h3>Verification and costs</h3><p>The earlier verifier passed ten checks but stopped one composite check because it compared relative and absolute parent paths as text. A new helper verifies both hashes and the resolved file identity. Its regression test failed before the fix and passed afterward. The previously skipped sample, mask, normalization, disjointness and supplemental replay checks now pass. The earlier failed page and evidence are unchanged.</p>
<p>Production finished after {charge['production_elapsed_seconds']/3600:.2f} elapsed hours from the original launch. The cumulative charge, including verification and an explicit 5-CPU-hour upper charge for root completion, is {charge['cumulative_cpu_upper_seconds']/3600:.2f} CPU hours and {charge['cumulative_gpu_seconds']/3600:.2f} allocated GPU hours. CPU totals contain conservative bounds, not just measured use. The original deadline remains 28 November 2026, 14:13 UTC. Final local delivery timing and output size are recorded separately.</p>
<p><a href="{OUT / 'verification.json'}">Corrected verification</a> · <a href="{OLD / 'verification.json'}">Preserved failed attempt</a> · <a href="{OUT / 'readout.json'}">Saved chart data</a> · <a href="{OUT / 'resource-charge.json'}">Cumulative charge</a></p>
<h3>Suggested next work</h3><p>Hypothesis: winner prediction is generalizing poorly even while move prediction improves. Inspect value calibration and errors by komi, rank and game phase, then test regularization or a smaller learning rate with a checkpoint rule that considers value accuracy as well as move loss. The present experiment does not identify the cause. Another data expansion and complete-game qualification need their own approved plan.</p>
<h3>Your review</h3>{actions}</section>'''
    page_path=Path(old_page['path'])
    if ref(page_path)!=old_page:
        raise ValueError('Earlier page changed')
    page=page_path.read_text()
    page=page.replace('<p><strong>Status:</strong> completed. No useful supported data-growth gain; stop and report next-step hypotheses</p>', '<p><strong>Status:</strong> production completed. Move prediction improved; the value-error guard stopped further growth. See the completion readout below.</p>',1)
    page=page.replace('<p>Used 99.81 elapsed hours, 59.17 CPU hours and 18.73 allocated GPU hours.', '<p>Production accounting, before later verification and root completion: 99.81 elapsed hours, 59.17 charged CPU hours (including conservative bounds) and 18.73 allocated GPU hours.',1)
    page=page.replace('</header>',section+'</header>',1)
    page=page.replace('<style>','<style>.completion-table{max-width:100%;overflow:auto}.completion-table table{width:100%;border-collapse:collapse;text-align:left}.completion-table th,.completion-table td{padding:8px;border-bottom:1px solid #d6dfdc;vertical-align:top}',1)
    page=page.replace('<section id="review-actions"><h2>What to review</h2>', '<section id="review-actions"><h2>What to review</h2><p>Use the three numbered actions in the completion readout above. The following original checklist and saved run details are retained for context.</p>',1)
    (OUT / 'review.html').write_text(page)
    (OUT / 'publication.json').write_text(json.dumps({'schema':'e2_progressive_completion_publication_v1','earlier_page':old_page,'production':summary_ref,'verification':correction_ref,'readout':ref(OUT/'readout.json'),'page':ref(OUT/'review.html'),'shared_learning_component':'Preserved verbatim from the pinned earlier page, including saved training/validation curves and all four historical runs.'},indent=2)+'\n')
    print(json.dumps({'page':ref(OUT/'review.html'),'readout':ref(OUT/'readout.json')}))

if __name__ == '__main__':
    main()
