# E1 source-access preflight

10 November 2026 | Audit `kenggo-6yr.2` | Access prerequisite `kenggo-6yr.8`

**Later on 10 November:** the owner supplied access and authorized bounded acquisition. The [completed candidate audit](e1_kgs_audit_2026-11-10.md) supersedes this report's acquisition status and unmeasured counts. The observations below describe the earlier unauthenticated preflight; its terms/robots evidence is preserved. E1 readiness remains open.

**The 1,000-game audit has not started.** No permitted, accessible candidate frame was established in this preflight. Both checked KGS archive landing pages returned login forms, and no local corpus was present. Keep `kenggo-6yr.2` open behind the access prerequisite and keep E1 readiness (`kenggo-6yr.7`) blocked. This report records a negative source-access finding, not a completed game audit or evidence that the target population does not exist.

The [E0 scope](../plan/e0_scope_decision.md) remains 2025 KGS ranked human games, both historical ranks within 15k–1k or 1d, even 19 x 19 boards without setup stones, the selected byo-yomi family, and KGS Japanese rules with 6.5 komi. The [audit budget contract](../plan/e0_budgets_and_handoff.md#audit-and-throughput-probe) requires a permitted route and a frozen sampling frame before collecting 1,000 candidate IDs. This preflight neither changes that population nor substitutes another source.

## Observations

The [capture manifest](../manifests/e1_source_preflight_2026-11-10.json) records all 12 direct HTTP observations with URLs, UTC times, status codes, response sizes and SHA-256 hashes. Raw responses and the request log remain local in ignored `runs/e1-source-preflight-2026-11-10/`.

| Observation | Evidence and consequence |
| --- | --- |
| KGS archive access | One unauthenticated GET each to [the original archive URL](https://www.gokgs.com/gameArchives.jsp) and [the URL linked by the KGS FAQ](https://www.gokgs.com/archives.jsp) returned HTTP 200 with identical **KGS Web Login** forms, not archive listings. A successful HTTP status is not evidence of corpus access. No login attempt, account enumeration or game request followed. |
| Automated archive access | [KGS robots.txt](https://www.gokgs.com/robots.txt) disallows `/gameArchives.jsp` for all user agents. The first archive landing-page request was in the initial fixed preflight batch before its robots response was inspected; no crawl of that path followed. The separately linked landing page was checked once to distinguish an obsolete URL from the current documented route. Neither returned a candidate frame. |
| Reuse terms | The inspected [KGS terms](https://www.gokgs.com/tos.jsp) describe service conduct and privacy. They do not establish permission for this project's planned bulk research reuse. That permission remains unresolved; this observation does not establish that permission is impossible to obtain. No external contact was sent. |
| Online API documentation | [KGS data types](https://www.gokgs.com/json/dataTypes.html) document game summaries, game-start timestamps/ranks, game types and time settings. They distinguish a six-month live-summary window from longer archive retention. These are schema descriptions, not verified coverage of 2025 games. The [JSON client example](https://www.gokgs.com/json/index.html) uses a login session; no service session was opened. |
| Public mirror | The [u-go.net main catalogue](https://u-go.net/gamerecords/) describes games involving a 7d-or-stronger player or two 6d players, with comments stripped and games containing variations omitted. Its newest listed period is **2019-04**. The linked [4d+ catalogue](https://u-go.net/gamerecords-4d/index.html) also ends at **2019-04**. These catalogues do not supply the frozen 2025 population; no archives were downloaded. The maintainer's stated permission to use KGS files is not evidence of this project's permission or coverage. |
| Inconclusive documentation request | `https://u-go.net/robots.txt` returned HTTP 500. Record it as a failed policy lookup, not permission. Only public catalogue/about pages were inspected. |
| Local inputs | Neither `data/` nor `datasets/` existed in this checkout. The owner was asked for a permitted export or authorized acquisition route; none had been supplied when this report was prepared. This says nothing about data stored elsewhere. |

The other requests were navigation/about/help checks and remain in the manifest even where they added no usable evidence. No alternate endpoint was used to retrieve game data, and no access restriction was bypassed. Web-search results helped locate documentation but did not supply game records or a sampling frame.

## What remains unmeasured

| Audit quantity | Result |
| --- | --- |
| Target sample | 1,000 candidate games; unchanged |
| Frozen candidate frame, sample seed and selected IDs | Not established |
| Game retrievals attempted / SGFs downloaded / games parsed | **0 / 0 / 0** |
| Eligible games and exclusions by reason | **Unknown**, not zero |
| Exact rank/rank-pair, month, clock and rules/komi coverage | **Unmeasured** |
| Stable account identity, games per identity and split feasibility | **Unmeasured** |
| Bot, teaching, review, tournament, duplicate and corrupt-record exclusions | **Unmeasured** |
| Explicit pass counts and scored/resigned/incomplete endings | **Unmeasured**; no EOF pass or terminal result inferred |
| Clock/overtime validity, elapsed-time labels and missingness | **Unmeasured**; no missing value treated as zero |
| Supported raw profiles, personas and timing capability | **Not established** |

There are no per-game provenance rows because no game was selected or retrieved. The manifest's observations are HTTP documentation/access checks, not 12 games or a substitute 1,000-game sample. No rank coverage, player strength, model qualification or release claim follows from this work.

## Costs and reproducibility

The preflight budget was recorded in the bead before direct requests: at most **40 requests**, **2 MiB per response**, **40 MiB of stored response bodies**, **60 minutes elapsed**, **15 CPU-minutes** and **zero GPU work**. Responses were requested serially with a one-second delay, a 20-second socket timeout, no retries and no automatic redirects. This was a bounded inspection of named pages, not a general crawler.

The completed capture used **12 requests and 110,965 response bytes**, comprising 11 HTTP-200 responses and one HTTP-500 response. The HTTP observation window was **14:05:29.723–14:08:43.895 UTC**. Instrumented request time summed to **3.504723 wall-seconds and 0.060095 process-CPU-seconds**. These timings cover request/body/hash work only: interpreter startup, local inspection, authoring, external web-search execution and time waiting for owner input are not included. Total CPU cost for those other activities was not measured. No SGF acquisition, GPU work or human-player participation occurred. These costs are separate from the SL/RL pilots and make no throughput claim.

To verify the stored observations locally, run this from the repository root:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path('manifests/e1_source_preflight_2026-11-10.json').read_text())
root = Path(manifest['local_response_directory'])
for row in manifest['http_observations']:
    body = (root / row['body_path']).read_bytes()
    assert len(body) == row['body_bytes']
    assert hashlib.sha256(body).hexdigest() == row['sha256']
print('Verified', len(manifest['http_observations']), 'stored HTTP observations')
PY
```

The ignored response bodies are available in the working checkout, not distributed in Git. A fresh clone can inspect the manifest and source links but cannot reproduce the old bytes from hashes alone. New permitted documentation checks must record new timestamps/hashes; changed content is a new observation. Do not replay the restricted archive requests as a crawler. Corpus acquisition requires the access prerequisite below.

## Resume the game audit

`kenggo-6yr.8` owns the missing input: a permitted KGS export with equivalent provenance, or an authorized discovery/download route with applicable reuse and automation conditions. If neither exists, record a source-rescope decision before a new audit. Older strong-player archives cannot silently replace the agreed population or support the same claims. Other server ranks must not be relabeled as KGS ranks.

Once that prerequisite passes, resume `kenggo-6yr.2`: freeze the actual frame, sampling algorithm/seed, population coverage and account-selection bias; set finite acquisition/retry/CPU/storage limits; retain all 1,000 candidate IDs and failed/excluded attempts; then measure the quantities above and record supported, narrowed or deferred capabilities. Keep inspected games out of reserved final evaluation. Access alone does not pass the game audit, and the game audit alone does not pass E1's rules, splits and readiness gates.
