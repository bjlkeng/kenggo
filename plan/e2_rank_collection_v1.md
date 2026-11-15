# More games across ranks

Bead: `kenggo-ydo.10`, under E2. Decision made 15 November 2026.

The owner asked for hundreds of games at each rank and authorized another
24 hours for downloading and processing. The previous expansion has 167,851
training moves in 757 games, but 4k accounts supply about 45% of its moves.
Several weak ranks have fewer than ten games. This run broadens the account
sample while keeping the existing data and game rules.

## What counts as success

Aim for at least **200 distinct training games involving each rank**, from
15k through 1d. Count only players with recorded moves. A same-rank game
counts once for that rank; a game between two different ranks counts once
for each. Report distinct players as well. This collection target differs
from E0's support check: 100 game sides and 30 players within each reported
rank, move-number and time-control group.

Report actual gains and remaining gaps even if the budget ends first. More
games do not establish model accuracy, playing strength or a human rank.
This run performs no model training or GPU work.

## Source and selection

Keep KGS 2025 ordinary ranked, even 19×19 games, both ranks 15k–1d, Japanese
rules, 6.5 komi and the approved 10–30-minute time controls. Preserve source
screening, full-history replay, recorded outcomes, missing-data masks and
the owner's two game exclusions. The
[previous expansion](e2_data_expansion_v1.md) remains unchanged.

Use historical rank hints in the pinned archive metadata to choose more
accounts. Training account hints come only from January–August ranked
19×19 games. A handicap game may identify an account and historical rank;
it cannot enter the training dataset. Exclude bots and games involving
reserved validation or test accounts from training-account discovery.

Freeze 1,000 new account-month pages before network collection:

- 750 training pages, January–August.
- 110 population-development pages, September–October.
- 140 unseen-account-development pages, September–October, from the fixed
  validation cohort. This expands its surveyed accounts without changing
  cohort membership.

Prioritize ranks by their shortfall from 200 training games, or 100
population-development game sides. Cycle through accounts before requesting
another month for the same account within a rank. For training, prefer months
near the observed historical rank. Choose September or October for development
by a fixed hash. Use a fixed hash order for ties; skip cached pages
and duplicate account-months. Pin the baseline counts used for this choice.
The selection code and request record the exact order. Failures do not
trigger replacement queries.

The 1,000-page plan leaves more retry space than a 1,150-page plan under the
same request cap. A 400-page plan reaches fewer accounts. These comparisons
describe planned requests, not expected usable games.

Freeze candidate game IDs after archive discovery and before new ZIP
requests. Balance candidate order by rank, month and archive account, then
interleave the three allowed data roles. Do not use outcomes or model scores
to choose games. Carry forward earlier exposure, base membership and rejected
duplicate fingerprints. Never reopen sealed final targets.

## Access and limits

The owner supplied KGS access. Credentials live in an owner-only file outside
the repository. The collector reads that file through a command option,
keeps session cookies in memory, and never records credentials, cookies or
login request bodies. A four-request access probe succeeded; its 10,151
response bytes count against this run's limits. Its CPU charge is a
conservative 120-second allowance, not a measured runtime.

The new allowance starts **15 November 2026, 00:05:23 UTC**, and ends
**16 November, 00:05:23 UTC**. Stop collection and preparation by
**15 November, 23:05:23 UTC**, reserving the last hour for the readout and
checks. The old failed request and its one-use marker remain in place.

| Resource | Maximum |
| --- | ---: |
| HTTP requests, including probe, login, redirects and retries | 2,400 |
| Response bytes, including the probe | 1 GiB |
| Time between request starts | At least 5 seconds |
| Source collection elapsed time | 6 hours |
| Source CPU, including the probe charge | 900 seconds |
| Retries per request | 2 |
| One HTML response / one ZIP | 2 MiB / 8 MiB |
| ZIP contents by directory size | 4 GiB |
| First-round candidate records | 25,000 train; 5,000 per development role |
| Preparation CPU | 68,000 seconds |
| Complete training moves, including the existing base | 1,000,000 |
| Complete moves in each development role, including the base | 250,000 |
| Selected SGF bytes / one member | 256 MiB / 1 MiB |
| Canonical Parquet / total build output | 64 MiB / 16 GiB |
| GPU allocation | Zero |

Use one serial collector and one builder. Keep complete games when a move
limit is reached. Persist costs and partial results; never reset a run's
budget to continue it. Authentication failure stops acquisition and is
reported separately from data rejection.

## Follow-up after archive discovery

The first 1,000 pages were retrieved without failures. Their frozen training
sample has only 233 games involving 15k and 265 involving 14k, before rules,
clock and replay checks. It is unlikely to supply 200 usable games at both
ranks. No SGF move sequence had been opened when this follow-up was chosen.

The new metadata expands the safe historical account hints from 11 to 74
at 15k, 11 to 99 at 14k, and 14 to 146 at 13k. A fixed second sample of
200 training pages reaches 169 distinct accounts, including 130 not queried
in the earlier cache or first round. These are planned account counts, not
a forecast of usable games.

After the first collector finishes, use its remaining allowance for those
200 pages and their required ZIPs. Freeze a separate request and one-use
execution marker. Keep the same total 2,400 requests, 1 GiB response limit,
900 charged CPU seconds and six-hour source deadline measured from the first
collector's start. Include earlier login, probe, redirect and retry costs;
never start a second full allowance. Keep at least five seconds between
requests across the two collectors.

This amendment permits **at most 5,000 additional training candidates**,
each involving 15k, 14k or 13k. Retain all first-round candidates. The combined
screening cap is therefore 30,000 training candidates and 5,000 in each
development role. The builder's move, CPU, SGF-byte, storage and 50,000-record
limits remain unchanged. Freeze the combined order before reading any SGF
move sequences. Earlier failures are retained; no candidate is replaced.

The follow-up must pass independent review before execution. All first-round
metadata pages must be complete. A monthly ZIP that exhausts its retries
remains explicitly unavailable, together with all affected candidate IDs.
That isolated failure does not prevent collecting from new accounts. Never
retry or replace those missing candidates in the follow-up. Carry the
partial status, failed-request costs and missing IDs into the final readout.
If a global stop or the remaining limits prevent continuation, prepare the
available first-round data and report the shortfall. The overall 24-hour
deadline does not change.

## Checks and owner readout

Independently check all allowed example rows, legal labels, recorded-winner
signs, masks, whole-game membership and training-only normalization. Preserve
every previous membership and example file. Check duplicate variants against
earlier accepted and rejected games without decoding old held-out targets.

The review page will show games and players at every rank, before/after
counts, exclusion reasons, source endings, recorded passes and a fixed sample
of new games. Explain the two supervised losses and the distinction between
rules history and network inputs. E3's planned history comparison remains
0, 8, 32 and 64 previous individual moves, including passes. This collection
retains full source sequences for it.

Finish with the independent Codex review required by `AGENTS.md`. Keep E2's
readiness gate open until a separately fixed model comparison and complete
game checks pass.
