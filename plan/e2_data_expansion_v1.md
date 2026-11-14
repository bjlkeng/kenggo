# Larger supervised dataset

Decision: 14 November 2026. Bead: `kenggo-ydo.8`.

The first E2 model learned from **22,916 recorded moves in 106 games**.
Repeating those games for 20 epochs did not add new data. The owner requested
at least 100,000 moves, preferably more, and allowed up to 24 hours to collect
data. This run aims for **one million usable training moves**, including the
existing rows. The actual result depends on access, admission and processing
cost. This is a data build; it does not train another model.

The shared clock starts at **2026-11-14 14:03:57 UTC** and ends 24 hours later.
Collection and preparation stop by **2026-11-15 13:03:57 UTC**, leaving an hour
for verification and the readout. Failed attempts count against the limits.

## What we will collect

Keep the [approved KGS population](e0_scope_decision.md#initial-population):
ordinary ranked, even 19 × 19 games from 2025, ranks 15k through 1d, Japanese
rules, 6.5 komi and the existing main-time/byo-yomi family. Reuse the owner's
research-use permission and presumed-human screening. This new decision
extends collection; it does not change the historical 1,000-game audit.

Start with the existing cached games. For broader coverage, query only the
original survey accounts: the remaining January–October months for the
42 population accounts, and September–October for the four validation
accounts. The frozen query list has 286 account-month pages. The two known
bot accounts remain excluded. No new account is chosen because of a game
result or a model score.

Freeze the query list before making requests. After archive discovery, freeze
the game IDs before downloading their monthly ZIP files. Select by archive
rank, account and month using a fixed deterministic order. Keep at most
25,000 training candidates, 3,000 population-development candidates and
1,500 validation-development candidates. Retain failures without replacing
them with more favorable games.

Use one request worker, at least five seconds between request starts, and
at most two retries for a transient failure. Redirects and retries count as
requests. Stop and report a login requirement if public access is unavailable;
cached games remain usable. No credentials are stored in the run artifacts.

## What stays fixed

- Training dates end before 1 September 2025. Development uses September and
  October. New confirmation and final targets are not inspected.
- Keep the original validation accounts and both participants of every
  reserved final game out of training. Keep all old group aliases, exposure
  records, split assignments and the owner's two excluded games.
- Reject copies matching an old source ID, SGF hash, move-sequence hash or
  canonical trajectory hash. Check new duplicate groups before examples are
  generated. A new copy cannot replace an old split member.
- Keep complete-game legality history. Do not invent a pass at the end of a
  file. Unknown, timeout and forfeit results do not supply value labels.
  Dynamic clocks, move durations and individual identities remain masked.

The output is a derived dataset with a new version and content hashes. Copy
old canonical strings and example files without interpreting their targets;
generate examples only for new admitted training and development games.
Refit the scalar normalizer using the combined training rows only.

## Limits

| Resource | Maximum |
| --- | ---: |
| New HTTP requests | 900 |
| Downloaded response bytes | 512 MiB |
| One HTML response / one ZIP | 2 MiB / 8 MiB |
| Source collection | 6 elapsed hours; 900 CPU seconds |
| Selected SGF bytes / one member | 256 MiB / 1 MiB |
| Declared expanded ZIP contents | 4 GiB |
| Dataset preparation | One CPU worker; 68,000 CPU seconds |
| Generated output / combined canonical file | 16 GiB / 64 MiB |
| Combined training rows | 1,000,000 |
| Combined rows per development role | 250,000 |

The absolute deadline applies to every stage. Complete games are the unit of
addition: a stopped build must not publish a partially encoded game. Keep its
stop reason and unprocessed candidates. A smaller, verified dataset may still
be useful; it must report its actual counts.

## Evidence to report

Report usable games and recorded moves separately from downloaded candidates.
Include valid value labels, passes, endings, exclusions, accounts and rank
coverage. For training and development, report joint cells by mover rank,
move number (1–60, 61–180, 181+) and main time (10–<20, 20–30 minutes).

The [E0 support rule](e0_evaluation_protocol.md#move-prediction-and-recorded-game-outcomes)
requires at least 100 game sides and 30 distinct mover accounts in each
claimed cell. Meeting these data counts does not establish model quality or
playing strength. Unsupported profiles stay deferred. Any later training
comparison needs its own fixed inputs, budget and evaluation plan.

Validation must verify preserved old data, final-test isolation, duplicates,
labels, complete-game counts and train-only normalization. A fresh independent
Codex review must pass before delivery. The review page will summarize the
network, data growth, remaining gaps and a small set of new examples.
