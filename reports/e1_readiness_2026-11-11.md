# E1 readiness and manual validation

**Later owner decision:** [version-2 admission](e1_kgs_admission_2026-11-11.md)
now admits 241 screened, presumed-human source games with owner-attested research
permission. The inspection below predates that approval. Downstream artifacts
still require regeneration; the endgame decision and overall readiness remain open.

**11 November 2026, UTC: E1 is not ready for baseline training.** The implemented
pipeline passes its checks, but the retained source supplies no admitted training
games and the selected endgame method has no demonstrated coverage on the real
two-pass endings. `kenggo-6yr`, `.10`, `.11` and the pass-only readiness gate `.7`
remain unfinished. This report records the gate result; Beads remains the task
record. It does not revise the source or game contract.

The [inspection manifest](../manifests/e1_readiness_2026-11-11.json) binds the
checked artifacts and validation results to starting commit
`524f270921e1ca338db716121f7f0bee77edd839`. No production code, admission policy,
source record, split assignment or training input changed in this handoff.

## What the checks establish

| Area | Evidence | Readiness consequence |
| --- | --- | --- |
| CPU workspace and behavior | 105 tests passed, including all five enabled native-reference methods; lock, environment, smoke and package build passed. | The implemented components work under the tested CPU configuration. This does not establish GPU or trained-agent readiness. |
| Retained records | 1,000 original records retained; 243 structurally supported games, 53,151 actions and 268 explicit passes. | These are diagnostic records; zero games pass source admission. |
| Ordinary rules | The earlier independent native corpus comparison matched 241 games / 52,900 actions. Native fixture tests passed again today. | The two subsequently accepted structural records were not included in a new native corpus comparison. |
| Endings | All 108 recorded two-pass endings still have no accepted automatic score. | Usable complete-game adjudication remains missing. A recorded source result is a separate supervised label, not an engine certificate. |
| Splits | Fixed date/configuration and whole-game implementation; all 1,000 records are in `excluded`. | There is no populated training/development/confirmation partition or reserved unseen final cohort. |
| Examples and normalization | All ten role directories have verified empty shards; fitted-normalizer counts are all zero. | No real model input population or fitted statistics exists. Positive synthetic tests do not change that result. |
| Rank support | The largest exact-rank count is 80 player-game appearances, at 1d, before admission and splitting. | Even admitting the entire structural subset would not meet the 100-game-side minimum for a rank/phase/time cell. |
| Optional capabilities | Identity and duration validity remain false; dynamic clocks are masked. | Personalization, duration modeling and timing-adaptive search remain deferred. These deferrals alone do not block the population baseline. |

The earlier [canonical report](e1_canonical_2026-11-11.md),
[example report](e1_examples_2026-11-11.md) and
[rules evidence](../docs/rules_engine.md) give the implementation details.
The current sample is a convenience audit, not a population-volume estimate.
More admissible data and untouched cohorts are needed before promising supported
profiles. E0's cell thresholds govern promotion claims; this inspection did not
train or qualify a candidate.

## Decision 1: establish a usable source

The smallest change to the current scope is an evidence-backed KGS admission
revision. Supply a source/operator statement or equivalent retained records that
cover all three points below for a concrete bounded set of games:

1. Historical human-account classification for both sides, with game or
   account-generation/date coverage. State the observational assumption; this
   need not assert that every move was unaided.
2. Ordinary-game classification, using a documented archive-type mapping or
   per-game evidence. Rank and absent event fields do not supply this evidence.
3. A documented basis for this project's local research training and evaluation.
   Record redistribution separately; public access and successful login do not
   establish all downstream uses.

Retain the issuer/source, dates, original text, exact covered source IDs and SGF
hashes, and evidence hashes. The [current eligibility policy](e1_kgs_eligibility_2026-11-10.md)
explains why the retained material cannot provide these affirmative decisions.
An owner authorization to use records does not by itself classify the players.
No message to a server operator has been sent.

If equivalent KGS evidence is unavailable, a concrete alternative is a **bounded
OGS schema preflight**, followed only on success by a separately approved audit.
The [publisher's catalogue](https://za3k.com/ogs/) describes a 2025 collection
distributed with OGS permission for research and AI training. Its
[README](https://za3k.com/ogs/readme.txt) reports a research-use statement from an
OGS representative, but still describes the 2021 release and warns about generated
SGF ranks. This is a promising evidence lead, not a concluded licence or admission
decision for this project's proposed population.

The [2025 release discussion](https://forums.online-go.com/t/2025-ogs-game-dump/56505)
provides acquisition history, contextual historical-rank guidance and a monthly
JSON export. It also records collection failures and interruptions. None of that
establishes complete historical human/ordinary-game classification. Use original
JSON and verify its field meanings. Do not start another live-server crawl.

The proposed preflight is one publisher-provided 1,000-record JSON sample, capped
at 2 MiB compressed / 16 MiB expanded, 300 CPU seconds / 600 wall seconds and no
GPU. Both catalogue editions link the same small sample, so its release and era
must be checked rather than assumed to be 2025. Stop if provenance, human/type
evidence or applicable use conditions cannot be established.

If that succeeds, the proposed replacement audit is **January 2025 UTC OGS**:
even 19×19, both historical **OGS** ranks 15k–1d, explicit Japanese rules and 6.5
komi, 600–1,800 seconds main time plus 3–5 periods of 20–60 seconds. Freeze one
complete publisher month partition and its metadata frame before selecting 1,000
candidates with a recorded seed. Proposed caps are 2 GiB compressed / 8 GiB
expanded, 30 CPU minutes / 90 wall minutes, one request at a time, at least five
seconds between starts and no retries. A cap or shortfall ends the audit without
silently broadening it. This is a budget proposal; no alternate games were fetched.

Changing server also changes rank provenance and may change Japanese-rules
behavior. Validate that behavior, record the source revision, choose new date
splits and preserve both old and new exposure registries. OGS ranks cannot become
KGS ranks. Neither this alternative nor new permission fixes the current ending
method or qualifies a raw profile.

## Decision 2: choose the endgame requirement

The [current contract](../plan/e1_game_and_data_contract.md) requires exact
accepted evidence for frozen-ending scores. Its separate pass judge requires
minimax continuation bounds: it must account for opponent responses, not merely
find a favorable continuation. Future qualification requires at least 99% pass
judgment coverage and an upper bound of 1% for games with a premature or
unjudgeable pass. Those candidate gates have **not** been measured here.

There are three distinct options:

| Option | Concrete input or change | Remaining obligation |
| --- | --- | --- |
| Keep the current contract | Develop broader independently checked life/death, territory and pass proofs within the existing limits. | Representative ending and pass-proof coverage remains unestablished. A larger eye detector alone is not evidence of sufficient coverage. |
| Keep KGS play and run an independent referee pilot | Supply a named independent referee, authorization and finite budget. Use frozen ending snapshots, whole-chain/point maps, provenance and an external authorization catalog. | Final maps address scoring only. The current pass judge has no neutral-map proof path; separate checked pass bounds and their validation are still required. |
| Adopt a formal cleanup game | Explicitly version gameplay in which both players actually enter the pinned KataGo territory cleanup phases. Preserve phase/ko state, actual cleanup actions and full history; validate scoring and learner boundaries. | This changes gameplay and data support. The exact minimax pass gate remains unresolved unless separately revised; any empirical replacement changes the claim and needs a specified validation protocol. |

The existing [native tests](../tests/test_rules_reference.py) demonstrate the
reference's two/four/six-pass transitions and separate no-result handling on
fixtures. The [pinned rules specification](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/docs/rules.html)
defines that formal game. Cleanup on a clone remains diagnostic under the current
contract; it cannot turn an unresolved KGS ending into certified evidence. Actual
cleanup moves would need an explicitly revised contract and cannot be appended to
human policy labels. An estimated neural score is also not an exact pass proof.

The present finding is limited: **no available method has demonstrated the
required coverage**, not a proof that broader certification is impossible. The
read-only exploration did not produce a validated new certifier. Its small
exploratory life-analysis probe is excluded from acceptance evidence.

## What to inspect manually

1. **Source evidence and scope.** Read the two decisions above first. Confirm
   whether KGS evidence can be supplied or an OGS preflight should be authorized.
   Separately choose whether to preserve the exact endgame/pass contract or
   commission a concrete revision. The [E0 fallback rules](../plan/e0_scope_decision.md#fallback-decisions)
   require a recorded revision and do not automatically authorize a new population
   or larger budget.
2. **One scored and one unfinished source record.** The retained diagnostic sample
   in `runs/e1-examples-2026-11-11-v3/diagnostics.json` identifies five already
   inspected games and exact pre-move plies. For the scored
   `2025/1/10/Jabul-yuumsuzuki.sgf`, inspect plies 304 and 305: these are the two
   actual passes. For `2025/11/17/Neptunus-m1.sgf`, inspect plies 0, 5 and 9 and
   the unchanged ten-action EOF. Compare cached records locally; no new download
   is necessary. A source result, final diagram or EOF must not invent an engine
   result or extra pass.
3. **Board orientation, masks and perspectives.** Read the short fixtures in
   [test_encoding.py](../tests/test_encoding.py) and
   [test_examples.py](../tests/test_examples.py). Confirm top-left row-major
   coordinates, action 361 for pass, mover-relative board/value, preserved full
   ko history under short windows, masked dynamic clocks and train-only rank
   masking. The `constant_fractional_clock` fixture covers the normalization
   defect fixed in the preceding bead.
4. **Leakage and empty outputs.** Inspect
   [the frozen split configuration](../manifests/e1_split_config_v1.json) and
   [the split report](e1_canonical_2026-11-11.md#frozen-cohort-and-date-rules).
   Both colors participate in cohort exclusion; validation–test bridges are
   excluded; adaptation uses earlier target-side actions only. All 695 exposed
   KGS groups remain unavailable for final evaluation. Empty reserved-group lists
   and empty shards are an honest failure to populate the corpus, not completed
   evaluation cohorts. The normalizer must still show `fitted: false`.
5. **Scoring and learning boundaries.** Read
   [test_go.py](../tests/test_go.py)'s two-pass, dead/seki, invasion, cap and invalid
   batch fixtures. A resolved result terminates with the proper learner sign; a
   valid administrative cap bootstraps from the saved learner-to-move state;
   unresolved adjudication invalidates the batch. There is no playable UI or
   trained A/B opponent to assess yet.

Run the checks from the repository root with the locked local environment:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export KENGGO_KATAGO_REFERENCE="$PWD/.cache/katago-reference/katago-reference"
uv lock --check
uv sync --locked --check
uv run --locked python -m kenggo.smoke
uv run --locked python -m unittest discover -s tests -v
uv build --no-sources
git diff --check
```

Expected here: **105 tests, no skips**, including five `test_rules_reference`
methods. If the reference binary is missing, follow the
[offline reference build](../docs/rules_engine.md#reproduce-validation); a skipped
native comparison is not a pass. For source-to-artifact reproduction, use the
commands in the canonical and example reports with fresh output directories and
compare their content hashes. These require the retained local inputs; raw games
and arrays are intentionally absent from Git.

The root-owned verification batch used 21.06 child-process CPU seconds and 18.47
wall seconds under 300/600-second caps. This is a run cost, not a performance
comparison. Logs are in `runs/e1-readiness-2026-11-11/`. An earlier artifact-only
inspection and one agent's exploratory probe are accounted separately in the
manifest. There were zero KGS requests, new game downloads, GPU jobs or external
messages in this work; public alternative-source documentation was read.
