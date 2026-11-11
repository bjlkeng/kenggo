# E0 scope decision

Decision date: 6 November 2026. Bead: `kenggo-n5n.1`.

Use KGS as the first candidate source for a population model of human moves on 19 x 19 Go. Propose KGS ranks **15 kyu through 1 dan**, subject to the E1 audit. These are conditioning labels from recorded games; the model's playing strength is uncalibrated. No dataset access, usable coverage or human participants have been secured by this decision.

This note implements the scope part of [E0](go_project_plan_v0.5.md#e0-fix-scope-and-evaluation). Evaluation design and budgets remain in `kenggo-n5n.2` and `kenggo-n5n.3`. The [selected stack](go_project_plan_v0.5.md#technology-stack) is unchanged.

**10 November E1 support decision:** the [1,000-candidate audit](../reports/e1_kgs_audit_2026-11-10.md) found 241 metadata-compatible records, with human status, identities and legal replay still unverified. The proposed 15k, 10k, 5k, 1k and 1d raw profiles all remain deferred; no rank/phase/time cell yet establishes the required support. The [game/data contract](e1_game_and_data_contract.md) preserves this population and fixes numerical qualification gates. Personalization, duration modeling and timing-adaptive search remain deferred; version 1 masks duration targets and dynamic-clock inputs. These findings do not authorize broader dates, rules, ranks or compute.

**11 November owner-approved admission revision:** use screened, presumed-human ordinary-ranked KGS games under [eligibility version 2](../reports/e1_kgs_admission_2026-11-11.md). Explicit Ranked metadata, known-bot exclusions and absence of contradictory event/type evidence support a documented presumption, not verified human control. The owner states that permission to use is available; record that as the local research training/evaluation basis. The frozen sample now has 241 source-admitted games. Preserve all existing population settings, masks and final quarantine; rank support, downstream artifact regeneration and endgame readiness remain unmet.

## Initial population

The following limits are project choices for the first audit, not measured properties of the corpus.

| Dimension | Proposed scope |
| --- | --- |
| Source and access route | Public KGS game archives and their original SGFs, with source metadata where available. E1 must establish a permitted, reproducible discovery and download route before collecting the sample. |
| Era | Games started from **2025-01-01 through 2025-12-31**, using the source date and recording its timezone or date ambiguity. This gives a fixed, completed year after the July 2024 Human SL release. |
| Players and ranks | Two presumed-human players in presumed ordinary ranked games under the approved version-2 screening policy, with both historical KGS ranks known and within **15k–1k or 1d**. Retain each exact rank and the pair of ranks. Missing, provisional or ambiguous ranks need an explicit E1 policy and do not establish support for an exact-rank profile. |
| Board and handicap | **19 x 19, even games**, Black first, no handicap or setup stones. Any 9 x 9 work is a separately labeled debugging experiment and supplies no evidence of 19 x 19 strength. |
| Time-control family | Main time **10–30 minutes per player**, followed by Japanese byo-yomi with **3–5 periods of 20–60 seconds**. Require known time settings; retain the exact tuple. These bounds define the initial family rather than asserting a server-wide definition of non-blitz play. Exclude other clock systems and settings initially. |
| Rules and komi | **KGS Japanese rules, 6.5 komi to White**, pending E1's exact legality and endgame contract. Require explicit metadata or a documented source-specific interpretation; do not infer rules from the server name alone. |
| Exclusions | Bots, reviews, teaching and demonstration games, rengo, simultaneous exhibitions, tournaments, duplicate records, corrupt games and records outside the above scope. Keep exclusion reasons. |

The proposed range includes weaker kyu play and a modest dan endpoint while limiting the first data requirement. Candidate profiles to investigate are **15k, 10k, 5k, 1k and 1d**; records at intermediate ranks remain useful for conditioning. E1 may narrow this list. Each retained rank/context combination needs data support and E5 complete-game qualification; interpolation, extrapolation and distinct playing strengths are not assumed. The release goal remains at least two measurably different supported settings under the plan's separate release criteria.

KGS documents game-start rank snapshots, game types, rules, komi and time settings. It also describes archived SGF paths and distinguishes the server's six-month game-summary window from the online archives. This supports investigating KGS, but does not establish archive completeness or research reuse rights. The schema warns that only rank information in a game-summary player's flags is meaningful; bot exclusion needs other verified evidence. ([KGS data types](https://www.gokgs.com/json/dataTypes.html))

The proposed clock family uses a system KGS documents. Missing per-move clocks do not automatically exclude an otherwise eligible game: E1 must preserve missingness and determine whether elapsed times are recoverable. The nominal clock settings alone do not establish valid timing labels. ([KGS time systems](https://www.gokgs.com/help/timesystems.html?helpLocale=en_US))

KGS's Japanese scoring uses territory and captures, excludes seki territory, and leaves life/death disputes to a neutral third player. Its repetition behavior also needs explicit verification. E1 must resolve these details before choosing an engine; silently substituting Chinese area scoring would change the population's game contract. ([KGS rules](https://www.gokgs.com/help/rulesets.html), [repetition warning](https://www.gokgs.com/json/downstream.html))

## External baseline and strength limits

KataGo Human SL was released in July 2024. The 2025 window follows the plan's preference for post-release evaluation games. E2 must pin the actual engine and checkpoint, verify game dates and provenance, and still disclose unknown training overlap; a date filter alone does not validate the entire comparison. ([Human SL guide](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide))

At the plan's pinned implementation, KataGo's ordinary rank profiles use KGS metadata, a 2020-03-01 date, 20 minutes main time and five 30-second byo-yomi periods. Thus the proposed data overlaps its clock family but differs in era and may differ in exact timing and opponent rank. E2 must report these context differences on its common eligible evaluation set. The source profile is not evidence of this project's strength. ([Pinned profile implementation](https://github.com/lightvector/KataGo/blob/cbeab3b0fb75548fc160428de4db02b51079dfeb/cpp/neuralnet/sgfmetadata.cpp))

Begin with population/rank conditioning. Individual personas and duration prediction remain conditional on separate coverage evidence. B still starts from random weights against qualified frozen A profiles, using A's trained unknown-opponent-rank path. This scope does not assign B a rank or promise that the weakest proposed A profile is beatable by an initial B.

## Human access

No recruiting route, participant pool or human-match budget is confirmed. The options are owner-provided contacts, consenting KGS players, or a Go club able to supply players with verified KGS ranks. Other servers' and clubs' rank labels are not interchangeable with KGS ranks. Any future recruitment must record the cohort, current rank evidence, matching rules/time controls and repeated opponents; archive access provides no access to those players for live matches.

`kenggo-n5n.2` must size the human-match budget for [E7's calibration criterion](go_project_plan_v0.5.md#e7-evaluate-transfer-and-hand-off): a 95% score interval inside 40–60% against the corresponding server cohort, with adjacent-rank checks and repeated-opponent dependence accounted for. If access or budget is unavailable, report engine-relative strength and label profiles uncalibrated. Training on 5k records alone does not justify displaying “plays at 5k.”

## Evidence required from the 1,000-game E1 audit

`kenggo-6yr.2` owns the audit. Freeze the discovery frame, sampling method and seed before collecting **1,000 candidate games**; cover the proposed ranks and dates and record any account-selection bias. Count attempted, retrieved, excluded and usable records separately, with rank/time-control breakdowns. Do not silently replace failures until only 1,000 clean games remain. Preserve an audit manifest and keep inspected games out of reserved final evaluation under E1's split policy.

| Question | Required evidence and consequence |
| --- | --- |
| Can the records be obtained and used? | Verify access and reuse terms, discovery coverage, retrieval limits, source identifiers, dates, SGF hashes and metadata provenance. Documentation review here did not download a corpus or verify permission. A failed access check requires the fallback below. |
| Does the population exist in sufficient numbers? | Report games and distinct accounts by exact historical rank, rank pair, month and clock tuple, including exclusions. Check bot-identification evidence and distinguish absent metadata from a verified human record. Estimate usable volume after the proposed filters and account/time splits; raw move counts alone cannot establish support. |
| Can accounts support the planned splits and extensions? | Verify stable source account identifiers, games per account and chronology. Deduplicate before chronological whole-game splits and reserve disjoint account cohorts as required by [E1](go_project_plan_v0.5.md#e1-establish-the-data-and-game-contract). Missing identities disable personalization for those accounts; inability to enforce the core split contract blocks readiness. |
| Are replay and results trustworthy? | Inspect played main lines, setup/handicap indicators, explicit rules/komi, captures, ko/repetition, resignation, result reasons and unfinished games. Preserve original results. `kenggo-6yr.3` and `.4` must settle and validate full-history legality, dead-stone handling, disputes, scoring and termination versus administrative truncation. |
| Can raw policies learn usable endings? | Count explicit passes and scored, resigned and incomplete endings by rank and result type. Check whether dead-stone/scoring information survives SGF export. File endings never supply an invented pass or proof of termination. Insufficient coverage narrows or defers affected raw profiles; a completion heuristic is a separate profile. |
| Are clocks and outcomes usable as labels? | Measure missing/ambiguous clocks, overtime transitions and valid elapsed-time coverage. SGF BL/WL are post-move values, so target times and post-move clocks must not enter pre-move inputs. Keep unknown durations masked; initially mask unknown, timeout and forfeit outcomes from value loss as the plan requires. ([SGF properties](https://www.red-bean.com/sgf/properties.html)) |

The audit must finish with supported, narrowed or deferred capabilities and their evidence. It does not qualify an opponent or calibrate strength. E0's remaining beads define metrics and budgets; E1 finalizes data-dependent support and behavior thresholds before qualification. The separate `kenggo-6yr.7` readiness gate stays open if a core prerequisite fails, even when a negative audit report is complete.

## Fallback decisions

| Failure | Response |
| --- | --- |
| Public access, discovery or reuse is unavailable | First seek an owner-supplied or explicitly permitted KGS export with equivalent provenance and metadata. If none is available, record a source-rescope decision before a new audit. No substitute corpus is assumed available, and ranks from another source must not be relabeled as KGS ranks. |
| Too few usable games in 2025 | First narrow the rank/profile range to supported groups. If that defeats the intended comparison, consider a separately recorded expansion to **2024-08-01 through 2026-10-31** within KGS, still after the baseline release. Reassess era effects and fix the revised sampling frame and splits before model selection; do not silently pool eras. |
| Too few games in the chosen clock/rules family | Revisit one explicit source-compatible family using audit counts. Version any changed rules, komi or time limits and rerun the affected audit checks. Unknown metadata does not become the proposed default. |
| Identities or timing are inadequate | Defer the unsupported personalization or duration/adaptive-search extension. The core rank-conditioned baseline can proceed only if its own split, input and result contracts still pass. |
| Pass/endgame coverage or rules compatibility fails | Acquire suitable records or narrow/defer the affected raw profiles. Keep E1 readiness blocked for an unresolved core game contract, and keep E6 blocked until E5a qualifies a raw opponent pool. |
| Humans cannot be recruited or funded | Continue permitted engine-relative evaluation. Human-rank labels remain unavailable until calibration evidence exists. |

These fallbacks require a recorded revision and evidence in the owning bead; they do not automatically authorize a broader population or a larger compute budget. This decision uses documentation only and runs no training. The 1,000-game audit and separate pilot budgets remain those in the project plan.
