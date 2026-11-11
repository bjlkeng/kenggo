# E1 game and data contract

Version 1 | 10 November 2026 | Bead `kenggo-6yr.3`

Use **KGS Japanese ordinary play with certified adjudication of a frozen ending**. Keep the source population at even 19 x 19, Black first, 6.5 komi and the [E0 rank/date/clock scope](e0_scope_decision.md). Two passes request adjudication; they do not supply a score. An unsupported ending stays unresolved. The [versioned manifest](../manifests/e1_game_contract_v1.json) pins the reference, limits, qualification thresholds and evidence.

This is the policy input to the shared engine, canonical records, examples and match runner. It does not implement or validate those components. The [audit](../reports/e1_kgs_audit_2026-11-10.md) found 241 metadata-compatible records among 1,000 candidates. The subsequent owner-approved [version-2 source policy](../reports/e1_kgs_admission_2026-11-11.md) admits those games as presumed human and ordinary ranked, using owner-attested research-use permission. That revision preserves grouping, quarantine and missing-data masks; it supplies no verified-human or generated-ending claim. Every proposed raw rank profile remains deferred, and E1 readiness remains open for data support, downstream artifacts and usable ending/pass judgments.

## Reference and compatibility boundary

Pin **KataGo v1.18.2**, commit `fd0723fdbc0e9d82cf269c9630af8c27c57c07c4`, as the implementation reference for *Supported Go Rules, Version 3*. Its [specification](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/docs/rules.html) describes a nearly-Japanese formalization with cleanup phases and known exceptional positions. It is not evidence of exact KGS equivalence or a selected Python engine.

Use the [source preset](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/game/rules.cpp#L273-L285) explicitly: `SIMPLE`, `TERRITORY`, `SEKI`, no suicide, no button, no handicap bonus, `friendlyPassOk=false`, komi 6.5. The pinned GTP documentation disagrees about `friendlyPassOk`; source settings govern this reference. Pin `BoardHistoryModes` explicitly, including `excludeTerritoryAdjacentToAtari=true` and `alwaysComputePassAliveUnderSuicideRules=false`; do not rely on legacy defaults. The former changes territory/no-tax behavior, not this seki-tax preset. ([Mode definitions](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/game/boardhistory.h#L11-L35))

[KGS's rules page](https://www.gokgs.com/help/ruelsets.html) describes territory plus captures, no territory in seki, no-result for persistent cycles, and a neutral third player's judgment when players disagree about life and death. Preserve those distinctions. Ordinary-play legality uses simple ko and forbids suicide; the adapter must verify these against the pinned reference and source fixtures. Do not substitute positional or situational superko.

KataGo's cleanup can inspect a **clone** as a diagnostic. Its extra decisions and phase-specific ko behavior must not enter actual candidate play, policy labels, move history or pass counts. Nor may a finite cleanup rollout certify arbitrary KGS endings: the reference's Japanese agreement depends on appropriate cleanup play and has explicit exceptions. Making candidates play that protocol would require a separately versioned scope decision.

Do not use GTP `final_score` or `final_status_list` as proof. The [pinned implementation](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/command/gtp.cpp#L1436-L1548) can round a predicted lead for an unfinished position and represent no-result as a draw. Its status output also does not supply a complete seki certificate. Reference estimates remain diagnostics with their own versions and costs.

## State, actions and replay

The shared engine owns the board, captures by color, side to play, simple-ko prohibition, actual pass streak, complete action/history record, recurrence records, rules/komi and lifecycle state. Persist enough state to reproduce every legal mask and adjudication decision. A model's 0/8/32/64-move window is only an input view; shortening it must never remove engine history. Clone and serialization preserve the full state without aliasing.

In `playing`, the learned policy has **362 actions**: the 361 intersections and explicit pass. A legal placement is on an empty intersection, removes adjacent opponent chains without liberties, then must leave its own chain with a liberty and satisfy simple ko. A pass changes the turn, clears the immediate-ko restriction and increments the pass streak; a placement resets that streak. An occupied point, suicide, immediate ko recapture, wrong player or action outside `playing` is rejected without mutation. Do not use tolerant SGF replay or assume that a library's legality call rejects actions after termination. ([Reference replay paths](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/game/boardhistory.cpp#L806-L972))

Resignation is a separate explicit control event, never a fabricated board/pass action. Initial raw A and B profiles have **no automatic resignation rule**. Human controls and independently named reference profiles may resign; any learned resignation heuristic is a separate profile and needs qualification. Resignation supplies a winner but no numerical territory margin.

Source ingestion preserves SGF bytes, archive metadata, hashes and conflicts. Require `GM[1]`, size 19, explicit Japanese/6.5 and the selected clock tuple; no missing field becomes a default. Exclude handicap, setup `AB/AW/AE`, noninitial `PL`, White-first games and unknown played variations from the initial corpus. Root `PL[B]` is redundant but acceptable; `HA` must be absent or zero. Never splice variations, repair a move or append cleanup to make a record replay. A genuinely illegal action or conflicting known player/date/rank provenance quarantines the game rather than retaining a misleading prefix.

Normalize explicit empty and legacy `tt` passes on boards of this size while retaining their raw encoding. SGF's display semantics allow more than this strict legality contract; syntax success is not replay success. `TB/TW` describe territory/area markings, not a general life/death certificate. ([SGF Go semantics](https://www.red-bean.com/sgf/go.html))

EOF leaves the recorded action sequence unchanged. A record ending without a trustworthy result may still contribute legally replayed policy actions under the masks below. A first pair of consecutive passes enters `pending_adjudication`; later ordinary SGF moves require an explicitly supported source resumption record and otherwise quarantine the game. None of the 1,000 audited main lines contained actions after a two-pass pair. That observation does not establish a general source guarantee.

## Endings and exact scoring

The lifecycle is `playing → pending_adjudication → resolved` or `unresolved`; resignation can resolve directly from `playing`. Administrative limits produce `truncated`, not a scored result. Keep source result, adjudication result, match accounting and learning validity as separate fields.

At two actual passes, freeze board, next player, captures, full history and source/candidate action count. Adjudication must identify this exact snapshot and provide complete, nonoverlapping chain and point classifications. Accept either an independently checkable life/death/territory certificate or an explicit, independently supplied neutral adjudication with provenance. The latter requires an authorized referee and a recorded budget; none has been recruited by this bead. An accepted certificate is reusable only for the identical state and contract version.

The initial automatic certifier may support a conservative subset: all remaining chains have unconditional-life evidence; no dead chain, seki, open ko, unfilled dame or unsettled region remains; and every empty region is proven territory of one living boundary. **A single-color flood-fill boundary or unconditional life of the surrounding chain alone is insufficient**: a large enclosed region may still support an invading group. Empty-board double pass is not certified. Additional dead/seki support needs independently checked evidence for whole-chain status, neutral/seki points and all empty regions. Mere engine agreement, finite search confidence, or `TB/TW` matching `RE` does not provide that evidence.

Remove certified dead chains only in a scoring view. Count each removed stone as an additional prisoner and classify its vacated point normally. Do not count living stones as territory; seki territory and neutral points score zero. With `T` certified territory, `C` prisoners already captured and `D` dead opposing stones removed, use exact integer half-points:

```text
white_minus_black_halfpoints = 2 * (T_white + C_black + D_black
                                      - T_black - C_white - D_white) + 13
```

Here `C_black` and `D_black` count Black stones captured/removed for White, and conversely for White stones. The sign selects the winner. No rounding of a model estimate is allowed. Since the result is odd, a scored draw is impossible at 6.5 komi in this contract. Preserve a source `RE[0]`, `Void`, unknown result or no-result as its own evidence; do not reinterpret it as a permitted draw. A conflicting independently certified score masks the recorded outcome and requires investigation.

Each automatic adjudication gets at most **100,000 proof/search nodes, 2 process-CPU seconds and 10 wall seconds**, whichever is reached first, with no GPU. Record implementation/version, evidence, limits, counts and failure reason. Diagnostic neural analysis requires a separately budgeted run and cannot override a failed certificate. Budget exhaustion, missing classifications, disagreement, unsupported Japanese exceptions or uncertain seki/death return `unresolved`; they never trigger automatic replay or extra candidate moves. Certifier coverage may be low and can block qualification. These limits authorize no expensive run by themselves.

### Repetition and administrative boundaries

Retain every actual board/turn state, but do not impose superko as a legality filter. [KGS's loop warning](https://www.gokgs.com/json/downstream.html#GAME_LOOP_WARNING) describes further repeats before no-result without fully specifying the state key. The pinned [KataGo cycle logic](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/game/boardhistory.cpp#L1175-L1180) ends at its third occurrence and has additional pass-history behavior. Exact equivalence is unverified.

Version 1 therefore uses an explicit **administrative recurrence guard**: if a placement recreates `(stone arrangement, next player)` already seen in the segment since the most recent actual pass, request truncation at the next candidate/learner decision boundary. Count the segment's initial state; compare boards after hash matches to exclude collisions. Passes start new segments, but the full history is never discarded. This guard does not declare a move illegal or the game a rule-level no-result. A verified source no-result remains an outcome without a win/loss target. Disable backend automatic phase/no-result conversions at this adapter boundary; reject a backend that cannot preserve the distinction.

Research matches have a nominal cap of **800 actual actions**, **60 wall seconds per agent response** and **3,600 wall seconds per game**. The action/recurrence cap is enforced after a complete candidate action plus opponent reply, at the next candidate-to-move state. At most one reply may complete that boundary after the action threshold, giving at most **801 actual actions**, including an opponent opening when the candidate is White. A genuine ending reached first is handled before the pending cap. Schedule enough shutdown/reply allowance within the experiment's hard resource cap; otherwise do not launch the next transition. Record these limits and timed versus untimed contexts in each match manifest; they do not enlarge E0's compute budgets.

For B, a nonterminal capped transition uses the saved **B-to-move** final observation, with `terminated=false`, `truncated=true`, bootstrap from that state and no recursion across reset. A resolved score/resignation uses `terminated=true`, zero bootstrap and fixed-B reward +1/−1. An unresolved two-pass ending has no valid payoff or playable continuation: return `learning_valid=false`, with reward and learning-boundary flags absent, retain the trajectory and fail the on-policy collection batch before an optimizer update. The same stop applies to invalid-action/state faults; do not train with an invented draw, silently drop just the failing game, or bootstrap from an opponent-turn/reset state. A response/infrastructure failure that prevents constructing the required final observation also invalidates the batch. No GAE/bootstrap calculation may consume an invalid transition. Rollout chunk boundaries remain nonterminal bootstrap boundaries under E6.

### Match accounting

Keep every scheduled game in `N = W + L + D + U`. `D=0` for this version. Certified score and explicit resignation resolve W/L. A demonstrably agent-caused illegal response, crash or exceeded response deadline is a forfeit loss in evaluation and a behavior failure; preserve its distinct reason. A server/scheduler fault, ambiguous responsibility, unfilled slot, cap, recurrence guard or unresolved adjudication stays U. Actual time-control expiry, where timed matches are authorized, is a known match loss but remains separate from an infrastructure deadline. No failure is replayed to obtain a favorable result.

Use E0's conservative all-game endpoints, group weights and uncertainty rules. A resolved disagreement still counts in the dispute numerator; an ordinary request for adjudication is not itself a dispute. Unresolved or failed B trajectories are not valid PPO terminal-reward examples merely because evaluation assigned a forfeit.

## Data and label masks

Canonical construction must publish new counts under this contract; do not rewrite the audit's frozen 241-record result. The audit's coarse exclusion flags are evidence, not the final per-head training masks.

| Field or condition | Policy |
| --- | --- |
| Human/ordinary-game eligibility and source use | Unknown remains unknown. `6yr.9` must establish a defensible policy before shared-model training or splits are accepted. |
| Historical ranks | Require both exact source ranks within 15k–1k/1d for the initial corpus. Exclude missing, provisional or conflicting labels; never substitute the 2026 survey rank. Deliberate input masking for the trained unknown-opponent-rank path is separately seeded and logged, without erasing the source labels. |
| Accounts | Preserve source handles and any account-generation evidence. No stable identity means no personalization. Use the default identity path; do not equate handle equality with a verified person or quietly relax disjoint-cohort rules. `6yr.9` supplies the grouping policy; `.5` implements and checks it. |
| Rules, setup, played line or replay | Unknown/incompatible settings, an ambiguous played line or illegal actions prevent policy/value training from that game. Diagnostics may still retain the original. |
| Missing `RE`, timeout or forfeit | Keep otherwise eligible observed policy actions; `value_valid=false`. The five archive-forfeit/absent-SGF-result cases remain outcome-masked. A disagreement between two known winners/margins is an integrity conflict and quarantines the game. |
| Known score or resignation | After source/identity/split/replay gates, recorded outcome may supervise value if source evidence agrees and no verified contradiction exists. This predicts the recorded result, not optimal play. It is not an endgame certificate. |
| EOF/unfinished source game | No new move or terminal event. Keep only supported policy actions; unknown outcome remains masked. |
| Position/persona/time features | Pre-move state only. Identity never enters value computation. Post-move clocks, target duration, final result and adjudication metadata are labels/diagnostics, not model inputs. |

For a valid recorded Black win, `y_black=+1`; for White, −1. The mover-view target is `y_black` for a Black-to-move observation and its negative for White. Search values use the player to move and flip per ordinary ply; B's environment uses B's fixed color throughout. No numerical payload is meaningful without its validity flag; masked fields must not enter losses or fitted normalizers.

### Timing version 1

Preserve nominal settings and raw `BL/WL/OB/OW` with presence and parsing masks. `BL/WL` are post-move times; the generic SGF overtime counters do not by themselves validate KGS Japanese-period semantics. ([SGF properties](https://www.red-bean.com/sgf/properties.html)) [KGS time controls](https://www.gokgs.com/help/timesystems.html) also permit adding time, while the [live clock schema](https://www.gokgs.com/json/downstream.html#GAME_STATE) distinguishes pause/running state and current-period time. An archive that lacks that event history cannot prove all elapsed durations.

**All duration targets and dynamic-clock inputs remain masked in version 1.** Static known time-control settings remain usable; untimed generated games use the trained missing-clock context. This is an explicit deferral of TIME and timing-adaptive search, not zero thinking time. The 50,330 compatible-subset main-time differences remain diagnostic candidates only.

A later timing-policy revision may admit `elapsed = previous_post_clock_same_color − current_post_clock` only with verified main-time state, consecutive usable values, nonnegative difference and evidence excluding intervening adjustments, resets, pause effects and overtime transitions. First-move extraction from `TM` needs the same evidence. Validate source serialization and precision on reference fixtures before admitting any label. Missing/invalid clocks break the chain; increases, overtime markers, transitions, interrupted records and unknown semantics stay masked. No duration is clipped or silently interpolated into validity. Fit duration transforms on valid training targets only.

## Qualification thresholds

These are frozen **project requirements**, not fitted guarantees about human play. The audit's 241 compatible games have observed-action median/p90/p99 **237/295/326** and maximum 346; its 102 scored records have **265/306/346**. They are a biased, unverified sample and are not certified termination lengths. The 600-action tail limit below leaves substantial room beyond these observations; it is not a claimed human percentile. No source profile currently meets E0's minimum cell-support requirements.

Apply the following to each advertised A profile and each B checkpoint proposed for qualification, on the full validation panel, each opponent-family group, and each stress context below. Use E0's fixed weights within a group. Each checkpoint in a three-seed promotion must pass. For game-level binary gates, use E0's one-sided exact stratum bounds, including unfilled/unjudgeable games as adverse outcomes; enumerate gates/seeds/groups before Bonferroni allocation. The 0.01 regression allowance still applies in addition to these absolute limits.

| Gate | Required bound or check |
| --- | --- |
| Rules/state/perspective integrity | Zero reproducible defects; all normative fixtures pass. No aggregate rate excuses an illegal mask or wrong outcome sign. |
| Agent runtime/response failure games | Upper bound ≤ **0.01**. A reproducible code defect still blocks under the integrity gate. |
| Cap/recurrence games | Upper bound ≤ **0.01**. |
| Unresolved games | Upper bound ≤ **0.01**. |
| Dispute games | Upper bound ≤ **0.01**, including disagreements later resolved. |
| Games with a premature or unjudgeable candidate pass | Upper bound ≤ **0.01**. |
| Length tail | Upper bound ≤ **0.05** for completion after **600 actual actions**; count every unresolved/censored game adversely, even if observed length is shorter. |
| Pass judgment coverage | At least **99%** of candidate passes judged, with at least **100 pass-bearing games per candidate color** in each declared panel/stress gate group. Counts are requirements on the frozen scheduled sample, never a reason for optional stopping. |

Use descriptive median/p90/p99 only for true resolved endings, and show all observed lengths separately with censoring. In stress continuations report prefix length and continuation length separately; total actual actions including the prefix determine the 600/800 limits. Do not impose a lower game-length limit that would reject a legitimate quick win or resignation.

The premature-pass judgment operates on the exact pre-pass state. Mark **premature** if a checked certificate identifies a legal non-pass continuation that improves the mover's achievable Japanese score by at least **1 point** over passing, or proves required unfinished play such as unresolved scoring-affecting ko/atari. Mark **acceptable** only with checked evidence that no scoring-affecting unfinished play remains and passing loses less than 1 point relative to every legal alternative. Score comparisons use minimax continuation under this ordinary-play/scoring contract, with both players maximizing their own score; certified bounds suffice only when they prove the required inequality. A favorable single continuation is not proof against all opponent responses, and an administrative cutoff does not supply a counterfactual score. A life/death or counterfactual proof uses the same frozen rules and adjudication authority; finite search estimates are not certificates. Anything else is **unjudgeable**, including exhausted judgment budgets. Apply the same 100,000-node/2-CPU-second/10-wall-second budget per judgment. This strict judge may have insufficient coverage; that is a qualification failure, not permission to waive the pass gate.

Report premature/all passes, unjudgeable/all passes and game-level failures separately. No passes makes the per-pass measures undefined and fails the opportunity requirement above. Counterfactual/referee moves never change the actual game or its raw-policy status. Certifier and pass-judge versions/fixture verdicts must be frozen before measuring candidate outcomes.

For A, also evaluate **32 independent temperature-1 samples per held-out position**, using E0's equal game-side/rank hierarchy. Require a lower confidence endpoint of **0.10** for mean pairwise disagreement and **2.0** for mean distinct legal actions. These are anti-collapse requirements, not estimates of human conditional entropy. On the same positions require an upper confidence endpoint of **0.10 total variation distance** between model and recorded-human marginal move-category frequencies, per supported rank/phase/time cell. Keep E0's account-dependence checks, at least 30 distinct mover accounts and 100 game sides per claimed cell, multiplicity and three-seed rules. Insufficient support defers that profile/cell; B has no rank-conditioned human-diversity claim.

Move categories are mutually exclusive in this priority order, computed by legal replay: **pass**; **capture** (removes an opponent stone); **save-atari** (an adjacent own chain with one liberty survives in a resulting chain with at least two); **give-atari** (an opponent chain with at least two liberties is reduced to one); **self-atari** (the placed stone's resulting chain has one liberty); **other**. These categories measure marginal behavior against recorded actions, not tactical correctness. Use the same classification for all sampled/model/human actions. Pairwise disagreement is `1 − Σ c_a(c_a−1)/(32×31)`; total variation is `0.5×Σ|p_model−p_human|` over these six categories.

The stress suite has six separate contexts: prefixes at **60, 180 and 300 actual actions** from (a) uniform legal-placement self-play and (b) capture-first versus uniform legal placement, balancing generator colors. Both generators pass only if no legal placement exists; capture-first samples uniformly among capturing placements, otherwise among all legal placements. Use independent seeds for every root/context, retain full history and retain early-ended/unavailable scheduled roots as missing. Do not repeatedly generate until only easy positions remain. Run candidate continuations in both colors against each of the two frozen weak profiles; opponent profiles and colors remain separate strata. Every certified rules fixture is a separate exact gate. Human, generated and match denominators are never pooled.

Before a confirmatory run, use the actual assertion count and strata to cost the necessary fixed sample. For example, with 30 assertions and six profile/context strata in one absolute-rate assertion, a marginal tail allowance of `0.05/(30×12)` needs **884 independent color-pair blocks per stratum** even with zero observed failures to bound a 1% rate. Nonzero failures can need more. This illustration grants no new GPU budget; if checks do not fit, retain an exploratory result and defer qualification under E0.

## Normative implementation checks and handoff

`kenggo-6yr.4` must select the Python-compatible engine/adapter and make the following cases executable against this contract and the pinned source. Reference tests are seeds for fixtures, not transferable expected scores under different rule settings. ([Pinned tests](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/tests/testrules.cpp))

| Case | Required observation |
| --- | --- |
| Capturing into an apparently surrounded point; single/multiple suicide | Capture is processed before testing own liberties; legal capture succeeds, actual suicide fails without mutation. |
| Immediate ko, intervening exchange, longer recurrence | Simple ko bans immediate recapture only; legal longer cycles trigger the administrative guard, never superko or an invented draw. |
| First/second pass, EOF, resumed source line | First pass remains playing; second freezes pending state; EOF adds nothing; unsupported later moves quarantine the source record. |
| Settled map; dead stones; seki/one-sided dame; large enclosed invasion space | Exact territory/prisoner arithmetic only with complete accepted evidence; unsupported/incomplete maps remain unresolved. A living boundary alone cannot certify its interior. |
| Empty board double pass; open ko; exotic Japanese positions | No certificate by convenience; unfinished or unsupported cases remain unresolved/unjudgeable. |
| KataGo diagnostic cleanup and GTP score/status | Clone phases/ko-unblocking never enter actual action counts. Estimated/rounded scores and no-result-as-draw never become certified results. |
| Conflicting/overlapping/partial-chain map; wrong snapshot | Reject the certificate without altering source state or recorded result. |
| Positive/negative half-point score; resignation; missing/timeout/forfeit result | Correct signed perspective, no rounded draw, no synthetic resignation move, correct per-head masks. |
| White B reset, B terminal move, A terminal reply, 800-action boundary | Preserve the opening and actual transition; terminal takes precedence; nonterminal truncation saves B-to-move state, never reset/opponent-turn bootstrap. |
| Infrastructure/agent failure; unresolved ending during PPO collection | Correct W/L/U reason, retained scheduled slot, invalid batch gets no optimizer update and no favorable resampling. |
| Clone/serialization and short model history | Full legality, pending state, captures and outcomes remain identical; no shared mutable state or post-terminal actions. |
| Missing/invalid/increasing clocks, overtime, pauses, added time | Raw values retained; version-1 dynamic inputs/duration targets remain masked. |
| Pass/capture/save-atari/give-atari/self-atari overlap | Priority yields one category per legal action; both source/model paths agree. |
| Premature/acceptable/unknown pass judgments and missing stress roots | Evidence and budgets determine verdicts; unknown/missing never reduces the failure denominator. |

`.5` owns canonical provenance, duplicate handling and cohort/split enforcement; `.6` owns pre-move encoding and masks; E2 implements the statistical gates and match runner. This bead fixes decisions without claiming those checks have run. The source/identity gate `.9`, shared-engine checks and readiness `.7` remain open. No training, GPU benchmark, human recruitment or additional corpus download occurred.
