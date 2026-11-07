# Human-like Go

Project plan | v0.5 | Updated 6 November 2026 | Task planning baseline

## Purpose and scope

Build two playable agents on 19 x 19 Go:

- **A: Human model.** Learn human moves from recorded games, conditioned on rank, with history, individual players and thinking time added through controlled experiments.
- **B: RL learner.** Learn from random weights by playing frozen versions of A. Measure whether learning transfers to independent opponents.

The goal is playable Go with adjustable difficulty. A offers selectable rank-conditioned opponent profiles; establish the supported range through data coverage and calibration. Measure B's achieved strength and transfer independently. Plan around one RTX 3090. Move prediction, strength and human-like behavior are separate outcomes. Pilot budgets do not promise a particular difficulty range or playing strength.

KataGo already provides a rank-conditioned human Go model [1, 2]. The research questions here are which added features improve a smaller model, how search changes its behavior, and whether it provides a useful training opponent. Allie, regularized search and Maia4All supply precedents for these extensions [4, 5, 6, 7]. Chess results are motivation for Go experiments, not evidence that the same gains will occur.

The B experiments test whether training against A produces learning that transfers. Claims that A is a better training opponent than self-play or another opponent family require a separately budgeted control under matched resource limits.

Unless linked to prior work, settings and thresholds below are **project choices**. An epic is complete when its evidence and decision are recorded. An extension can finish with a decision to keep, defer or drop it. Report research completion and playable-release readiness separately under E7.

## Technology stack

Use the following stack for implementation:

| Area | Selected technology |
| --- | --- |
| Learning and research | Python + PyTorch for A and B, training, evaluation and search orchestration |
| Browser | TypeScript + React + Vite, with an SVG Go board |
| API | FastAPI with a separate inference worker; the server owns legality and turn state |
| Data | Preserve original SGFs; use Parquet for metadata and split manifests, and sharded arrays for training examples |
| Experiments | Weights & Biases (W&B) for experiment tracking, alongside versioned configurations and checkpoint manifests |
| Rules engine | A Python-compatible Go engine shared across replay, search, RL and serving; select and validate the library during E1 |

The rules engine may use native internals behind its Python interface. It must pass E1's legality, full-history and endgame checks. Difficulty settings map to evaluated, versioned opponent profiles as specified in E5.

Track experiment configurations, metrics and compute costs in W&B. Link each run to its code revision, data/split versions and checkpoint manifests, and include W&B run references in experiment reports. Keep the configurations and manifests needed to reproduce a run alongside its saved outputs. Configure W&B authentication and select the destination workspace/project when implementing experiment tracking; ask the owner for login help if needed.

## Epic map

| Epic | Outcome | Depends on |
| --- | --- | --- |
| E0: Scope and evaluation | Target population, evaluation protocol, success measures and pilot budgets | None |
| E1: Data and rules | Usable corpus, fixed splits and one game contract | E0 |
| E2: Human baselines | Rank-conditioned policy, external baseline report and reusable match runner | E1 |
| E3: History and personas | Evidence for useful history and individual variation | E2 |
| E4: Timing and search | Fair comparison of direct play and search allocation | E2; valid timing from E1 for timing-based experiments |
| E5: Qualify and freeze A | E5a: frozen opponent pool; E5b: browser play | E2, including the match runner; retained E3/E4 features for later profiles |
| E6: Train B from scratch | Valid PPO learner and measured learning cost | E5a and E2's match runner |
| E7: Transfer and review | Independent results, bounded claims and handoff | E6; recorded decisions on A extensions |

E3 and E4 are conditional experiments that can run in parallel. E2 supplies the match runner before E5 qualification or E6 training. An E2 baseline qualified and frozen through E5a can support the E6 pilot while extensions proceed. E5b browser delivery is a separate milestone required for release; E7 adds B and runs the reserved final evaluation. Qualify and freeze each later profile after combining its retained features. Deferred extensions do not block RL or research completion.

<!-- pagebreak -->

## E0. Fix scope and evaluation

**Goal:** Decide what the project will measure before committing to large runs.

- Choose a candidate source, proposed difficulty range, corresponding rank coverage, era and time-control family. KGS exposes rank/game/clock fields [18]; confirm access and usable coverage in E1.
- Start with even games and one source-compatible ruleset. Confirm komi, ko rules and the endgame protocol in E1. Any 9 x 9 pilot is a separate debugging experiment with separate strength claims.
- Define three scorecards: move prediction, playing strength and complete-game behavior. Use move negative log-likelihood (NLL) as the main imitation measure; use independent match scores for strength. Confirm access to humans at the ranks proposed for the difficulty settings. Size the human match budget for E7's calibration criterion at each claimed rank, accounting for repeated opponents. If access or the required budget is unavailable, keep human-rank claims provisional and report engine-relative strength.
- Define the NLL aggregation and weights: pooled moves, equal-weight games, accounts or rank groups. Keep rank/phase/time-control breakdowns and specify how unsupported groups are reported. Before each feature comparison, freeze a common eligible set of positions or games for paired evaluation; report its coverage and exclusions alongside gains.
- Define the training, independent validation and reserved final-test opponent panels and the color-balanced match protocol. Specify score aggregation, uncertainty, repeated-opponent handling and the treatment of capped or unresolved games; finalize rules-dependent details in E1. E2 implements this protocol for qualification and training decisions. Keep final-test results unopened until E7 selection is complete.
- Include weak independent opponents in B's validation panel so score comparisons can detect early progress. Define a setup check for floor effects, where the initial checkpoints and weak reference policies all score zero, before freezing the panel for E6's training comparison.
- Define measurable behavior criteria, including premature-pass and unresolved-game rates, game-length distributions and within-rank move diversity. Apply rank-conditioned diversity criteria to A and complete-game reliability criteria to both agents. Record measurement methods, populations and denominators in E0; fix acceptance thresholds after the E1 audit and before model qualification.
- Before each comparison, record its primary metric, minimum useful gain, allowed regressions, compute budget and selection rule. Define B's full-board learning success by improvement over its initial checkpoint on the independent validation panel, subject to those gain and regression limits.
- Start with a 1,000-game data audit, a short throughput benchmark, a 24 GPU-hour SL pilot and a 48 GPU-hour RL pilot. Each pilot cap covers training, tuning, reference inference, evaluation and failed runs across all its comparison arms. Record audit/benchmark costs separately. Set a separate three-training-seed confirmation budget from measured pilot costs before promoting a comparison; single-run findings remain exploratory. Extend budgets only after measuring cost and progress.

**Done when:** Record the proposed difficulty range, support limits, evaluation panels/protocol, metric weighting, behavior definitions, promotion criteria and budgets. Resolve data-dependent choices and acceptance thresholds in E1.

The [initial scope decision](e0_scope_decision.md) records the candidate population, source evidence, support limits and E1 audit fallbacks.

The [evaluation protocol](e0_evaluation_protocol.md) defines scorecards, opponent panels, comparison rules, uncertainty and the human-calibration budget.

## E1. Establish the data and game contract

**Goal:** Make recorded games, search, RL and the interface agree on legal play and outcomes.

- **Audit before scaling.** Verify reuse terms, rank coverage, stable identities, timing and games per account. Record usable counts and exclusions. Initially exclude bots, reviews, teaching games, duplicates and corrupt records.
- **Retain provenance.** Store source/date, accounts, historical ranks, rules/komi, moves, result/reason and clocks. Select played main lines; handle setup stones, handicap, missing rules and unfinished games. SGF syntax alone does not establish legal play [11].
- **Audit passes and endings.** Count explicit pass actions and complete endings by rank and result type. Distinguish scored endings, resignation and incomplete records; reaching the end of an SGF supplies neither a pass action nor proof of termination. Never fabricate pass labels from file endings. Inconsistent pass recording can impair learned endings [1]. If coverage is insufficient, acquire suitable records or narrow/defer affected raw-policy profiles before scaling. Any completion heuristic is a separately versioned and evaluated profile and does not qualify as raw policy; E6 waits for a qualified raw pool through E5a.
- **Prevent leakage.** Deduplicate before chronological whole-game splits. Reserve disjoint unseen-account validation/test cohorts. Exclude test-account games from shared-model training/tuning, validation-account games from shared-model training, and games bridging reserved cohorts. For held-out accounts, adapt only the embedding on earlier games; evaluate on untouched later games [7]. Freeze shared weights and the adaptation procedure before testing; choose settings using validation accounts. Fit normalizers on training data only. Target-move time, post-move clocks and results are labels only.
- **Validate timing.** SGF BL/WL encode time left after a move [11]. Derive elapsed time using the actual increment/overtime rules. Keep validity masks; ambiguous or missing durations are not zero. Missing identities disable personalization for those accounts. Unusable timing defers duration modeling and adaptive search.
- **Use one rules engine.** Pin rules and engine versions for replay, search, RL and the interface. Retain full legality history independently of the model's context window. Verify capture, suicide, ko/superko, pass, scoring and resignation behavior against a fixed reference [12].
- **Resolve endgames.** Specify dead-stone handling and any cleanup/adjudication. Raw area scoring of an uncleared human end position is insufficient. Preserve recorded results and distinguish true termination from an administrative move cap [12, 15]. Finalize the match protocol's handling of unresolved games and the behavior thresholds, including how premature passes are identified, before model qualification.

**Done when:** Data supports each promised capability; pass/end coverage and resulting profile limits are recorded; splits are fixed; replay/scoring checks pass; endgame, missing-data and behavior acceptance policies are explicit. Revise E0 if coverage is insufficient.

<!-- pagebreak -->

## E2. Build and measure human baselines

**Goal:** Establish a reproducible rank-conditioned policy before adding complexity.

- Start with a compact board CNN and metadata input. An 8-block, 128-channel design is an initial candidate. Compare the same model with and without rank conditioning. Alternative Maia-2/Chessformer-style architectures are follow-on experiments [8, 9].
- Use the mover's board view, legal actions, color, komi, ranks and pre-move clocks. Include rule-relevant state, such as prisoners and pass/cleanup status. Train missing-clock and unknown-opponent-rank fallbacks; searched play against unranked opponents also requires a trained unknown-mover-rank path. Output 361 board moves plus pass and a value in [-1, 1]. Learned resignation is deferred.
- Train policy cross-entropy on human actions and a masked value loss on the recorded result from the mover's view. Initially exclude unknown, timeout and forfeit outcomes from the value loss. The value predicts eventual human-game outcomes, not human beliefs or optimal play [3]. No search-generated action labels enter SL.
- Include pinned **KataGo Human SL** as an external reference. Use raw `humanPolicy` for imitation scores and sample it at temperature 1 for raw play. Record engine/checkpoint/profile and history availability. Report any pass-corrected version as a separate playing baseline, including its extra compute [1]. Its profiles include KGS, fixed-era/time-control defaults [2]. Report context mismatches and unknown training overlap; prefer post-release evaluation games.
- Report NLL, move matching and value calibration by rank, game phase and time control. Compare raw sampling separately from argmax or search. Record training/inference cost and complete-game failures.
- Build the reusable match runner from E0's protocol and E1's game contract. Record profiles/checkpoints, seeds, colors, results/reasons, caps/unresolved games, compute and match SGFs. Check terminal, truncated and unresolved-game reporting with fixtures. Use the validation panel for E5 qualification and E6 decisions; reserve the final-test panel for E7.

**Done when:** The baseline runs reproducibly, finishes legal games and has a held-out validation comparison against the unconditioned model and KataGo; the match runner passes its protocol fixtures. Investigate ineffective rank conditioning before adding extensions. Matching a larger external model is not a prerequisite.

## E3. Test history and individual players

**Goal:** Improve prediction and game-level consistency only where the data supports it.

- **History:** compare 0, 8, 32 and 64 prior individual moves by both players, including passes. Start with a simple recent-move representation; add a small history Transformer only if it justifies its cost. Rules retain full history regardless of this window.
- **Personalization:** first train population behavior. Then train the identity-conditioning pathway jointly with varied training-account embeddings before freezing shared weights and adapting new embeddings. Maia4All motivates this preparation stage [7].
- Evaluate feature comparisons on E0's common eligible positions and report coverage. For added shared-model training stages, continue a population-model control on the same training-account data with comparable additional compute; report resource differences. Held-out-account adaptation still changes only embeddings under E1's split rules.
- Train a default identity. Initially personalize the policy; keep identity out of the value computation, including shared features. Hold each identity fixed per game and assign it consistently by color. Restrict personas to supported rank/context combinations.
- Compare correct, default and same-rank shuffled identities on later games for known personas and newly adapted accounts. For held-out accounts, report default-identity and adapted results separately. Report benefit versus games available per account. History may explain behavior without an explicit identity. Do not label latent differences as tilt or psychological traits without separate evidence.

**Done when:** A comparison of each feature's benefit and cost supports keeping or dropping history and personas. Promote personalization only if it improves individual prediction beyond rank/history controls. Otherwise continue with the population model.

<!-- pagebreak -->

## E4. Test thinking time and search

**Goal:** Determine whether search improves useful play while preserving the desired human behavior.

### Timing model

Add a masked elapsed-time loss to the SL model using only pre-move information. Allie provides the precedent for predicting time and using it to allocate search [4]. Compare against a metadata-only predictor with the same rank, move number, time-control, clock and missingness inputs. Report error in seconds, including long-think and overtime cases. Visible waiting time is a separate interface choice.

### Search contract

- Use policy/value PUCT [5, 6], seeking favorable moves for each side at its own nodes. This is adversarial search, not an expectation over sampled human replies. Values use the player-to-move view; flip signs per ply. Terminal results are exact; imagined-position outcome estimates require validation [3].
- Preserve rules history and append imagined moves. Hide unobserved future clock/time fields together; hide the same groups during training. Keep ranks/identities consistent by color. Before using searched profiles against unranked opponents, validate trained unknown-rank inputs in both player roles. Start each root with fresh counts and no root noise.
- Evaluate the root first. A budget counts additional simulations. Initialize unvisited action values to the node's value prediction. Record actual network calls, latency, searched prior mass and how often an unvisited root action is selected.
- Retain the custom root rule below over **all legal moves**, with Q in [-1, 1]. Sample the result at temperature 1. Keep beta fixed across allocation comparisons; beta = 0 returns the human policy.

```text
q(a) = p(a)*exp(beta*Q(a)) / sum_b p(b)*exp(beta*Q(b))
maximize_q: E_q[Q] - KL(q || p)/beta    (beta > 0)
```

This is the project's explicit regularizer. The PUCT interpretation in Jacob/Grill uses the opposite KL direction, KL(p || q) [5, 6]. Changing visits at fixed PUCT settings also changes effective regularization; keep the search setup explicit and distinguish it from the final root rule.

### Outcome-value validation

Before promoting any searched profile, freeze a small validation set with rule-verified tactical/endgame fixtures and positions generated by raw A, weak/random opponents and candidate search. Include unknown-rank and masked-clock inputs used in deployment. For generated positions, record frozen continuation policies for both colors, ranks/contexts, rollout counts and outcome uncertainty. Their empirical returns measure outcomes under those policies; strong-engine evaluations are separate diagnostics and are not human-outcome labels.

Before measuring the candidate, specify the target for each subset, metrics and numerical acceptance thresholds. Check legality and exact terminal-result signs on fixtures; measure value calibration/error against continuation returns on generated positions, with uncertainty accounted for. Held-out human-game calibration alone does not satisfy this gate. If the gate fails, defer fixed and adaptive searched profiles and retain only qualified raw play. Passing it permits controlled search comparisons; promotion still requires their imitation, strength and compute evidence.

### Controlled comparisons

Compare direct sampling, fixed search, time-adaptive search, metadata-only allocation and shuffled allocation using one frozen network/root rule. On common positions, shuffle allocated budgets within rank/clock-context/phase groups, preserving the budget multiset.

Start fixed search at 32 additional simulations. Initially allocate simulations in proportion to predicted seconds, following Allie [4]. Use the same rule for the full and metadata-only predictors; calibrate each predictor's scale on validation so its mean is 32 after rounding/clipping, with a cap of 128. Match PUCT settings and freeze validation-selected beta. Report realized network calls and latency. Claim an allocation benefit only at comparable measured compute; otherwise report the performance and cost tradeoff.

**Done when:** Record the outcome-value gate results and the keep/defer decision. For searched profiles, correctness and outcome-value checks pass and results separate imitation, strength and compute. Promote adaptive allocation only if it beats fixed, metadata-only and shuffled controls within E0 limits; otherwise keep qualified raw or fixed play. SE-MCTS remains a later alternative; its Go study excluded players below 5 kyu [10].

<!-- pagebreak -->

## E5. Qualify and freeze opponent A

**Goal:** Produce a stable training opponent and a usable playing interface.

### E5a. Qualify and freeze the opponent pool

- Freeze the selected checkpoint, inference behavior, supported rank/persona, sampling, clock fallback, search and endgame settings as versioned profiles. For the initially unranked B, use a fixed, trained unknown-opponent-rank path in the project's A model.
- Stress-test A on complete games and positions generated by weak/random opponents. Use E2's match runner and E0/E1's behavior criteria to assess premature passes, scoring disputes, repetition, game lengths, unresolved games and truncations. Held-out human move prediction alone does not qualify a simulator for RL [16].
- Use raw policy sampling for the first RL opponent pool. Treat searched opponents as a separate, costed experiment. Hold the selected opponent profile fixed per episode.

**Done when:** A meets the complete-game acceptance thresholds under the selected contract and the frozen pool is reproducible. Report each profile's measured strength or label it uncalibrated. E6 may start after this milestone; training readiness and a human-rank claim are separate gates.

### E5b. Deliver browser play

- Provide a browser board with server-owned turn state, pass/resign, a difficulty selector mapped to supported versioned opponent profiles, profile/checkpoint selection, replay and SGF export. Verify that the offered difficulty settings produce distinct playing strengths. Reject stale moves and save the settings needed to reproduce each game.

**Done when:** Browser play against A, profile selection, stale-move rejection and replay/SGF export work, and displayed strength labels match the qualification evidence. This milestone is required for release; E7 adds B and assesses the full release gate.

## E6. Train B from random weights

**Goal:** Establish valid policy-gradient learning and its cost before scaling.

- Initialize a separate policy/value network from random weights. Reuse architecture or rules code, but copy no SL weights. Initially omit MCTS, imitation loss, persona/rank inputs, timing prediction and reward shaping. Play untimed games with equally sampled learner colors.
- One transition is B's move plus A's reply, unless play ends first. When B is White, reset includes A's opening move. Decision observations are learner-to-move; rewards/values use B's fixed view. Use gamma = 1, zero intermediate rewards and terminal +1/-1 for win/loss, or 0 for a permitted draw.
- Use PPO with GAE [13, 14]. Start lambda at 0.95 and compare complete-game lambda = 1 returns in the pilot. Keep sampling/log-probability legal masks consistent, use fresh on-policy rollouts and track KL, entropy and update stability.
- True termination uses zero bootstrap and stops advantage recursion. At rollout boundaries or administrative caps, bootstrap from the saved final observation; stop recursion across resets and never use the reset state. Apply caps at learner decision boundaries [15].
- Compare a single opponent, a fixed pool and that pool with a curriculum at matched learner-transition budgets. Set advancement/retention rules before running. Keep a fixed validation panel and report GPU-hours. If almost always losing, check scoring before reducing difficulty. Any random-move mixture is an explicit unranked training aid.
- Save each seed's initial random checkpoint. During E6 setup, use E2's runner with the initial checkpoints and weak reference policies to check E0's score sensitivity requirement; revise a panel with uniform losses and freeze it before the training comparison. Compare trained checkpoints against their initial baselines using this same independent validation panel and match protocol. Report score improvement and uncertainty, applying E0's minimum useful gain and behavior regression limits to the full-board learning decision.
- Use the 48 GPU-hour pilot to measure complete games, terminal feedback, transitions, updates, memory and evaluation cost. Set the next budget from measured progress and projected sample cost.

**Done when:** Return/sign and boundary checks pass, and stored and recomputed action probabilities agree before an update; a small task shows learning. Base the full-board keep/change/stop decision on terminal feedback, independent validation and projected cost. Training wins alone do not justify scaling [16].

<!-- pagebreak -->

## E7. Evaluate transfer and hand off

**Goal:** Make claims that are supported by independent evidence.

- Select models/settings on validation only. Freeze final deployment profiles before opening the final test set. Repeat promoted comparisons across three training seeds; single-run findings remain exploratory. Report uncertainty and all runs [17].
- For imitation, use E0's fixed aggregation/weights and common evaluation sets; report NLL and move matching by rank/phase/time control, coverage and exclusions. Assess uncertainty at the game/account level. For behavior, report E0/E1's premature-pass and unresolved-game rates, game-length distributions and within-rank move diversity against their acceptance criteria. Add whole-game inspection of mistakes, consistency and time usage, plus a small blinded human pilot where feasible.
- For strength, use color-balanced matches against independent engine/model families. Include held-out humans when access and the E0 match budget permit. Separate those results from new profiles of A's existing model family. Report score = (wins + 0.5*draws)/games, 95% intervals and repeated-opponent dependence. Fix how capped or unresolved evaluation games are resolved and reported. Report their count separately; do not silently exclude them or count administrative caps as draws.
- For each difficulty setting labeled with a human rank, seek a 95% score interval inside 40-60% against the corresponding server cohort and check adjacent ranks under matching conditions. This is a project calibration criterion. Human-rank claims for B also need matching human evidence; report its achieved strength. Otherwise report engine-relative strength only.
- Use E2's match runner and the fixed independent **validation** panel for B's transfer curves, including its initial-checkpoint baseline, by learner transitions and GPU-hours. Run the reserved independent **final-test** panel only after selection is complete. Inspect winning strategies: exploiting one Go opponent can produce high wins without broad strength [16].

### Planned comparisons

| ID | Comparison | Main decision |
| --- | --- | --- |
| BASE | Unconditioned / rank-conditioned / KataGo Human SL | Does rank help, and how does the baseline compare? |
| HIST | History length and simple encoding / Transformer | Is the prediction gain worth the cost? |
| PERSONA | Correct / default / same-rank shuffled identity | Is individual information useful? |
| TIME | Full / metadata-only duration prediction | Does position/history add timing information? |
| SEARCH | None / fixed / adaptive / metadata / shuffled | Does allocation help beyond budget and context? |
| RL | Single / pool / pool with curriculum | Does improvement transfer at an acceptable cost? |

Run comparisons in small stages; E3/E4 extensions depend on data, pilot results and budget. Deferred extensions do not block completion. Re-run the winning additions together to check interactions. An inconclusive or negative result is a valid outcome; distinguish it from an implementation failure.

### Handoff for task planning

Each epic supplies a decision note, acceptance evidence and dependencies. Later tasks name one bounded output, its epic/reference, inputs and a completion check. Agents choose routine structure, libraries, tuning and batching within the selected technology stack.

The final package contains code and environment setup, training/evaluation/play commands, pinned data/model/rules versions, split manifests, configs, checkpoints, metrics, W&B run references, match SGFs and a short report of retained, deferred and rejected ideas. Integrate the selected B checkpoint into the browser with its measured results and profile metadata; keep its controls consistent with E5b's legality and replay contract.

**Research complete when:** Both agents and experiment decisions are reviewable, results reproduce, transfer is reported and claims state their limits. A negative full-board learning result can complete the research. Record unmet learning or difficulty goals and the next justified experiment.

**Playable release ready when:** Browser play against both A and B works; both meet the complete-game acceptance thresholds; at least two supported A difficulty settings have measurably distinct strengths under E0's comparison rule; and B meets the declared full-board learning criterion relative to its initial checkpoint. Apply the human-rank calibration gate separately to any rank labels. If a release condition fails, deliver the research evidence and mark release readiness unmet.

<!-- pagebreak -->

## References

Primary papers, specifications and official implementation notes. Links support methods and factual limits; project settings are proposals.

1. **KataGo.** [Human SL Analysis Guide](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md). Raw human policy, profiles and inference settings.
2. **KataGo.** [Human profile implementation](https://github.com/lightvector/KataGo/blob/cbeab3b0/cpp/neuralnet/sgfmetadata.cpp). Rank, source, era and time-control defaults.
3. **KataGo maintainer.** [Human SL policy, value and search](https://github.com/lightvector/KataGo/issues/1030). Outcome prediction versus human judgments.
4. **Zhang et al. (2025).** [Human-Aligned Chess With a Bit of Search](https://proceedings.iclr.cc/paper_files/paper/2025/file/0ef1afa0daa888d695dcd5e9513bafa3-Paper-Conference.pdf). ICLR. Allie and time-adaptive search.
5. **Jacob et al. (2022).** [Modeling Strong and Human-Like Gameplay with KL-Regularized Search](https://proceedings.mlr.press/v162/jacob22a.html). ICML. Imitation and search tradeoffs.
6. **Grill et al. (2020).** [Monte-Carlo Tree Search as Regularized Policy Optimization](https://proceedings.mlr.press/v119/grill20a.html). ICML. PUCT, KL direction and visit dependence.
7. **Tang et al. (2025).** [Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess](https://www.cs.toronto.edu/~ashton/pubs/maia4all-preprint2025.pdf). Maia4All preprint. Preparing and adapting player embeddings.
8. **Tang et al. (2024).** [Maia-2: A Unified Model for Human-AI Alignment in Chess](https://proceedings.neurips.cc/paper_files/paper/2024/file/250190819ff1dda47cd23cecc0c5a69b-Paper-Conference.pdf). NeurIPS. Shared skill conditioning.
9. **Monroe et al. (2026).** [Chessformer: A Unified Architecture for Chess Modeling](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d167db04a90885ad5208fe8b273668b-Abstract-Conference.html). ICLR. Architecture behind Maia-3.
10. **Chen, Shih and Wu (2025).** [Strength Estimation and Human-Like Strength Adjustment in Games](https://proceedings.iclr.cc/paper_files/paper/2025/file/17a1a1439421f1837e10cd612bf92861-Paper-Conference.pdf). ICLR. SE-MCTS and evaluated rank range.
11. **SGF FF[4].** [Properties](https://www.red-bean.com/sgf/properties.html) and [Go-specific semantics](https://www.red-bean.com/sgf/go.html). Moves, setup, results and timing.
12. **KataGo.** [Supported Go Rules, Version 3](https://lightvector.github.io/KataGo/rules.html). Precise legality, scoring and endgame reference.
13. **Schulman et al. (2017).** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). PPO policy updates.
14. **Schulman et al. (2016).** [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR. Advantage estimation.
15. **Pardo et al. (2018).** [Time Limits in Reinforcement Learning](https://proceedings.mlr.press/v80/pardo18a.html). ICML. Terminal versus truncated episodes.
16. **Wang et al. (2023).** [Adversarial Policies Beat Superhuman Go AIs](https://proceedings.mlr.press/v202/wang23g.html). ICML. Opponent exploitation and transfer limits.
17. **Agarwal et al. (2021).** [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264). NeurIPS. Uncertainty across limited training runs.
18. **KGS.** [JSON Data Types](https://www.gokgs.com/json/dataTypes.html). Player, rank, game and timing fields; availability still requires an audit.
