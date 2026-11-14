# kenggo

Two playable 19×19 Go agents: a rank-conditioned model of human moves and an RL learner trained from scratch against frozen versions of that model.

The goal is playable Go with adjustable difficulty, planned around one RTX 3090. Move prediction, playing strength, and human-like behavior are evaluated separately.

The [Python research workspace](docs/workspace.md) provides a pinned CPU setup and artifact locations. The [approved admission policy](reports/e1_kgs_admission_2026-11-11.md) screened 241 presumed-human source games with owner-attested research-use permission. After the [owner's two-game exclusion](reports/e1_manual_review_2026-11-13.md), the [populated E1 data](docs/e1_population.md) provides 106 training games, separate development/confirmation splits, 70 sealed final games, masked arrays and a fitted train-only normalizer. The [shared rules adapter](docs/rules_engine.md) validates ordinary play against pinned KataGo and supports separately tagged empirical ending judgments. The [completion evidence and manual checks](reports/e1_completion_2026-11-11.md) record the current gate results and limitations. This supports exploratory baseline work; no trained or qualified opponent profile exists.

See the [project plan](plan/go_project_plan_v0.5.md) for scope and experiments, and [AGENTS.md](AGENTS.md) for the Beads, pstack, and review workflow.

The [experiment-tracking guide](docs/experiments.md) covers W&B runs, pinned
checkpoints, exact continuation checks and resource accounting.
