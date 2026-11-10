# kenggo

Two playable 19×19 Go agents: a rank-conditioned model of human moves and an RL learner trained from scratch against frozen versions of that model.

The goal is playable Go with adjustable difficulty, planned around one RTX 3090. Move prediction, playing strength, and human-like behavior are evaluated separately.

The [Python research workspace](docs/workspace.md) provides a pinned CPU setup, an installation smoke check and artifact locations. The [E1 source audit](docs/e1_source_audit.md) adds bounded KGS acquisition and offline SGF inspection. The [eligibility policy](reports/e1_kgs_eligibility_2026-11-10.md) supplies conservative handle groups but admits no training games yet. The [shared rules adapter](docs/rules_engine.md) validates ordinary play against pinned KataGo; automatic ending coverage remains a blocker. Training-data construction and agent training are still planned work.

See the [project plan](plan/go_project_plan_v0.5.md) for scope and experiments, and [AGENTS.md](AGENTS.md) for the Beads, pstack, and review workflow.
