# kenggo

Two playable 19×19 Go agents: a rank-conditioned model of human moves and an RL learner trained from scratch against frozen versions of that model.

The goal is playable Go with adjustable difficulty, planned around one RTX 3090. Move prediction, playing strength, and human-like behavior are evaluated separately.

The [Python research workspace](docs/workspace.md) provides a pinned CPU setup, an installation smoke check and artifact locations. The [E1 source audit](docs/e1_source_audit.md) adds bounded KGS acquisition and offline SGF inspection. The [approved admission policy](reports/e1_kgs_admission_2026-11-11.md) admits 241 screened, presumed-human games with owner-attested research-use permission. The [canonical builder](reports/e1_canonical_2026-11-11.md) preserves source records and enforces whole-game splits; the [example encoder](reports/e1_examples_2026-11-11.md) adds masked arrays and shared pre-move inputs. Their current artifacts still use the earlier policy and require regeneration before training. The [shared rules adapter](docs/rules_engine.md) validates ordinary play against pinned KataGo; automatic ending coverage remains a blocker. The [E1 readiness and manual-validation report](reports/e1_readiness_2026-11-11.md) records the earlier gate inspection and manual checks.

See the [project plan](plan/go_project_plan_v0.5.md) for scope and experiments, and [AGENTS.md](AGENTS.md) for the Beads, pstack, and review workflow.
