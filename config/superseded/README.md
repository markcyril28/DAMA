# Superseded training configs

Configs in this directory are **retired**: they are preserved because the
merged policy-distillation audit
(`.docs/policy-distillation-merged-audit-2026-08-19.md`) cites their exact
contents, but no launcher selects them and the GUI cannot offer them -- its
preset list globs `config/training_config*.yaml`, which does not recurse.

| File | Why retired |
| --- | --- |
| `training_config_policy_distillation.yaml` | Resumes step 134,000 and writes every mutable output into `policy_distillation_recovery_wd1e4`, the preserved read-only namespace that supplies the c174k anchor, lineage base, and ledger seed. |
| `training_config_server_policy_distillation.yaml` | Same outputs and anchor; the server path now runs the c174k config with `--profile server`, which `train_server.sh` already selects. |

`src/tests/test_recovery_output_isolation.py` still loads both from here, and
`test_no_selectable_config_writes_into_a_preserved_namespace` asserts that
nothing left in `config/` directs mutable outputs into a preserved namespace.
Move a file back only after repointing every output path in it.
