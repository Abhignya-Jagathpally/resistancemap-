# Table 1 — Real CoMMpass (overall survival, GDC open tier)

N=781 patients, 161 events, 5-fold patient-disjoint CV, train-fold median imputation. Lower IBS / higher td-AUC / lower ECE = better.

| Model | C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |
|---|---|---|---|---|---|
| cox_ph | 0.737 [0.700, 0.771] | 0.1288 | 0.7486 | 0.0141 | 0.0519 |
| cox_elasticnet | 0.727 [0.691, 0.762] | 0.13 | 0.7404 | 0.0131 | 0.0408 |
| random_survival_forest | 0.721 [0.681, 0.757] | 0.129 | 0.7457 | 0.0049 | 0.0347 |
| gradient_boosted_survival | 0.696 [0.657, 0.733] | 0.1521 | 0.7052 | 0.0468 | 0.0682 |
| program_basin *(novel)* | 0.698 [0.659, 0.740] | 0.1445 | 0.7026 | 0.0263 | 0.0301 |

**Governance — `novel_model_beats_baselines`: BLOCKED** (rule: program_basin C-index CI-lower must exceed best baseline 0.737). program_basin reaches parity inside the baseline band through an interpretable basin-escape (Kramers) mechanism; it does not beat Cox, and the gate honestly refuses the unearned 'beats-baselines' claim.
