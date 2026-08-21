# Manuscript reproduction summary

Status: **REPLAY_VERIFIED** (`evidence-replay`).

No MLIP, MD, AIMD, DFT, training, or structure generation was executed.

## Source documents

| Document | Included |
| --- | --- |
| Manuscirpt_0510zdl.docx | no; read-only external source |
| SI_0510zdl.docx | no; read-only external source |

## Evidence-driven routes

| Task | Selected model | Selected evidence metrics |
| --- | --- | --- |
| electrochemical-voltage | `chgnet` | voltage_mae=0.182777777777778 V |
| ionic-transport | `deepmd-se_atten_v2` | conductivity_mae=0.09599531 mS/cm |
| static-pes | `deepmd-dpa2` | energy_rmse=6.63 meV/atom, force_rmse=151.0 meV/angstrom |

The registry contains no `recommended_tasks` hints; each route is obtained from the registered metrics and the policy recorded in `project.yaml`.

## Recomputed timing evidence

| Prototype | DeePMD mean (h) | AIMD mean (h) | Exact AIMD/DeePMD ratio | Manuscript approximation |
| --- | ---: | ---: | ---: | ---: |
| I | 1.179583 | 136.350625 | 115.592194 | 116 |
| II | 1.214236 | 137.558472 | 113.288075 | 114 |
| III | 1.258750 | 217.864028 | 173.079665 | 173 |

## Large-supercell screening replay

Status: **REPLAY_VERIFIED**. Candidate/evaluated count: **247/247**; excluded for missing metrics: **0**.

Metric: `ionic_conductivity_300k_s_per_m` (S/m; direction `maximize`). Candidate and transport records: `external-screening-artifact/candidates.json` and `external-screening-artifact/transport.json`.

| Rank | Candidate | Composition (Mn/Fe/Ni/Cu/Zn) | Conductivity (S/m) |
| ---: | --- | --- | ---: |
| 1 | `Mn6_Fe3_Ni8_Cu4_Zn7` | 6/3/8/4/7 | 18.420176685825222 |
| 2 | `Mn2_Fe2_Ni9_Cu9_Zn6` | 2/2/9/9/6 | 17.626045819654934 |
| 3 | `Mn4_Fe8_Ni7_Cu6_Zn3` | 4/8/7/6/3 | 14.282118728392 |
| 4 | `Mn7_Fe5_Ni6_Cu8_Zn2` | 7/5/6/8/2 | 14.034032196953927 |
| 5 | `Mn5_Fe9_Ni8_Cu3_Zn3` | 5/9/8/3/3 | 12.28403894229092 |
| 6 | `Mn4_Fe9_Ni5_Cu7_Zn3` | 4/9/5/7/3 | 12.145340623018692 |
| 7 | `Mn7_Fe5_Ni8_Cu2_Zn6` | 7/5/8/2/6 | 12.103564653934505 |
| 8 | `Mn3_Fe4_Ni7_Cu5_Zn9` | 3/4/7/5/9 | 11.669317033317999 |
| 9 | `Mn8_Fe2_Ni6_Cu5_Zn7` | 8/2/6/5/7 | 11.368330421666451 |
| 10 | `Mn6_Fe7_Ni8_Cu4_Zn3` | 6/7/8/4/3 | 11.349884215572143 |

## Top-candidate validation status

`Mn6_Fe3_Ni8_Cu4_Zn7` maps exactly to historical folder ID `6_3_8_4_7` under the declared Mn/Fe/Ni/Cu/Zn token order.

Historical transport status: **EXACT_NUMERICAL_PARITY** (`READ_ONLY_HISTORICAL_POSTPROCESS_PARITY`). The exact match is limited to read-only replay of existing target.msd/stdout post-processing; it is **not** AIMD/DFT high-fidelity validation.

AIMD/DFT high-fidelity status: **EXTERNAL_VALIDATION_PENDING**; numerical parity: **NOT_TESTABLE_WITH_AVAILABLE_DATA**.

## Unseen/transfer claim representation

Evidence level: **CLAIM_LEVEL_DOCUMENT_EVIDENCE**; status: **EXTERNAL_VALIDATION_PENDING**; numerical parity: **NOT_TESTABLE_WITH_AVAILABLE_DATA**.

Recorded source: `Manuscirpt_0510zdl.docx`. The document reports 787 AIMD configurations for Li24M12(PS4)16 and three qualitative conductivity comparisons, but supplies no paired values needed for numerical parity.

Blocking evidence:

- No per-configuration reference and predicted energy values are available.
- No per-configuration reference and predicted force values are available.
- The three doped composition identities and their paired DeepMD/AIMD conductivity values are not available in the supplied evidence tables.
- No raw AIMD or MLIP trajectories/output artifacts for this unseen/transfer assessment are included.

## Claim boundary

This report reconstructs evidence, reductions, and routing decisions; it does not independently validate the underlying model predictions.
The 247-candidate screening and historical top-candidate post-processing are replay-verified. Top-candidate AIMD/DFT validation remains external and pending; unseen/transfer evidence remains claim-level and cannot support numerical parity with the available data.
