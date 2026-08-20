# Manuscript reproduction acceptance summary

Status: **REPLAY_VERIFIED**.

The automated acceptance test reconstructs the manuscript's task-specific
model choices from compact recorded evidence:

| Task | Selected model | Recomputed criterion |
| --- | --- | --- |
| Static PES | `deepmd-dpa2` | test RMSE 6.63 meV/atom and 151 meV/angstrom |
| Ionic transport | `deepmd-se_atten_v2` | conductivity MAE 0.09599531 mS/cm against AIMD |
| Electrochemical voltage | `chgnet` | voltage MAE 0.182777777777778 V against DFT |

The exact timing reductions give AIMD/MLIP ratios of 115.592, 113.288, and
173.080 for prototypes I, II, and III. The canonical detailed report is
[`examples/high_entropy_sulfide_reproduction/reports/manuscript_reproduction_summary.json`](../examples/high_entropy_sulfide_reproduction/reports/manuscript_reproduction_summary.json),
and the JSON report in this directory records that canonical path.

The acceptance fixture also replays all 247 available 228-atom screening
results. It deterministically selects `Mn6_Fe3_Ni8_Cu4_Zn7` at
18.420176685825222 S/m and links that candidate ID to the historical
`6_3_8_4_7` MSD/post-processing parity record. This link is not an AIMD/DFT
high-fidelity validation; that comparison remains
`EXTERNAL_VALIDATION_PENDING`. The unseen/cross-prototype result is represented
only as manuscript claim-level evidence because the paired raw
reference/prediction values are unavailable.

This is evidence replay, not a fresh MLIP, MD, AIMD, DFT, training, or
production-scale execution.
