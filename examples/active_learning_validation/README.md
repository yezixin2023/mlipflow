# Bounded active-learning validation

This example supports one finite, fresh Strategy A validation round for the
Li10M7P8S32-derived disordered sulfide system. It is a validation recipe, not a
claim of production convergence.

## Fixed policy choices

- CHGNet same-framework committee with four members and seeds 11, 23, 37, and
  53. The four-member choice follows the original DP-GEN practice; committee
  neural-network studies commonly compare 1/2/4/8 members, while uncertainty-
  driven MLIP work often uses 5--10 members.
- NVT Langevin exploration at 400, 600, and 800 K, 1 fs timestep, 200 steps,
  and one replica. These short trajectories test the complete data path only;
  they do not establish transport convergence.
- Six fixed calibration structures and six disjoint immutable audit structures,
  obtained either from 12 freshly labeled, condition-balanced frames or from a
  deterministic 12-record capture of an existing converged VASP export. The
  bootstrap structures are never added to the training dataset. Historical
  capture keeps unknown condition metadata explicit and is not fresh DFT.
- Calibrated force-error QUERY and abort thresholds of 0.20 and 0.50 eV/A. The
  values are validation gates chosen at the scale of published force-error and
  DP-GEN trust-window examples, not universal constants for this chemistry.
- At most three DFT labels in the round: no more than two QUERY structures plus
  one deterministic SAFE spot check. Severe invalid structures are excluded.
- Audit limits are 0.05/0.08 eV/atom for energy MAE/RMSE,
  0.15/0.20 eV/A for force MAE/RMSE, and 0.50 eV/A for maximum atomic force
  error. A failure produces `CONTINUE`, not a relaxed threshold or a convergence
  claim.

The bounded fresh static labels use the repository's reviewed manuscript preset
(`ENCUT=450 eV`, `EDIFF=5e-6 eV`, spin polarization, D3(BJ), and gamma-only
sampling for the large cells), with `IALGO=38`, `ISMEAR=0`, `SIGMA=0.1 eV`,
and `NELM=700`. For the transition-metal sulfide validation, the explicit
slow-mixing additions are `AMIX=0.2`, `BMIX=0.0001`,
`AMIX_MAG=0.8`, `BMIX_MAG=0.0001`, and `ISYM=0`, following the VASP Wiki's
guidance for difficult magnetic/insulating convergence
(<https://vasp.at/wiki/Difficult_to_converge_systems>,
<https://vasp.at/wiki/AMIX_MAG>). These are recorded validation parameters, not
new plugin defaults.

The literature basis is deliberately limited to scale and committee design:

- Zhang et al., DP-GEN, `arXiv:1910.12690`
  (<https://arxiv.org/abs/1910.12690>).
- Schran et al., committee neural-network potentials, `arXiv:2006.01541`
  (<https://arxiv.org/abs/2006.01541>).
- Zaverkin et al., uncertainty-driven dynamics for active learning,
  *Nature Computational Science* (2023)
  (<https://www.nature.com/articles/s43588-023-00406-5>).
- Conformal calibration of MLIP uncertainty, *npj Computational Materials*
  (2026) (<https://www.nature.com/articles/s41524-026-02080-3>).

## Utility boundary

`prepare_round_inputs.py` only converts collected ASE trajectories, canonical
DFT records, split manifests, model references, and plugin results into explicit
handoff files. Model inference, DIRECT, VASP, dataset publication, training, MD,
benchmarking, and the final decision remain MLIPFlow plugin operations. Every
utility output is fresh and refuses overwrite.

`historical-chgnet-bootstrap` is the bounded replay alternative when a reviewed
CHGNet JSON export and its generator script are available. It filters one exact
composition, ranks source indexes by an explicit seed, writes only the requested
calibration/audit subset plus a benchmark view, and records both source paths.
Its provenance declares `existing-result-structured-capture` and
`fresh_numerical_execution=false`; it never upgrades missing historical
temperature metadata into a guessed condition.

Before assembling the cumulative dataset, run `split-seed-review` with the exact
canonical sources, previous training split, new QUERY source IDs, and bundled
`dft-labeling/dataset_contract.py`. It records the lowest non-negative seed for
which every prior train/validation record and newly acquired QUERY record remains
in train or validation; the dataset plugin independently recomputes the same
split during assembly. The independent immutable audit records are never part of
this split. `assessment-inputs` checks the collected split literally and refuses
to fold an ordinary test partition into the active-learning training pool. The
initial evaluation handoff follows the same rule: a canonical dataset's ordinary
test records are excluded, not renamed as validation records. Final audit metrics
are accepted only with fresh prediction evidence whose exact sample IDs match the
immutable audit IDs.

If a completed fresh committee result used the older split claim,
`prediction-split-rebind` creates a new structured correction: it preserves every
model/member prediction, requires unchanged calibration and audit IDs, records both
source paths, and declares that no numerical prediction changed. It never
overwrites the scheduler-collected evidence.

`direct-selection-replay` similarly turns a collected `OK` selection result into
the small DIRECT JSON contract needed to recompute selection after such a split
correction. It preserves the exact input/selected candidate IDs and records
the source result; it does not rerun or imitate DIRECT.

For the four-member Strategy A audit, `assessment-inputs` requires one named
metrics file and matching prediction-evidence file per retrained member. Every
member must contain the same immutable audit sample IDs and units. The handoff
preserves all member values and uses the conservative maximum error for each
declared audit gate, so no arbitrary seed is promoted to the production result.
The reported cumulative DFT-label count includes SAFE spot checks even when the
policy keeps them out of the cumulative training dataset; the canonical training
record count is therefore reported separately in the final validation summary.

For reviewed scheduled committee inference, copy
`cluster/run.sh.example` into the site's
`active-learning-committee-canonical` family and fill only the site-owned Python
interpreter and canonical model-root paths. Projects continue to contain model
root-relative references, never cluster paths.
Generate the model index from collected `model-reference.json` files with the
`model-index` helper with the exact canonical source files and bundled dataset
contract; it checks each adjacent training result and cluster report, then records
the shared canonical-label/export/split/foundation/config paths and parameters before inference.
