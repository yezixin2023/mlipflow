---
name: dft-labeling
description: Supervise MLIPFlow DFT input preparation and labeling. Use for generating VASP POSCAR/INCAR/KPOINTS/POTCAR sets with pymatgen, choosing static/relax/AIMD contracts, reviewing pseudopotential functional and symbols, launching reviewed local or SSH-SLURM static DFT labels, or interpreting convergence and dataset manifests.
---

# DFT labeling

Use the `dft-labeling` plugin as the deterministic implementation. Do not generate scientific values in the Skill.

## Choose the operation

- Use `vasp-prepare` to generate a fresh VASP input set and `dft-input-manifest.json`. It does not run VASP or submit a job.
- Use `label` only after preparation has been reviewed. It is a separate expensive operation requiring explicit approval.
- After a scheduled `label` reaches final `OK`, require `canonical-labeled-dataset.json` with stable record IDs; do not treat a loose `labels.json` as the reusable training contract.
- Use `dataset-assemble` on `backend: ssh-slurm` to convert those verified artifacts into any non-empty subset of `deepmd`, `m3gnet`, `chgnet`, and `mace`. This operation does not run VASP or training, but it is still a state-changing scheduled publish and follows dry-run/approval.
- Never combine preparation and DFT execution into one implied action.
- Reject legacy `prepare_script` hooks in `label`; prepared inputs must arrive through the reviewed `dft-input-manifest.json` binding.

## Review `vasp-prepare`

Require an immutable structures manifest, a labeling config, an explicit Python interpreter that can import the mandatory `pymatgen` runtime dependency, and a portable pseudopotential reference. A missing or broken pymatgen import blocks/fails preparation; there is no fallback generator. The config must select exactly one calculation type:

- `static`: require `NSW=0` and `IBRION=-1`.
- `relax`: require positive `NSW`, `IBRION` in 1/2/3, and explicit `EDIFFG` and `ISIF`.
- `aimd`: require `IBRION=0`, positive `NSW` and `POTIM`, and explicit `TEBEG`, `TEEND`, and `MDALGO`.

Do not infer missing parameters. Report uncertainty and wait.

The optional `manuscript-static-v1` preset is evidence-bound. The SI reports `ENCUT=450 eV`, `EDIFF=5e-6`, and `IALGO=38`; an audited historical template supplies the remaining settings and a 1×1×1 Monkhorst mesh. Keep these evidence levels separate. The SI labels EDIFF as eV/atom, while VASP treats EDIFF as an absolute electronic threshold; preserve the numeric value and disclose the mismatch instead of silently converting it. Exclude historical `NPAR=4` because it is hardware/version dependent.

## Protect pseudopotentials

- Accept POTCAR data only through the user's licensed `PMG_VASP_PSP_DIR`, configured either as an environment variable or with pymatgen's official persistent settings. Never record the resolved path.
- Require `license_acknowledged=true`, the functional, and an explicit element-to-POTCAR-symbol map.
- Record the portable reference ID, functional, and symbols internally.
- Never display, copy, collect, package, commit, or place POTCAR content in reports. `collect` must return only the manifest plus POSCAR/INCAR/KPOINTS.
- Treat a pseudopotential functional/symbol mismatch as `FAIL`, not a warning.

## Execute and verify

Before preparation, use MLIPFlow dry-run and show calculation type, structure count, important inputs, pymatgen executable/version expectation, KPOINTS rule, pseudopotential reference, output directory, and that VASP/scheduler execution is false.

After preparation, accept `OK` only when the checker verifies source/config/reference paths, INCAR semantics, KPOINTS content, structure lineage, POTCAR policy, and every declared file. A missing result is `WAIT`; a mismatch is `FAIL`. Retry into a new attempt.

Before `label`, bind the reviewed `dft-input-manifest.json`, show DFT resources, units, convergence policy, and backend, then request explicit approval for the expensive execution. Scheduler completion alone is insufficient: require electronic convergence, applicable ionic convergence, non-truncation, the calculation-type-specific label cardinality, exact units, and matching dataset records.

For `backend: ssh-slurm`, require a bounded reviewed static/relax/AIMD batch and exact abstract resources: `cpus`, `gpus`, `memory`, and `walltime`. Use an explicit named `backend_profile` when the user chooses a cluster; otherwise MLIPFlow selects among site profiles by currently available capable nodes and records the selected profile in the dry-run. Never guess or place an SSH host, partition, account, QoS, module, executable, launcher, template path, work root, data root, full submit script, or `remote_cwd` in the workflow node. The user-local `~/.mlipflow/site.yaml` selects the cluster and separates `remote_template_root` from `work_root`; persistent remote scheduler templates provide site execution knowledge. `resources.cpus` is the MPI rank count **for each calculation/job**, never a budget divided across calculations. With `calculation_concurrency=1`, use one sequential scheduler allocation. When the user approves independent batch submission, set `calculation_concurrency` at least as large as the submitted calculation count: core creates one fresh sub-workspace and one Slurm job per calculation, submits them in one MLIPFlow action, alternates the site-owned partition candidate preference, and then lets Slurm run or queue each job independently. Show job count, full resources per job, maximum simultaneous ranks, selected profile, rendered scripts, and every exact sub-workspace in the dry-run. Track and aggregate all job IDs; never fetch POTCAR.

After the scheduler reports `COMPLETED`, run ordinary `advance`. Core bounded-fetches only the run's output allowlist and runs the scientific `check/collect`. Treat changed or missing output, malformed/truncated XML, OUTCAR without a normal footer, electronic steps reaching `NELM`, or label/raw-output mismatch as `FAIL`.

For a failed multi-structure label, inspect the calculation-level diagnostics before
retry. A normal `retry` creates a fresh attempt and never overwrites the failed attempt
or its remote workspaces. Each canonical record must retain its actual producing DFT
attempt. Never copy outputs manually or treat scheduler `COMPLETED` as scientific
success.

## Canonical dataset and remote framework views

The canonical record preserves stable `record_id`, source structure/calculation/ionic-step IDs, species and atom order, cell and fractional coordinates, total energy, forces, optional raw VASP stress, units, raw-output paths, and producing DFT attempt. Energy, force, stress, atom order, and cell transformations remain explicit. A checker must deterministically rebuild the record from verified labels and reject scientific mismatch.

`dataset-assemble` creates the framework-independent split before serialization. Use
`deterministic` for stable seeded record-level assignment. Use `group-aware` when every
canonical record has `source_group_id`; no group may cross train/validation/test. The
minimal `split.json` contains only dataset/split IDs, strategy, optional seed,
partition record IDs, and counts.

For `dataset-assemble`, bind the collected canonical dataset and explicitly review the
split strategy, seed, and fractions. The selected site's `dft-dataset/run.sh` owns the
reviewed dpdata/ASE Python and canonical data root. The remote converter:

- uses dpdata and rereads the generated DeepMD directory;
- emits the exact JSON record fields consumed by the M3GNet/MatGL and CHGNet runners;
- emits ASE extxyz for MACE;
- selects every framework partition from the exact same canonical record-ID list;
- records the necessary unit/sign/virial conventions in one assembly result;
- publishes only a fresh dataset-id path and refuses an existing target;
- returns `split.json`, one assembly result, and only the small dataset references needed by training.
- exports `benchmark/test.json` from the exact held-out test record IDs and returns a
  small `benchmark-dataset-reference.json` for scheduled fresh benchmarking. Energy is
  total eV, force is eV/angstrom, and stress uses ASE Voigt
  `xx,yy,zz,yz,xz,xy` in eV/angstrom^3 after the explicit sign/unit conversion from
  VASP-native kbar.

For an active-learning cumulative training set, bind a
`mlipflow/canonical-dataset-merge` manifest instead of editing or copying labels into a
new synthetic source dataset. Its ordered `sources` entries must use project-relative
paths. `dataset-assemble` verifies every source, rejects
duplicate `record_id` values and unit/target/convention drift, and concatenates records
in the declared order. Every record keeps its actual `source_dft_attempt`; the merged
top-level `source_dataset_ids` records the source datasets. Fixed calibration, fixed
audit, and SAFE spot-check labels must stay outside this training merge unless the
active-learning policy explicitly schedules them for a later training round. This is
reuse of verified existing results, not new DFT or independent numerical validation.

If a completed conversion published the immutable dataset but local collection/check
failed, diagnose the mismatch before retry. A fresh `dataset-assemble` retry may use the
site's reviewed `dft-dataset-reuse/run.sh` only when MLIPFlow can first reverify the
prior completion and collected scientific result. The reuse runner must compare the
canonical dataset and split exactly, regenerate and compare the held-out benchmark
with a tight numeric tolerance for cross-platform floating roundoff in derived
Cartesian coordinates, and check every requested framework tree and benchmark file
against the prior recorded paths and scientific records. It writes only the new attempt's
small collected outputs and never rewrites, removes, or republishes the existing
dataset directory. Any path, record, or scientific mismatch is `FAIL`, not a
reason to overwrite the publication.

All four assembled references have `kind: directory`: DeepMD contains train/valid/test
dpdata systems, M3GNet and CHGNet contain train/valid/test JSON, and MACE contains
train/valid/test extxyz. A missing dpdata dependency is an explicit dependency block
for a DeepMD conversion request; it does not invalidate canonical labels. Never replace
dpdata with an approximate writer.

If a collected framework reference already exists for the requested dataset and split,
hand it directly to `$mlip-training`; do not rerun VASP or reconvert.

Hand the collected benchmark reference directly to `$mlip-benchmark`; do not generate a
separate random test split or manually transform the framework-specific datasets.

Never describe contract tests, generated inputs, scheduler completion, or one smoke calculation as historical numerical parity. Successful local pymatgen preparation validates neither VASP execution nor any scheduler/HPC path.
