---
name: dft-labeling
description: Supervise MLIPFlow DFT input preparation and labeling. Use for generating VASP POSCAR/INCAR/KPOINTS/POTCAR sets with pymatgen, choosing static/relax/AIMD contracts, reviewing pseudopotential identity, launching reviewed local or SSH-SLURM static DFT labels, or interpreting convergence and dataset manifests.
---

# DFT labeling

Use the `dft-labeling` plugin as the deterministic implementation. Do not generate scientific values in the Skill.

## Choose the operation

- Use `vasp-prepare` to generate a fresh VASP input set and `dft-input-manifest.json`. It does not run VASP or submit a job.
- Use `label` only after preparation has been reviewed. It is a separate expensive operation requiring explicit approval.
- Never combine preparation and DFT execution into one implied action.
- Reject legacy `prepare_script` hooks in `label`; prepared inputs must arrive through the reviewed `dft-input-manifest.json` binding.

## Review `vasp-prepare`

Require an immutable structures manifest, a labeling config, an explicit pymatgen interpreter, and a portable pseudopotential reference. The config must select exactly one calculation type:

- `static`: require `NSW=0` and `IBRION=-1`.
- `relax`: require positive `NSW`, `IBRION` in 1/2/3, and explicit `EDIFFG` and `ISIF`.
- `aimd`: require `IBRION=0`, positive `NSW` and `POTIM`, and explicit `TEBEG`, `TEEND`, and `MDALGO`.

Do not infer missing parameters. Report uncertainty and wait.

The optional `manuscript-static-v1` preset is evidence-bound. The SI reports `ENCUT=450 eV`, `EDIFF=5e-6`, and `IALGO=38`; an audited historical template supplies the remaining settings and a 1×1×1 Monkhorst mesh. Keep these evidence levels separate. The SI labels EDIFF as eV/atom, while VASP treats EDIFF as an absolute electronic threshold; preserve the numeric value and disclose the mismatch instead of silently converting it. Exclude historical `NPAR=4` because it is hardware/version dependent.

## Protect pseudopotentials

- Accept POTCAR data only through the user's licensed `PMG_VASP_PSP_DIR`, configured either as an environment variable or with pymatgen's official persistent settings. Never record the resolved path.
- Require `license_acknowledged=true`, the functional, and an explicit element-to-POTCAR-symbol map.
- Record the portable reference ID, symbols, and verifiable component identity internally.
- Never display, copy, collect, package, commit, or place POTCAR content in reports. `collect` must return only the manifest plus POSCAR/INCAR/KPOINTS.
- Treat a pseudopotential identity mismatch as `FAIL`, not a warning.

## Execute and verify

Before preparation, use MLIPFlow dry-run and show calculation type, structure count, important inputs, pymatgen executable/version expectation, KPOINTS rule, pseudopotential reference, output directory, and that VASP/scheduler execution is false.

After preparation, accept `OK` only when the checker verifies source/config/reference identity, INCAR semantics, KPOINTS content, structure lineage, POTCAR policy, and every declared file. A missing result is `WAIT`; a mismatch is `FAIL`. Retry into a new attempt.

Before `label`, bind the reviewed `dft-input-manifest.json`, show DFT resources, units, convergence policy, and backend, then request explicit approval for the expensive execution. Scheduler completion alone is insufficient: require electronic convergence, applicable ionic convergence, non-truncation, one label per source structure, exact units, and dataset identity.

For `backend: ssh-slurm`, require one static structure, a named `backend_profile`, and exact abstract resources: `cpus`, `gpus`, `memory`, and `walltime`. Never guess or place an SSH host, partition, account, QoS, module, executable, launcher, template path, work root, full submit script, or `remote_cwd` in the workflow node. The user-local `~/.mlipflow/site.yaml` selects the cluster and separates `remote_template_root` from `work_root`; persistent remote `slurm/mpi/cpu.sbatch` or `slurm/mpi/gpu.sbatch` plus `vasp/run.sh` provide site execution knowledge. Show the selected profile, rendered scripts, resource contract, important staged inputs, and exact `attempt-XXXX` workspace in the dry-run. The run creates only that fresh workspace, stages the declared inputs and scripts, and submits one job. Never fetch POTCAR.

After the scheduler reports `COMPLETED`, run ordinary `advance`. Core bounded-fetches only the run's output allowlist, verifies transport integrity, and runs the pinned `check/collect`. Treat changed or missing output, malformed/truncated XML, OUTCAR without a normal footer, electronic steps reaching `NELM`, or label/raw-output mismatch as `FAIL`.

Never describe contract tests, generated inputs, scheduler completion, or one smoke calculation as historical numerical parity.
