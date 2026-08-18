# MLIPFlow Development Handoff

Generated 2026-08-15 from the local repository state at `e45d750214551469f1004c298ed88506f1350e03`. The source worktree is clean; this handoff and the accompanying empty uncommitted patch are intentionally untracked.

## A. Non-negotiable architecture

- The control chain is `Agent -> Skill -> MLIPFlow Core -> Plugin -> scientific software`.
- A scheduler state of `COMPLETED` is not scientific `OK`. Scientific success requires the pinned plugin completion checks and collection to pass.
- The scheduled lifecycle is: resolve site profile -> select remote template -> create a fresh attempt -> stage approved files -> `sbatch` -> monitor -> inspect immutable remote inventory -> second approval -> bounded fetch -> scientific check -> collect.
- Site-specific environment activation, modules, thread variables, cluster-local data/model roots and the framework Python executable belong in remote templates.
- Projects and workflows must remain portable: no SSH host, partition, environment path, dataset absolute path, model absolute path, template root or work root in project nodes.
- Do not add MCP for this work.
- Do not refactor the already validated HPC core or introduce a second bundled runner/template architecture.
- The site-managed historical archive is production evidence and is always read-only. Never execute or edit historical scripts.

## B. Cluster/site layout

### CPU validation site

- Template root: `~/mlipflow/templates`.
- MLIPFlow remote root: `~/mlipflow`.
- The site-managed historical archive is read-only production evidence.
- The MACE template at `~/mlipflow/templates/mlip-mace/run.sh` was derived from a genuinely completed production run, not from an online API assumption. It activates the already installed, production-backed MACE Python environment and calls the staged `training_cluster.py`.
- The trusted CPU baseline is a read-only MACE fine-tuning archive, Slurm job `8662607`, with matching script, training log and final model evidence. Treat the remote template and that evidence as the authority for the exact interpreter/activation; do not replace or upgrade the environment.
- No current CPU smoke was run in this development sequence. Do not project the independent GPU site's package versions onto the CPU validation site.

### cluster GPU site

- SSH alias: `cluster`.
- Every new MLIPFlow runtime/generated file must be below `/fs1/home/yezixin/mlipflow`.
- Existing software below `/fs0/...` may be called read-only; it is not a generated runtime artifact.
- Do not modify the installed Python, MACE, Torch or CUDA environment. Do not run `pip install` or `conda install`.
- Verified runtime: Python `3.12.2`, MACE source/dist `0.3.12`, Torch `2.5.1`, Torch CUDA build `11.8`, GPU `NVIDIA A800-SXM4-80GB`, observed driver `580.126.09`.
- Active templates:
  - `/fs1/home/yezixin/mlipflow/templates/mlip-mace/run.sh`
  - `/fs1/home/yezixin/mlipflow/templates/slurm/gpu.sbatch`
- Site roots used by the template:
  - datasets: `/fs1/home/yezixin/mlipflow/datasets`
  - models: `/fs1/home/yezixin/mlipflow/models`
  - work: `/fs1/home/yezixin/mlipflow/work`
- The read-only Python executable selected by the template is the existing `${HOME}/.conda/envs/mace/bin/python` under the `/fs0` home filesystem.

## C. MACE historical evidence

### CPU-site baseline

- Read-only archived MACE fine-tuning run, job `8662607`.
- This is a successful production fine-tune with mutually consistent scheduler script, training log and final native model.
- It remains the production-backed source for CPU-site MACE environment and invocation behavior.

### GPU historical baseline

- Historical archive: `GPU_para_0`.
- Historical MACE: `0.3.9`; historical run length: 30 epochs.
- It is useful for configuration evidence only. Current cluster MACE `0.3.12` changes custom-foundation/multi-head behavior: for a custom foundation without `pt_train_file`, it warns and disables multi-head fine-tuning internally.
- Historical reproduction has not been completed. Neither current smoke validates exact reproduction, 30 epochs or production-scale behavior.

## D. MACE code changes currently made

The implementation below is present in current HEAD commit `e45d750214551469f1004c298ed88506f1350e03` (`mace_ready`). There is no remaining tracked uncommitted source diff. During handoff generation an external/user process committed the six previously modified files; this Codex session did not run `git commit`.

### `plugins/mlip-training/mlip_mace.py`

- Removed unsupported `--work_dir` from the MACE argv used by the older CPU-site-compatible parser.
- Corrected `ema` and `amsgrad` handling for `store_true` parser flags; added strict boolean handling for `multiheads_finetuning`.
- Added compatibility with installed MACE packages exposing either `run(args)` or only `main()`, restoring `sys.argv` after the fallback.
- Added pinned auxiliary `test_file` support relative to the approved train dataset directory, with SHA-256 verification.
- Added the verified MACE options needed by the GPU reference configuration: `valid_fraction`, `E0s`, `scaling`, `ema_decay` and `multiheads_finetuning`.
- Added `energy_key -> --energy_key` support so fresh training can read the existing extxyz `energy` labels without rewriting the dataset.
- Pre-creates `models/`, `logs/`, `checkpoints/` and `results/`; this fixes MACE `0.3.12` failing to save to an explicit non-existent `model_dir`.
- Native model selection prefers stage-two when present, otherwise the exact/native `.model`, and excludes `_compiled.model`.
- Requires exact requested/completed epoch evidence, `Training complete`, a final-epoch eval record and finite JSON history values.
- Reloads the fetched native model with Torch and records the native class.
- Records compute node, Python, MACE source/dist, Torch, Torch CUDA, CUDA driver and GPU model evidence.

### `plugins/mlip-training/adapter_cluster.py`

- Adds MACE-specific scheduled completion checks for requested/completed epochs, normal completion, finite history, native reload and required runtime environment evidence.
- Checks finetune cluster-report foundation `id` and `observed_fingerprint` symmetrically with the dataset identity.
- Enforces finite generic metrics, return code zero, fetched model size/SHA-256 and approved result/report identities.
- A substantial part of this file's large diff is Ruff formatting around the generic scheduled adapter; do not treat every formatting hunk as a new semantic feature.

### `src/mlipflow/backends.py`

- `SshSlurmBackend.status()` now tries `squeue`, then `sacct`, then `scontrol show job -o` when the site's accounting daemon is unavailable.
- This fallback is intentionally narrow. On the GPU site, completed jobs can still age out of `scontrol`, after which the state becomes `UNKNOWN`.

### Tests added with the MACE changes

- `tests/test_mlip_runners.py`: MACE parser/API compatibility, output directory creation, native selection, pinned auxiliary test data, fresh-train `energy_key`, exact epoch completion, non-finite rejection, shared metric validation and existing CHGNet early-exit/non-finite behavior.
- `tests/test_scheduled_mlip_matrix.py`: scheduled MACE completion/environment acceptance and failure paths, including foundation identity evidence.
- `tests/test_backends.py`: the `sacct -> scontrol` remote scheduler fallback.

### Existing related behavior

- `mlip_common.write_result()` rejects bool, non-numeric, NaN and Inf metrics instead of dropping them.
- CHGNet completion requires every requested target's train/val history length to match requested epochs and every history numeric value to be finite.
- These implementations are present at the current HEAD. The MACE commit added/extended regression coverage but did not modify `mlip_common.py` or `mlip_chgnet.py`.

## E. Real cluster validation evidence

### MACE finetune GPU smoke

`GPU_MACE_FINETUNE_1EPOCH_SMOKE = PASS`

- Job `894115`, node `gpu3`, GPU `NVIDIA A800-SXM4-80GB`.
- Runtime: MACE `0.3.12`, Torch `2.5.1`.
- Requested/completed epochs: `1/1`; normal completion true; all recorded metrics finite.
- Native reload: `OK`, class `mace.modules.models.ScaleShiftMACE`.
- Dataset and foundation observed fingerprints matched their approved references.
- Cluster report, bounded fetch, pinned check and collect passed; final MLIPFlow state is `OK`.
- Model size: `29,878,114` bytes.
- Model SHA-256: `8378e7ee99e80fcd805e74015e9b84dca232adc35afa15ea4d6f38548965dc95`.
- Remote workspace: `/fs1/home/yezixin/mlipflow/work/mace-gpu-finetune-1epoch-small-fs1-smoke/finetune-mace-gpu-small/attempt-0001`.

Preserve the failed precursor:

- Job `893999` ran the real epoch but failed during final native-model save.
- Root cause: MACE `0.3.12` did not create the explicit `model_dir`; `torch.save` reported that the parent `output/mace-work/models` did not exist.
- Fix: pre-create all explicit MACE output directories in the runner.
- Its remote attempt and logs must not be deleted. The older local project's persisted state may remain stale `PENDING` because `sacct` was unavailable and the `scontrol` record aged out before the approved FAIL transition could be applied; the remote completion and logs are the failure evidence.

### MACE fresh train GPU smoke

`GPU_MACE_TRAIN_1EPOCH_SMOKE = PASS`

- Job `894116`, node `gpu3`, GPU `NVIDIA A800-SXM4-80GB`.
- Environment: Python `3.12.2`, MACE source/dist `0.3.12`, Torch `2.5.1`, Torch CUDA `11.8`.
- Mode: `framework=mace`, `operation=train`, foundation `NONE`, LoRA absent, runtime multi-head fine-tuning false.
- Dataset source artifacts: 1,024 train frames and 256 test frames. MACE's seeded split was 922 train / 102 validation / 256 test, seed `123`.
- Train SHA-256: `1851a748894148dd68088f5c03a12ad548a56813f49af7e94539e31f6dafbaea`.
- Test SHA-256: `cde7428691048b2b08eafc71b9bc6001cdc778c13daae5b9a4a6512e9b45e05c`.
- Config fingerprint: `37f4cf22d865b256ec1974e83600a9661c0bdc460dd8ad58be1992bec3c29b65`.
- Key actual config: seed `123`; float64; batch `8`; valid batch `10`; valid fraction `0.1`; `r_max=5.0`; two interactions; hidden irreps `128x0e + 128x1o`; correlation `3`; `max_ell=3`; `lr=0.01`; weight decay `5e-7`; energy/forces/stress weights `1/100/1`; EMA false; AMSGrad true; `E0s=average`; `WeightedEnergyForcesLoss`; `ReduceLROnPlateau`; SWA/stage-two false.
- `E0s=average` used actual energy labels and least-squares regression. `energy_key=energy` was passed explicitly; no dataset rewrite occurred.
- Requested/completed epochs: `1/1`; normal completion true; all recorded metrics finite.
- Epoch-0 validation: loss `6.928844629632581e-05`; MAE E/atom `20.4583 meV`; RMSE E/atom `25.8125 meV`; MAE F `317.438 meV/A`; RMSE F `445.470 meV/A`.
- Model: `MACEFresh.model`, `16,837,011` bytes, SHA-256 `92594542844423bba08230e77a3deb08d81526f9f927a4531d89d3be3e621277`.
- Native class: `mace.modules.models.ScaleShiftMACE`; reload `OK` in the same cluster MACE environment.
- Cluster report `OK`, return code `0`, bounded fetch/check/collect `OK`, final state `OK`.
- Remote workspace: `/fs1/home/yezixin/mlipflow/work/mace-gpu-train-1epoch-small/train-mace-gpu-small/attempt-0001`.

## F. Logging behavior

Empty Slurm logs are normal for the current successful scheduled path:

- `logs/slurm-894116.out`: 0 bytes.
- `logs/slurm-894116.err`: 0 bytes.
- `output/training.stdout.log`: 6,984 bytes.
- `output/training.stderr.log`: 6,773 bytes.

`training_cluster.py` redirects `training_wrapper` and framework stdout/stderr into the bounded `output/training.*.log` files. The successful `run.sh` path prints nothing, so Slurm's launcher logs remain empty.

- `slurm-*.out/err`: launcher/template-level output.
- `training.stdout/stderr.log`: scientific framework runtime.
- `cluster-run-report.json`: execution, framework and artifact-fingerprint evidence.
- `completion.json`: wrapper exit/status evidence.

Do not interpret an empty Slurm stdout file as evidence that training did not run.

## G. Validation boundaries

Real cluster validation completed:

- MACE GPU fresh train, one epoch.
- MACE GPU finetune, one epoch.

Not validated:

- Exact historical reproduction.
- Full 30 epochs.
- Production scale.
- CPU fresh train.
- Multi-head equivalence between MACE `0.3.9` and `0.3.12`.
- LoRA.
- Restart/resume.
- Multi-GPU.
- Multi-node.

Do not expand the smoke-test claims beyond these boundaries.

## H. Current local repository state

Current state after generating this handoff package:

```text
branch = main
HEAD = e45d750214551469f1004c298ed88506f1350e03
HEAD subject = mace_ready
tracked diff = empty
```

The six files that were modified at the beginning of handoff generation are now committed in `e45d750`:

- `plugins/mlip-training/adapter_cluster.py`: scheduled MACE correctness checks, foundation identity symmetry, finite metrics and formatting.
- `plugins/mlip-training/mlip_mace.py`: MACE site/API compatibility, verified options, completion/model/environment evidence and fresh `energy_key` support.
- `src/mlipflow/backends.py`: SSH Slurm `scontrol` fallback when `sacct` is unavailable.
- `tests/test_backends.py`: scheduler fallback regression test.
- `tests/test_mlip_runners.py`: MACE runner/completion/fresh-train regressions plus metric and CHGNet coverage.
- `tests/test_scheduled_mlip_matrix.py`: scheduled MACE evidence acceptance/rejection tests.

Current `git status --short --untracked-files=all` contains only the handoff deliverables:

- `MLIPFLOW_HANDOFF.md`.
- `MLIPFLOW_UNCOMMITTED.patch`.

`MLIPFLOW_UNCOMMITTED.patch` is intentionally 0 bytes because the required `git diff --binary` was run after commit `e45d750` appeared and there is no tracked uncommitted diff. Do not replace it with `git show`; its purpose is to represent the actual uncommitted state.

Important ignored local evidence/configuration not captured by `git diff --binary`:

- `.mlipflow/site-cluster-gpu.json`: the earlier `/fs0` cluster profile used for failed attempt `893999`.
- `.mlipflow/site-cluster-gpu-fs1.json`: active GPU profile with `/fs1` template/work roots.
- `.mlipflow/cluster-fs1-bootstrap/`: local copies of the `/fs1` GPU templates and deterministic extxyz prefix extractor.
- `.mlipflow/mace-gpu-finetune-1epoch-smoke/`: failed full-dataset finetune control/evidence.
- `.mlipflow/mace-gpu-finetune-1epoch-small-fs1-smoke/`: successful finetune control state and fetched evidence/model.
- `.mlipflow/mace-gpu-train-1epoch-small/`: successful fresh-train control state and fetched evidence/model.

The `.mlipflow` tree also contains older unrelated ignored VASP/DeepMD validation projects. Preserve them; they are not part of this patch. Do not reset, clean, stash or automatically add any of these files.

## I. Tests currently passing

Latest targeted command actually run after adding fresh-train `energy_key` support:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
  tests/test_mlip_runners.py \
  tests/test_scheduled_mlip_matrix.py \
  tests/test_backends.py
```

Result: `52 passed`.

Latest relevant Ruff command actually run:

```bash
/opt/anaconda3/bin/ruff check \
  plugins/mlip-training/mlip_mace.py \
  tests/test_mlip_runners.py \
  src/mlipflow/backends.py \
  tests/test_backends.py \
  tests/test_scheduled_mlip_matrix.py
```

Result: `PASS` (`All checks passed!`).

The full repository pytest suite was not rerun after the final `energy_key` change. Do not claim a current full-suite pass without running it.

## J. Next recommended development step

Close MACE for now. The next main line should be CHGNet real-cluster validation.

The existing CHGNet guard requires `trainer.train()` to be followed by validation of every requested target's train/val history: lengths must equal the requested epoch count and all numeric history values must be finite, otherwise it raises `TrainingError`.

The next session should:

1. Read this handoff and inspect the current diff without resetting it.
2. Read the `mlip-training` Skill.
3. Inspect the actual cluster CHGNet/Python/Torch/CUDA environment read-only.
4. Design a site-compatible `mlip-chgnet/run.sh` in the existing template architecture.
5. Use a tiny, fingerprint-pinned one-epoch scheduled validation with a fresh `/fs1/home/yezixin/mlipflow` workspace.

Do not repeat the MACE investigation unless a regression is observed.

## K. Safety / operating rules

- Never modify the site-managed historical archive; never execute historical scripts.
- Do not modify installed cluster environments and do not install packages.
- Put every new cluster-generated file below `/fs1/home/yezixin/mlipflow`.
- Use fresh attempt workspaces for real scheduler attempts and preserve failed attempts.
- Use MLIPFlow's two exact approval gates for scheduled work unless the user explicitly pre-authorizes a precisely bounded lifecycle.
- Do not bypass MLIPFlow with direct `mace_run_train`, `sbatch`, fetch or collection commands.
- Do not commit or push without explicit user authorization.
- The cluster's actually installed package/source/API is authoritative over online latest documentation.

## L. Post-handoff CHGNet validation addendum (2026-08-15)

This addendum records development and real CPU-site scheduler evidence produced after the original handoff. It supersedes section J as the current development baseline. Validation moved from the unavailable GPU site to the CPU-only validation site, and the full-dataset comparison was later explicitly stopped in favor of the bounded 128-record workflow.

### CHGNet historical inputs and site runtime

- Historical labeled data: a read-only site-managed CHGNet JSON artifact.
- Dataset size: `283,387,555` bytes; SHA-256 `94946ec4f19a2e6a0e0e3d74e498b8a9d12255580746590b724f27c8c776ac0d`.
- Dataset schema: columnar JSON with 13,586 structures; energy labels are `energy_per_atom`, with `force` and `stress` targets.
- Foundation checkpoint (read-only installed artifact): `$HOME/deepmd-kit/lib/python3.11/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar`.
- Foundation size: `4,863,221` bytes; SHA-256 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.
- Runtime: `$HOME/deepmd-kit/bin/python`, Python `3.11.8`, CHGNet `0.4.0`, Torch `2.5.0+cu124`, CPU execution on `node901`.
- Active template: `$HOME/mlipflow/templates/mlip-chgnet/run.sh`; generated artifacts remain below `$HOME/mlipflow` and historical data/model inputs remain read-only.

### Full-dataset attempt intentionally stopped

- Project/node: `chgnet-cpu-site-finetune-1epoch-historical/finetune-chgnet-historical`.
- Slurm job: `27432782` on `node901`.
- The job was actively computing (roughly 46 CPU cores used on average and about 10 GiB resident memory), not stalled. The 13,586-structure dataset and 680 CPU training batches made the validation slow.
- The user explicitly chose to abandon this full-dataset comparison. MLIPFlow `stop --dry-run` and the matching approval were used; final state is `STOPPED`.
- Attempt-0001 and its remote workspace are preserved. No historical epoch-0 equivalence or scientific-success claim may be made from this stopped attempt.

### Bounded CHGNet finetune smoke

`CPU_SITE_CHGNET_FINETUNE_128_1EPOCH = PASS`

- Project/node: `chgnet-cpu-site-finetune-1epoch-prefix-128/finetune-chgnet-prefix-128`.
- Slurm job `27432906`, elapsed `00:03:27`, exit `0:0`, node `node901`.
- Uses the deterministic first 128 records without copying or rewriting the full dataset; seed `23`; split train/val/test = `102/6/20`.
- Requested/completed epochs: `1/1`; epoch 0 completed all `7/7` train batches.
- Historical freeze contract applied; frozen parameters: `358,349`.
- Final train E/F/S MAE: `0.026572 / 0.258298 / 0.637276`.
- Final validation E/F/S MAE: `0.037735 / 0.257174 / 0.411983`.
- Test E/F/S MAE: `0.033234 / 0.255551 / 0.385783`.
- All recorded metrics finite; native reload `OK`; class `chgnet.model.model.CHGNet`.
- Model: `2,164,077` bytes; SHA-256 `dc605dfad40eb6f03b47050c40ed1020ec00fa8e0ee8e1e512a4e515a0b83851`.
- Bounded remote inventory, second approval, fetch and pinned checker all passed; final MLIPFlow state is `OK`.

### Bounded CHGNet fresh-train smoke on the same split

`CPU_SITE_CHGNET_TRAIN_128_1EPOCH = PASS`

- Project/node: `chgnet-cpu-site-train-1epoch-prefix-128/train-chgnet-prefix-128`.
- Slurm job `27432926`, elapsed `00:04:32`, exit `0:0`, node `node901`.
- Operation is true fresh `train`: no foundation reference, no frozen modules, frozen parameters `0`.
- Uses the same dataset fingerprint, first 128 records, seed `23`, and train/val/test counts `102/6/20` as the finetune smoke.
- Requested/completed epochs: `1/1`; epoch 0 completed all `7/7` train batches.
- Final train E/F/S MAE: `0.244057 / 0.449831 / 0.822261`.
- Final validation E/F/S MAE: `0.204863 / 0.437312 / 0.461181`.
- Test E/F/S MAE: `0.198203 / 0.424631 / 0.497373`.
- All recorded metrics finite; native reload `OK`; class `chgnet.model.model.CHGNet`.
- Model: `4,847,881` bytes; SHA-256 `2fb7662bf977b2d8ca4ecec07521dd3f2e1795479b6abd8e4a2c168b83e9efdc`.
- Bounded remote inventory, second approval, fetch and pinned checker all passed; final MLIPFlow state is `OK`.

The finetune and fresh-train result manifests independently report identical split identities:

- train indices SHA-256: `ed343a4ba6a5b357c780524b0d6f0bd66fe4ade7f6bfe53d3718113ff86451df`.
- validation indices SHA-256: `2decab66d7420a7aa914e4e90670418b6046443a8e82d03440b2fb3a064beb62`.
- test indices SHA-256: `54001c943edcc2e1dc853f91a4ada9da307dbbdfd10521ee83f1b3cbb538f884`.

### CHGNet implementation now present in the uncommitted diff

- `plugins/mlip-training/mlip_chgnet.py`: explicit historical columnar-JSON contract, deterministic bounded `max_records`, seeded split identities, historical freeze groups, exact/finite completion evidence, final metrics, native reload and runtime environment evidence.
- `plugins/mlip-training/adapter_cluster.py`: pinned CHGNet completion, freeze, split, runtime and device checks.
- `tests/test_mlip_runners.py` and `tests/test_scheduled_mlip_matrix.py`: columnar/bounded dataset, deterministic split, freeze and scheduled completion/environment acceptance and rejection coverage.

Latest targeted verification after the real scheduler runs:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m ruff check \
  plugins/mlip-training/mlip_chgnet.py \
  plugins/mlip-training/adapter_cluster.py \
  tests/test_mlip_runners.py \
  tests/test_scheduled_mlip_matrix.py

PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
  tests/test_mlip_runners.py \
  tests/test_scheduled_mlip_matrix.py \
  tests/test_backends.py
```

Result: Ruff `PASS`; `61 passed`.

Current boundary: CHGNet CPU one-epoch finetune and fresh train are validated on the bounded real dataset prefix. Full-dataset historical epoch-0 comparison remains intentionally unvalidated because the user stopped that attempt. Do not repeat MACE or claim full-dataset CHGNet equivalence from these smoke results.

## M. Post-handoff M3GNet/MatGL validation addendum (2026-08-15)

This addendum supersedes the CHGNet next step. M3GNet fine-tuning and fresh training have both passed a bounded real-data, one-epoch CPU-site validation. The current high-level MatGL API and the historical lower-level API must coexist; the implementation therefore retains both instead of choosing one permanently.

### Dual MatGL API contract

- `m3gnet.api` accepts `auto`, `high_level`, or `legacy`.
- `auto` selects the high-level route when both `MGLDatasetLoader` and `MGLPotentialTrainer` are exposed, otherwise it falls back to the historical route.
- `high_level` retains the repository's `MGLDatasetLoader`/`MGLPotentialTrainer` direction, including a mapping of deterministic pre-built train/valid/test splits accepted by the newer trainer.
- `legacy` uses `MGLDataset`, `MGLDataLoader`, `PotentialLightningModule`, and an explicit Lightning trainer. This is the route actually exercised on the CPU validation site because its MatGL `1.1.3` exposes the DGL-era lower-level API and not the newer high-level pair.
- The current upstream MatGL direction is PyG-only and documents `MGLDatasetLoader` plus `MGLPotentialTrainer`; this is why the high-level route remains first-class. It is capability/unit covered locally but was not numerically exercised in this CPU-site validation.
- The high-level local-JSON route requires a loader exposing a compatible `from_json` factory. Otherwise it fails explicitly and directs the caller to a pre-built MatPES dataset integration or the historical JSON `legacy` route; it does not silently reinterpret the data.

### Real data, model and runtime identities

- Read-only dataset: the same site-managed CHGNet JSON artifact described above.
- Dataset size: `283,387,555` bytes; SHA-256 `94946ec4f19a2e6a0e0e3d74e498b8a9d12255580746590b724f27c8c776ac0d`.
- M3GNet uses the total-energy field `uncorrected_total_energy`, with `force` and `stress`; `energy_is_per_atom=false` is pinned and checked.
- Both successful runs use the deterministic first 128 records, seed `23`, cutoff `5.0`, three-body cutoff `4.0`, batch size `24`, and train/validation/test counts `102/12/14`.
- Fine-tune foundation: a read-only site-managed M3GNet potential directory.
- Runtime-verified foundation tree fingerprint: `sha256:8327c6b0748e517bc5dd8cab853ba5282ad241c0f63729e025eaebc59357955a`.
- Runtime: site-managed Python `3.11.8`, MatGL `1.1.3`, DGL `1.1.3`, Lightning `2.3.0`, and Torch `2.5.0+cu124`; CPU execution on an inventoried compute node.
- Dedicated CPU templates use the site's canonical template root; historical inputs stay read-only and generated workspaces remain below the site's canonical MLIPFlow work root.

### Preserved diagnostic attempts

- Job `27433040` failed before training because the initially recorded foundation tree fingerprint was wrong. The runtime observed `8327c6...955a`; the failed attempt is preserved.
- Job `27433162` was still `PENDING (Resources)` on one CPU partition; at the user's request it was explicitly stopped through MLIPFlow and preserved before switching to an idle node.
- Job `27433186` reached data graph construction but failed Lightning's Slurm validation because the first 09 template used `--ntasks=64` for a single Python process. Its structured result preserved the exact error.
- The 09 template was corrected to `--ntasks=1 --cpus-per-task=64`, and the run template now takes the CPU thread budget from `SLURM_CPUS_PER_TASK`. The original 08 template was not modified.

### Bounded M3GNet fine-tune smoke

`CPU_SITE_M3GNET_FINETUNE_128_1EPOCH = PASS`

- Project/node: `m3gnet-cpu-site-finetune-1epoch-prefix-128-v2/finetune-m3gnet-prefix-128`.
- Successful attempt: attempt 3, Slurm job `27433199`, exit `0:0`, final MLIPFlow state `OK`.
- Operation is true `finetune`; the approved foundation identity matches the observed identity and foundation element references are present.
- Requested/completed epochs: `1/1`; normal completion true; every recorded metric finite.
- Final train Energy/Force MAE: `0.0490935519 / 0.3821626902`.
- Final validation Energy/Force MAE: `0.0637179241 / 0.3252920210`.
- Test Energy/Force MAE: `0.0483060889 / 0.3861340582`.
- Native class: `matgl.apps.pes.Potential`; native reload `OK` in the same cluster MatGL environment.
- Fetched model archive: `1,087,681` bytes; SHA-256 `a7bbd66ff58061f070dd1598774d94848bef78139e4b2f3d65b7fbbd78b78a5d`.
- Bounded inventory, second approval, fetch and pinned checker all passed.

### Bounded M3GNet fresh-train smoke on the identical split

`CPU_SITE_M3GNET_TRAIN_128_1EPOCH = PASS`

- Project/node: `m3gnet-cpu-site-train-1epoch-prefix-128/train-m3gnet-prefix-128`.
- Slurm job `27433379`, exit `0:0`, final MLIPFlow state `OK`.
- Operation is true fresh `train`: no foundation model reference; the M3GNet architecture is initialized from scratch.
- Requested/completed epochs: `1/1`; normal completion true; every recorded metric finite.
- Final train Energy/Force/Stress MAE: `4.6592001915 / 0.4497961104 / 13.2503004074`.
- Final validation Energy/Force/Stress MAE: `4.5972533226 / 0.3912590444 / 11.3942518234`.
- Test Energy/Force/Stress MAE: `4.5754776001 / 0.4820899367 / 13.1811666489`.
- Native class: `matgl.apps.pes.Potential`; native reload `OK` in the same cluster MatGL environment.
- Fetched model archive: `1,074,281` bytes; SHA-256 `8144f0050ed2a968e6946486e55f193dc1919cd7bbf0628548a7252d88ffe1c8`.
- Bounded inventory, second approval, fetch and pinned checker all passed.

The successful fine-tune and fresh-train result manifests independently report identical dataset, seed and split identities:

- train indices SHA-256: `ed343a4ba6a5b357c780524b0d6f0bd66fe4ade7f6bfe53d3718113ff86451df`.
- validation indices SHA-256: `8919f5fff67ab15b883ed478664809630702250851fdd4ac7bdd9ec9ac65656c`.
- test indices SHA-256: `fa53b799b0778a60242f971a90a2aeef7055175e4f24358603488d8502a0b530`.

### M3GNet implementation and current verification

- `plugins/mlip-training/mlip_m3gnet.py`: dual API selection, strict historical JSON contract, deterministic bounded prefix/splits, exact finite epoch histories, total-energy guard, native save/reload, environment evidence and foundation element-reference handling.
- `plugins/mlip-training/adapter_cluster.py`: pinned M3GNet epoch, finite metric, split, foundation, environment, device and native-reload checks.
- `tests/test_mlip_runners.py`: historical schema, total-energy guard, deterministic split/history and dual API selection coverage.
- `tests/test_scheduled_mlip_matrix.py`: scheduled M3GNet train/fine-tune acceptance and completion-mismatch rejection coverage.

Latest targeted verification after both real scheduler runs:

```bash
/opt/anaconda3/bin/python -m ruff check \
  plugins/mlip-training/mlip_m3gnet.py \
  plugins/mlip-training/mlip_chgnet.py \
  plugins/mlip-training/adapter_cluster.py \
  tests/test_mlip_runners.py \
  tests/test_scheduled_mlip_matrix.py

PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
  tests/test_mlip_runners.py \
  tests/test_scheduled_mlip_matrix.py \
  tests/test_backends.py

git diff --check
```

Result: Ruff `PASS`; `72 passed`; `git diff --check` `PASS`.

Current validation boundary: only the `legacy` MatGL `1.1.3` route has real CPU-site numerical evidence. Both high-level and legacy control paths remain in code, but do not claim high-level PyG execution, historical full-dataset epoch equivalence, production convergence, GPU behavior, multi-node behavior, or model-quality ranking from these one-epoch smoke tests.

## N. Scheduled CPU resource semantics addendum (2026-08-15)

The M3GNet Lightning failure from job `27433186` is now fixed as a generic
scheduled-execution contract, not as a site-specific exception.

### Explicit execution models

- `scheduled_execution schema_version=3` requires `execution_model`.
- `single-python` means one Python process. `resources.cpus` is its thread budget,
  and the selected site template must declare exactly `--ntasks=1` and
  `--cpus-per-task={{CPUS}}`.
- `mpi` means `resources.cpus` is the MPI task/rank count. Its site template must
  declare `--ntasks={{CPUS}}` and must not reuse `CPUS` for `cpus-per-task`.
- MLIP training (including the strict legacy DeepMD scheduler path) and ASE MD use
  `single-python`.
- VASP, scheduled LAMMPS, and scheduled LASP use `mpi`.
- Schema v2 plans retain their historical `slurm/{cpu,gpu}.sbatch` lookup only for
  compatibility. New built-in scheduled plans use v3.

Core template selection is now
`slurm/<execution-model>/{cpu,gpu}.sbatch`. Before staging, the core parses the
raw Slurm directives and rejects an execution-model mismatch. The plan records
both `execution_model` and `cpu_resource_semantics`, so approval evidence states
whether `CPUS` means threads or ranks.

The repository contains four checked examples below
`examples/site_templates/slurm/`, covering single-python/MPI and CPU/GPU. A
regression test reads these files directly; additional tests reject the exact
bad Lightning layout (`single-python` with `--ntasks={{CPUS}}`) and the converse
MPI mis-mapping. Adapter tests pin MLIP training and ASE MD to `single-python`
and VASP/LAMMPS/LASP to `mpi`.

### Verification

- Ruff on all touched Python contract/adapter/test files: `PASS`.
- Python compileall for `src`, `plugins`, and `tests`: `PASS`.
- Scheduled/HPC targeted matrix: `PASS`.
- Full functional/scientific suite excluding repository hygiene: `462 passed, 3 skipped`.
  The complete pytest run has only two pre-existing
  repository-hygiene checks fail because the worktree contains the root
  `.DS_Store` and this untracked handoff intentionally records the real cluster
  alias. Neither item was modified as part of this change.
- `git diff --check`: `PASS`.

### Temporary workaround cleanup status

The exact cleanup targets were inventoried as the local validation-only site
profile/bootstrap and the dedicated remote CPU template tree. The deletion
approval was rejected, and a subsequent read-only check confirmed all three
targets still exist. No validation result, attempt, or workspace was deleted.
Cleanup therefore remains pending explicit permission; do not claim it has been
completed or remove any broader template/work directory.

## O. LAMMPS DeepMD and MatGL/GNNP CPU validation addendum (2026-08-16)

The `lammps-md` real-site status is no longer “contract only.” Two independent
CPU five-step NVT smokes completed the full MLIPFlow lifecycle on `hfeshell`:

- DeepMD: job `27442624`, `pair_style deepmd`, LAMMPS 2 Aug 2023, 5/5 steps,
  process exit 0, bounded fetch, checker/collect `OK`.
- MatGL/M3GNet through AdvanceSoft GNNP: final job `27442885`,
  `pair_style gnnp ${INTERFACE_PATH}` with `matgl ${MODEL_FILE}`, LAMMPS
  2 Aug 2023, 5/5 steps, process exit 0, bounded fetch, checker/collect `OK`.

Both used one CPU rank, no GPU, NVT at 400 K, timestep 1 fs, seed 11, and the
explicit `Li P S Mn Fe Ni Cu Zn` type map. Both result manifests bind the model,
prepared input manifest, LAMMPS executable, output files, step count and exact
completion marker. The log order proves that `final.data` and `final.restart`
were written before the marker.

The GNNP run preserved a failed attempt (`27442851`) whose embedded Python could
not import MatGL. This was classified as `ENVIRONMENT/SITE`, not a core failure.
The existing canonical `lammps-m3gnet-gnnp-cpu` family received the matching
Python 3.11 prefix/site-packages, and MLIPFlow retry created a fresh attempt.

The generic LAMMPS contract now distinguishes current native MatGL TorchScript
from explicit legacy `gnnp`/`m3gnet` Python bridges by interface, artifact kind
and artifact format; it does not infer an interface from a LAMMPS version.

Sanitized evidence is in
`reports/lammps_hfeshell_cpu_functional_smokes.json`. Do not call these runs
scientific validation or production MD. GPU, NPT, binary restart, native MatGL,
MACE and CHGNet LAMMPS remain outside this recorded success. The tested GNNP
model does not provide virial pressure, so its pressure output is not usable as
a scientific result.
