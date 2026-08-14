# Agent Skills 使用说明

仓库 Skills 位于 `.agents/skills/`。它们面向 Codex 等能读仓库规则、调用 CLI 和解析 JSON 的 tool-using Agent。

| Skill | 何时使用 | 对应计算插件 |
|---|---|---|
| `$mlip-workflow` | 拆解和监督端到端流程 | 组合全部插件 |
| `$high-entropy-structure` | 构造无序/SQS 高熵候选 | `high-entropy-structure` |
| `$pes-sampling` | 监督 DIRECT 代表构型选择、LASP/SSW 随机行走采样、历史 archive 归一化和受控 SSH-SLURM 执行 | `pes-sampling` |
| `$dft-labeling` | 用 pymatgen 准备 static/relax/AIMD VASP 输入，并以独立审批监督 local 或受控 SSH-SLURM static 标注 | `dft-labeling` |
| `$mlip-training` | 选择并监督 DeepMD/M3GNet/CHGNet/MACE 的训练、微调、数据/基础模型绑定和 SSH-SLURM 生命周期 | `mlip-training` |
| `$ase-md` | 用显式 DeepMD/M3GNet/CHGNet/MACE 模型运行集群 ASE NVT/NPT，并监督 checkpoint salvage 与断点续跑 | `ase-md` |
| `$lammps-md` | 为 LAMMPS-ready DeepMD/MACE/MatGL-M3GNet 准备 CPU/GPU NVT/NPT 输入并监督 SSH-SLURM 执行 | `lammps-md` |
| `$mlip-benchmark` | 产生机器可读 benchmark/ranking | `mlip-benchmark` |
| `$ionic-transport` | MD→MSD→D/电导/Arrhenius | `ionic-transport` |
| `$composition-screening` | 大超胞组分筛选与 top-k 验证 | `composition-screening` |
| `$electrochemical-voltage` | Li 含量能量与电压曲线 | `electrochemical-voltage` |

Skills 是监督说明，不是计算实现。更新 Skill 时，应以当前 CLI 帮助、plugin manifest 和 schema 为接口事实；旧 claw skills 已发现命令名和参数漂移，不可直接复制。

## PES sampling / LASP

`$pes-sampling` 中的 LASP 能力是刘智攀团队 LASP 的 SSW/random-walk 外部执行契约，不实现或捆绑 LASP 本身。local 与 `ssh-slurm` 都复用 `lasp_ssw.py` 的 ARC 解析、历史顺序、能量筛选和 provenance 逻辑。集群路径通过 `lasp-ssw/run.sh` 的 site-owned template 进入，项目节点不得嵌 SSH host、partition、module 或 LASP executable 路径。Scheduler `COMPLETED` 后仍需第二次 `advance` 审批，才会 bounded fetch 并运行 pinned checker。

## DFT labeling

`$dft-labeling` 将 `vasp-prepare` 与 `label` 视为两个独立 operation。前者只在 fresh attempt 中生成输入并记录 provenance，不运行 VASP；POTCAR 只能来自用户合法配置的 `PMG_VASP_PSP_DIR`，只记录 symbol/hash 且永不 collect/入库。后者必须重新 dry-run 并用新的 plan digest 审批。单结构 static `ssh-slurm` 还需要 scheduler 完成后的第二个 `advance` 审批，才会按大小/SHA-256 allowlist 拉回输出并运行 pinned checker；POTCAR 只允许 stage，永不 fetch。Agent 只产生科学输入与 `cpus/gpus/memory/walltime`，不猜 SSH host、partition、module、executable、template root 或 work root；这些由用户本地 site profile 与站点远端模板提供。

## MLIP training / fine-tuning

`$mlip-training` 现在对应统一的四框架 scheduler contract。DeepMD、M3GNet/MatGL、CHGNet、MACE 都可通过 `ssh-slurm` 计划 `train` 或 `finetune`；旧 DeepMD fresh-training reference 仍保留兼容路径。大数据和 foundation model 不经控制面复制，而由 project-scoped `dataset_reference` / `foundation_model_reference` 绑定逻辑 ID、site-root 下相对路径、file/directory kind 和 SHA-256 identity。集群模板族为 `mlip-deepmd`、`mlip-m3gnet`、`mlip-chgnet`、`mlip-mace`。

Agent 负责根据用户选择或 benchmark/routing 证据提出 framework/operation、绑定 config/data/model fingerprint、声明 seed/device/precision 和抽象资源；不得猜 cluster path、conda/module、foundation model 或超参数。第一阶段审批只允许 core stage/submit；scheduler 完成后第二次审批绑定远端 `cluster-run-report.json`、`training-result.json`、`model-artifact` 的 immutable inventory，再 fetch/check/collect。模型 size/SHA-256、dataset/foundation fingerprint、framework/operation/seed/device/precision 不一致都必须 `FAIL`。

## ASE molecular dynamics

`$ase-md` 是独立的 trajectory 生成能力，不是 `$ionic-transport` 的别名。0.3 版支持单温度 `nvt-langevin` 与 `npt-isotropic-mtk`，支持 DeepMD、M3GNet/MatGL、CHGNet、MACE 四种显式本地模型，并可通过周期 checkpoint 在 fresh scheduler attempts 之间断点续跑。项目只绑定结构 SHA-256 与 `model_reference`；后者包含逻辑 model id、site-owned `MODEL_ROOT` 下的相对路径、file/directory kind 和模型内容 fingerprint。任何会触发 Hub/网络/包缓存自动下载的模型名都不能进入 scheduled contract。

每个 calculator 使用独立 site template family：`ase-md-deepmd`、`ase-md-m3gnet`、`ase-md-chgnet`、`ase-md-mace`。两种 ensemble 都必须显式声明温度、timestep、总 `steps`、trajectory/thermo interval、seed、COM policy、device/dtype 和抽象资源。NVT 额外要求 Langevin friction；NPT 不接受 friction，而要求显式 `pressure_gpa`、`thermostat_damping_fs`、`barostat_damping_fs`，并固定为各向同性 MTK volume fluctuation。

NPT 只能用于 full-rank 3D 周期 cell，`fix_com` 必须为 false，且不能带 ASE constraints。Framework 名称本身不作为 stress 证据：cluster runner 在积分前必须确认具体 calculator/model 声明 stress 并实际返回有限 3x3 stress，否则立即失败。0.3 将 thermostat/barostat chain 长度固定为 3/3，chain integration substeps 固定为 1/1，并把这些值写入 approval/provenance。

断点续跑必须显式配置 `checkpoint_interval` 与 `restart_policy: auto-from-previous-attempt`。Checkpoint 是原子替换的严格 JSON，而不是 pickle：NVT 保存 atom state 与 PCG64 RNG state；NPT 额外保存 MTK particle/cell、thermostat chain、barostat chain extended state，并要求 restart 时 ASE 版本完全一致。`steps` 始终表示整条 trajectory 的总目标；retry 只从 checkpoint 的 global completed step 跑剩余部分。

当 Slurm 因 `TIMEOUT`、`PREEMPTED` 等终止时，core 不会直接把远端 checkpoint 当可信输入。`advance --dry-run` 先对 adapter 声明的 `failure_salvage` 子集做远端 size/SHA inventory；匹配审批后才 bounded fetch，并保持原 attempt 为 `FAIL/STOPPED`。随后 `retry` 创建 fresh attempt；新 run plan 只能 stage 立即上一 attempt 已经本地 salvage 的 checkpoint，并将 checkpoint SHA、来源 attempt、segment start/remaining steps 重新纳入审批。Scheduler `COMPLETED` 的普通科学 FAIL 不自动 resume。

每个 retry attempt 都生成独立 trajectory/thermo segment，使用连续的 global step/time 编号；0.3 暂不自动拼接 segment。成功完成后仍需第二次 `advance` 审批，checker 会核对 total completed steps、segment schedule、模型/结构/restart identity、checkpoint SHA，以及 NVT/NPT 对应热力学约束。多温度和 transport 仍不自动串接。

## LAMMPS molecular dynamics

`$lammps-md` 0.2 也采用与 VASP 类似的 prepare/execute 双边界。`lammps-prepare` 只在 local fresh attempt 中生成并审核输入；`execute` 必须重新 plan/approve，消费一个已经 fingerprint 的 `lammps-md-input-v2` bundle，并通过 `ssh-slurm` 运行一个明确的 CPU 或 GPU target。

Preparation bundle 仍由周期结构、LAMMPS-ready model reference 和显式 NVT/NPT config 生成。DeepMD CPU/GPU 均使用 `pair_style deepmd`；MACE 使用已导出的 ML-MACE TorchScript，CPU 为 `pair_style mace`、GPU 为单 GPU Kokkos `mace no_domain_decomposition`；MatGL/M3GNet 使用导出的 LAMMPS TorchScript，CPU 为 `pair_style matgl`、GPU 为单 GPU `pair_style matgl/kk`。CHGNet 继续由 `$ase-md` 处理，不能由 agent 猜一个未审核的 native LAMMPS bridge。

v2 input deck 永远只引用 `${MODEL_FILE}`。每个 deck 在 `final.data` 和 `final.restart` 写完之后才输出绑定总 step 数的 `MLIPFLOW_LAMMPS_COMPLETED` marker。Execute plan 会重新核对 prepared manifest、`structure.data`、所选 deck 的 SHA/size，并将 model id/fingerprint、target、ensemble、steps、type map、resources 与 site template family 放入审批。

集群模板族为 `lammps-deepmd-cpu/gpu`、`lammps-mace-cpu/gpu`、`lammps-m3gnet-cpu/gpu`。`PYTHON_BIN`、`LAMMPS_BIN`、`MODEL_ROOT`、MPI/srun argv、modules/conda 和 CUDA/Kokkos 环境都必须保留在 site-owned `run.sh`。Compute-node runner 只在 MODEL_ROOT 下解析模型，执行前后都重算 SHA，并通过 LAMMPS `-var MODEL_FILE` 注入模型路径。

CPU target 要求 `gpus=0`；MACE/MatGL GPU v0.2 要求恰好一张 GPU。DeepMD 可以申请多 GPU，但 site launcher 负责保持每个 MPI rank 最多使用一张 GPU；项目暂不加入独立 MPI-rank 参数。

Scheduler `COMPLETED` 后仍必须第二次 `advance` 审批。Checker 要求 `lammps-execution-result.json`、`cluster-run-report.json`、trajectory、`final.data`、`final.restart`、LAMMPS log/screen log 在已批准大小范围内并匹配 SHA；同时要求记录 LAMMPS version、completed steps 等于批准 steps，且 fetched `lammps.log` 中存在完全一致的 completion marker。Scheduler/LAMMPS 失败时只允许 salvage 已审核的 diagnostic log/report 子集，不把 partial trajectory 或 restart 误判为成功。

0.2 虽然收集 `final.restart`，但还没有实现 LAMMPS restart/resume。LAMMPS binary restart 与写出它的 executable/platform 有兼容边界，而且部分 fixes/commands 需要重新声明，因此后续 restart 必须单独设计，不能直接套 ASE-MD JSON checkpoint 逻辑。

## Skill validation

开发或更新后，用仓库环境可用的 Python 运行官方 skill-creator 验证器：

```bash
python /path/to/skill-creator/scripts/quick_validate.py .agents/skills/<skill-name>
```

每个 `agents/openai.yaml` 的默认提示必须以对应 `$skill-name` 开头。
