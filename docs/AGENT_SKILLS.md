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
| `$lammps-md` | 为 LAMMPS-ready DeepMD/MACE/MatGL-M3GNet 模型生成可审核 CPU/GPU NVT/NPT 输入 bundle | `lammps-md` |
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

## LAMMPS MLIP input preparation

`$lammps-md` 0.1 是与 VASP `vasp-prepare` 类似的 prepare boundary：只生成输入并绑定 provenance，不执行 LAMMPS。输入是一个周期结构、一个 LAMMPS-ready `model_reference` 和一个显式 `lammps_config`；输出是 `structure.data`、`lammps-input-manifest.json` 以及请求的 `in.cpu.lammps` / `in.gpu.lammps`。

DeepMD 使用 `pair_style deepmd`，CPU/GPU 的差异留给 site-owned DeePMD/LAMMPS runtime；MACE 0.1 使用已导出的 ML-MACE TorchScript，CPU 为 `pair_style mace`、GPU 为单 GPU Kokkos `mace no_domain_decomposition`；MatGL/M3GNet 使用 `mgl create-lammps-model` 导出的 TorchScript，CPU 为 `pair_style matgl`、GPU 为 `pair_style matgl/kk`。CHGNet 暂不生成 native LAMMPS deck，因为当前 contract 没有固定一个可信的原生 export/pair-style bridge，继续使用 `$ase-md`。

模型绝对路径永远不写进 input deck。生成文件只引用 `${MODEL_FILE}`；后续 scheduler execution 必须在新的审批边界中从 site-owned model registry 解析并重新 fingerprint 模型，然后通过 `-var MODEL_FILE <resolved-path>` 注入。Version 0.1 支持 NVT 和各向同性 NPT，并把 workflow 的 fs/GPa 显式转换到 LAMMPS `metal` units 的 ps/bar；不自动生成 hybrid potential、charge/topology、long-range electrostatics 或 restart orchestration。

## Skill validation

开发或更新后，用仓库环境可用的 Python 运行官方 skill-creator 验证器：

```bash
python /path/to/skill-creator/scripts/quick_validate.py .agents/skills/<skill-name>
```

每个 `agents/openai.yaml` 的默认提示必须以对应 `$skill-name` 开头。
