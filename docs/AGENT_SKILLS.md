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
| `$lammps-md` | 为 LAMMPS-ready DeepMD/MACE/MatGL-M3GNet 准备和执行 CPU/GPU NVT/NPT，并监督 binary restart salvage 与断点续跑 | `lammps-md` |
| `$mlip-benchmark` | 产生机器可读 benchmark/ranking | `mlip-benchmark` |
| `$ionic-transport` | 仅本地监督已有 ASE/LAMMPS/VASP trajectory 或 MSD → MSD/D/电导/Arrhenius，以及 bounded MD smoke | `ionic-transport` |
| `$candidate-ranking` | 按已有数值指标做确定性 ranking/top-k | `candidate-ranking` |
| `$electrochemical-voltage` | Li 含量能量与电压曲线 | `electrochemical-voltage` |

Skills 是监督说明，不是计算实现。更新 Skill 时，应以当前 CLI 帮助、plugin manifest 和 schema 为接口事实；旧 claw skills 已发现命令名和参数漂移，不可直接复制。

## Candidate ranking

`$candidate-ranking` 只监督已有 candidate manifest 与 metric-results manifest 的单指标确定性排序。必须显式给出 metric、maximize/minimize、top-k 和 `reject`/`error` missing policy，并调用 `candidate-ranking` plugin；Skill 不生成候选、不运行 MLIP/MD/DFT、不计算性质，也不承担模型选择。

## Ionic transport

`$ionic-transport` 是 local-only 分析 Skill，不提供 `ssh-slurm`。正式 `analyze-existing` 使用 MLIPFlow 打包的 `ionic_conductivity.py`；项目只绑定已有 trajectory/MSD，不再提供或覆盖 `analysis_script`。当前输入为 ASE `production.traj`、LAMMPS unwrapped dump、VASP AIMD `vasprun.xml` 或显式 MSD 表；无法从可靠 metadata/structure 得到 timestep、temperature、carrier count、charge 或 volume 时必须显式声明。

正式链固定为 MSD → `D=slope/(2d)`（当前 `d=3`）→ 未做 Haven 修正的 Nernst–Einstein conductivity → 单直线 `ln(D)` 对 `1/T` Arrhenius。piecewise、Green-Kubo、Haven ratio estimation、anisotropic transport 和 bootstrap 不在合同内。`md-smoke-and-analyze` 只做 tiny local handoff，并复用同一分析 runner。

Runner 的 `analysis_manifest.json` 固定科学参数与 source/result identity。Checker 会重新读取 MSD curve 并独立复算 D、conductivity 和 Arrhenius，不能只相信结果 JSON。历史 Li10 `N=7` 只存在于隔离的 `legacy_script` parity convention；普通分析从结构或显式参数获得 composition-corrected count，绝不默认 N=7。

## PES sampling / LASP

`$pes-sampling` 中的 LASP 能力是刘智攀团队 LASP 的 SSW/random-walk 外部执行契约，不实现或捆绑 LASP 本身。local 与 `ssh-slurm` 都复用 `lasp_ssw.py` 的 ARC 解析、历史顺序和能量筛选逻辑。集群路径通过 `lasp-ssw/run.sh` 的 site-owned template 进入，项目节点不得嵌 SSH host、partition、module 或 LASP executable 路径。Scheduler `COMPLETED` 后，普通 `advance` bounded fetch 并运行 pinned checker。

## DFT labeling

`$dft-labeling` 将 `vasp-prepare` 与 `label` 视为两个独立 operation。前者只在 fresh attempt 中生成输入，不运行 VASP；POTCAR 只能来自用户合法配置的 `PMG_VASP_PSP_DIR`，只记录可验证的 reference 且永不 collect/入库。后者是独立的昂贵执行，需要单独审查。单结构 static `ssh-slurm` 在 scheduler 完成后由普通 `advance` bounded fetch 并运行 pinned checker；POTCAR 只允许 stage，永不 fetch。Agent 只产生科学输入与 `cpus/gpus/memory/walltime`，不猜 SSH host、partition、module、executable、template root 或 work root；这些由用户本地 site profile 与站点远端模板提供。

## MLIP training / fine-tuning

`$mlip-training` 对应统一的四框架 scheduler contract。DeepMD、M3GNet/MatGL、CHGNet、MACE 都可通过 `ssh-slurm` 计划 `train` 或 `finetune`。大数据和 foundation model 不经控制面复制，而由 project-scoped `dataset_reference` / `foundation_model_reference` 绑定逻辑 ID、site-root 下相对路径、file/directory kind 和可验证内容 identity。集群模板族为 `mlip-deepmd`、`mlip-m3gnet`、`mlip-chgnet`、`mlip-mace`。

Agent 负责根据用户选择或 benchmark/routing 证据提出 framework/operation、绑定实际 config/data/model、声明 seed/device/precision 和抽象资源；不得猜 cluster path、conda/module、foundation model 或超参数。scheduler 完成后普通 `advance` fetch/check/collect。模型或数据 identity、framework/operation/seed/device/precision 不一致都必须 `FAIL`。

## ASE molecular dynamics

`$ase-md` 是独立的 trajectory 生成能力，不是 `$ionic-transport` 的别名。0.3 版支持单温度 `nvt-langevin` 与 `npt-isotropic-mtk`，支持 DeepMD、M3GNet/MatGL、CHGNet、MACE 四种显式本地模型，并可通过周期 checkpoint 在 fresh scheduler attempts 之间断点续跑。项目绑定实际结构与显式 `model_reference`；后者包含逻辑 model id、site-owned `MODEL_ROOT` 下的相对路径、file/directory kind 和可验证内容 identity。任何会触发 Hub/网络/包缓存自动下载的模型名都不能进入 scheduled contract。

每个 calculator 使用独立 site template family：`ase-md-deepmd`、`ase-md-m3gnet`、`ase-md-chgnet`、`ase-md-mace`。两种 ensemble 都必须显式声明温度、timestep、总 `steps`、trajectory/thermo interval、seed、COM policy、device/dtype 和抽象资源。NVT 额外要求 Langevin friction；NPT 不接受 friction，而要求显式 `pressure_gpa`、`thermostat_damping_fs`、`barostat_damping_fs`，并固定为各向同性 MTK volume fluctuation。

NPT 只能用于 full-rank 3D 周期 cell，`fix_com` 必须为 false，且不能带 ASE constraints。Framework 名称本身不作为 stress 证据：cluster runner 在积分前必须确认具体 calculator/model 声明 stress 并实际返回有限 3x3 stress，否则立即失败。0.3 将 thermostat/barostat chain 长度固定为 3/3，chain integration substeps 固定为 1/1。

断点续跑必须显式配置 `checkpoint_interval` 与 `restart_policy: auto-from-previous-attempt`。Checkpoint 是原子替换的严格 JSON，而不是 pickle：NVT 保存 atom state 与 PCG64 RNG state；NPT 额外保存 MTK particle/cell、thermostat chain、barostat chain extended state，并要求 restart 时 ASE 版本完全一致。`steps` 始终表示整条 trajectory 的总目标；retry 只从 checkpoint 的 global completed step 跑剩余部分。

当 Slurm 因 `TIMEOUT`、`PREEMPTED` 等终止时，core 不会直接把远端 checkpoint 当可信输入。普通 `advance` 只对该 run 中 adapter 声明的 `failure_salvage` 子集做 bounded fetch 与 transport 验证；原 attempt 保持 `FAIL/STOPPED`。随后 `retry` 创建 fresh attempt；新 run 只能 stage 立即上一 attempt 已经本地 salvage 的 checkpoint，并显示来源 attempt、segment start/remaining steps。Scheduler `COMPLETED` 的普通科学 FAIL 不自动 resume。

每个 retry attempt 都生成独立 trajectory/thermo segment，使用连续的 global step/time 编号；0.3 暂不自动拼接 segment。成功完成后普通 `advance` 会核对 total completed steps、segment schedule、模型/结构/restart identity，以及 NVT/NPT 对应热力学约束。多温度和 transport 仍不自动串接。

## LAMMPS molecular dynamics

`$lammps-md` 0.3 将 prepare 与 execute 作为两个独立操作，并增加 runtime-bound binary restart。`lammps-prepare` 只在 local fresh attempt 中生成并审核 `lammps-md-input-v2`；`execute` 是独立的外部执行，通过 `lammps-<framework>-<cpu|gpu>` site template 运行 DeepMD、MACE 或 MatGL/M3GNet。CHGNet 仍不能由 agent 猜一个未审核 native pair-style bridge，继续使用 `$ase-md`。

Prepared deck 永远只引用 `${MODEL_FILE}`，并在 `final.data` 与 `final.restart` 写完之后才输出绑定全局总 step 的 completion marker。Execute plan 绑定 prepared manifest、`structure.data` 和所选 deck；MODEL_ROOT、LAMMPS executable、MPI/srun launcher、module/conda、CUDA/Kokkos 与远端目录都留在 site-owned `run.sh`。

长作业可显式设置 `checkpoint_interval` 和 `restart_policy: auto-from-previous-attempt`。Runner 不修改用户物理参数，只在 reviewed fresh deck 的 run 前增加 `restart N checkpoint.1.restart checkpoint.2.restart`，让两个 fixed binary restart 文件轮换。长进程开始前还写 `restart-runtime.json`，绑定 framework/target、prepared/model identity、checkpoint cadence、resources、executable/launcher identity 与平台信息。

当 Slurm `TIMEOUT`、`PREEMPTED`、`NODE_FAIL`、`OUT_OF_MEMORY` 等 scheduler terminal failure 发生时，普通 `advance` 从 bounded failure-salvage allowlist fetch 可用的两个 checkpoint、runtime sidecar 和 diagnostics。原 attempt 仍保持 `FAIL/STOPPED`。之后 `retry` 创建 fresh attempt，只接受立即上一 attempt 已经本地 salvage 的 runtime/checkpoint；后续 run 绑定其来源 attempt 和 executable/platform identity。Scheduler `COMPLETED` 后的普通科学 FAIL 不自动 resume；首个 checkpoint 之前就中断则 retry 必须 BLOCKED，不能偷偷 fresh-start。

控制面不解析 LAMMPS binary restart。新 attempt 在 compute node 上先要求 executable、平台、launcher、resources、model/prepared manifest、target 与 cadence 和 salvaged runtime 完全匹配，再用同一个 LAMMPS executable 探测候选，选择 timestep 最大且落在 checkpoint cadence、低于 global target 的有效 restart。Resume deck 使用 `read_restart ${RESTART_FILE}`，重新声明 MLIP `pair_style/pair_coeff`，原样复用同一 `fix mlipflow all nvt/npt ...` ID/style/参数，不执行原 fresh `velocity create`，并用 `run <original-total-steps> upto` 继续到原总步数。

成功 completion 由普通 `advance` fetch/check。Checker 继续核对 result/report、trajectory、`final.data`、`final.restart`、logs、LAMMPS version、completion marker 和内容 identity；resume 另外核对 selected checkpoint 来自当前 attempt 的 restart inputs、segment start 合法、source attempt/runtime identity 一致。正常成功后临时 periodic checkpoint 与 runtime sidecar 被移除，只保留 `final.restart`；失败时它们才作为 recovery artifacts salvage。

LAMMPS binary restart 不被描述为跨平台 portable checkpoint。即使 executable/platform/launcher/resources 都一致，MPI decomposition 和浮点顺序仍可能导致恢复轨迹与 uninterrupted run 数值分叉，所以 0.3 显式记录 `bitwise_exact_guaranteed: false`。能力定义是 **pinned compatible runtime 下的 state-continuous restart**。当前仍不自动 stitching 多 attempt trajectory/log segment，也不自动进入 transport 分析。

## Skill validation

开发或更新后，用仓库环境可用的 Python 运行官方 skill-creator 验证器：

```bash
python /path/to/skill-creator/scripts/quick_validate.py .agents/skills/<skill-name>
```

每个 `agents/openai.yaml` 的默认提示必须以对应 `$skill-name` 开头。
