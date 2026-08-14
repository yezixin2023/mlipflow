# Agent Skills 使用说明

十个仓库 Skills 位于 `.agents/skills/`。它们面向 Codex 等能读仓库规则、调用 CLI 和解析 JSON 的 tool-using Agent。

| Skill | 何时使用 | 对应计算插件 |
|---|---|---|
| `$mlip-workflow` | 拆解和监督端到端流程 | 组合全部插件 |
| `$high-entropy-structure` | 构造无序/SQS 高熵候选 | `high-entropy-structure` |
| `$pes-sampling` | 监督 DIRECT 代表构型选择、LASP/SSW 随机行走采样、历史 archive 归一化和受控 SSH-SLURM 执行 | `pes-sampling` |
| `$dft-labeling` | 用 pymatgen 准备 static/relax/AIMD VASP 输入，并以独立审批监督 local 或受控 SSH-SLURM static 标注 | `dft-labeling` |
| `$mlip-training` | 选择并监督 DeepMD/M3GNet/CHGNet/MACE 的训练、微调、数据/基础模型绑定和 SSH-SLURM 生命周期 | `mlip-training` |
| `$ase-md` | 用显式 DeepMD/M3GNet/CHGNet/MACE 模型在集群上运行受控 ASE NVT Langevin 或各向同性 MTK NPT 轨迹 | `ase-md` |
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

`$ase-md` 是独立的 trajectory 生成能力，不是 `$ionic-transport` 的别名。0.2 版支持单温度 `nvt-langevin` 与 `npt-isotropic-mtk`，支持 DeepMD、M3GNet/MatGL、CHGNet、MACE 四种显式本地模型。项目只绑定结构 SHA-256 与 `model_reference`；后者包含逻辑 model id、site-owned `MODEL_ROOT` 下的相对路径、file/directory kind 和模型内容 fingerprint。任何会触发 Hub/网络/包缓存自动下载的模型名都不能进入 scheduled contract。

每个 calculator 使用独立 site template family：`ase-md-deepmd`、`ase-md-m3gnet`、`ase-md-chgnet`、`ase-md-mace`。两种 ensemble 都必须显式声明温度、timestep、steps、trajectory/thermo interval、seed、COM policy、device/dtype 和抽象资源。NVT 额外要求 Langevin friction；NPT 不接受 friction，而要求显式 `pressure_gpa`、`thermostat_damping_fs`、`barostat_damping_fs`，并固定为各向同性 MTK volume fluctuation。

NPT 只能用于 full-rank 3D 周期 cell，`fix_com` 必须为 false，且不能带 ASE constraints。Framework 名称本身不作为 stress 证据：cluster runner 在第一步积分前必须确认具体 calculator/model 声明 stress 并实际返回有限 3x3 stress，否则立即失败。0.2 将 thermostat/barostat chain 长度固定为 3/3，chain integration substeps 固定为 1/1，并把这些值写入 approval/provenance。

Scheduler `COMPLETED` 后仍需第二次审批；checker 会核对完成步数、trajectory/thermo step-time schedule、有限热力学值、模型/结构 identity 和全部输出 SHA-256。NPT 还会核对 target pressure/damping identity，并要求 thermo 中的 pressure 有限、volume/cell lengths 为正。restart、多温度和 transport 暂不自动串接。

## Skill validation

开发或更新后，用仓库环境可用的 Python 运行官方 skill-creator 验证器：

```bash
python /path/to/skill-creator/scripts/quick_validate.py .agents/skills/<skill-name>
```

每个 `agents/openai.yaml` 的默认提示必须以对应 `$skill-name` 开头。
