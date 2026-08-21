# 实现状态

最后更新：2026-08-20。

MLIPFlow 不再为输入、脚本、模型、数据集、结果、artifact、plan、approval、template、
launcher 或目录树生成自定义内容身份字段。普通记录保留路径、参数、软件版本、seed、attempt 与输出；
文件大小仅用于 bounded staging/fetch 的安全上限，不作为身份。

## 六个独立状态轴

这些状态不能合并成一个含糊的“完成度”。`DONE` 表示该轴的本阶段目标已完成；
`PARTIAL` 表示已有实质实现/证据，但仍缺规范要求的闭环；
`EXTERNAL_VALIDATION_PENDING` 表示等待站点外部验证；`NOT_PERFORMED` 表示明确
没有执行，且本阶段不要求执行。

| 状态轴 | 状态 | 已完成 | 尚未完成/不可声称 |
|---|---|---|---|
| `SOFTWARE_CORE` | DONE | 配置/schema、DAG、SQLite attempt lineage、布尔审批、plugin discovery；多 cluster local site control plane；remote template selection；确定性 CPU/GPU rendering；fresh remote workspace；SSH-SLURM stage/submit/monitor/fetch/check/collect 状态机；evidence-gated routing 与只读命令语义均有软件测试 | 该状态不代表任何科学算法、远端模板或真实集群已验证 |
| `SCIENTIFIC_INTEGRATION` | DONE | 六模型 benchmark 归一化；真实 SQS/icet 薄封装；pymatgen `vasp-prepare` 与 DFT label contract；四框架 bundled training runner 与通用 scheduled contract；scheduled LASP；ASE NVT/NPT checkpoint restart；LAMMPS prepare/execute/binary restart；离线主动学习 committee/calibration/selection/round decision；既有轨迹输运后处理、247 候选 normalization/ranking、电压 post-process 与论文证据 replay | DONE 只表示本阶段代码/契约/本地 fixture 集成完成；真实集群证据目前包括一次 14-atom LASP CPU tiny smoke、一次 DeepMD CPU 训练和 DeepMD、MatGL/GNNP 两条 LAMMPS CPU 5-step functional smoke，仍不能据此声称生产数值 parity |
| `MANUSCRIPT_REPRODUCTION` | DONE | 紧凑真实论文证据可重建 static/transport/voltage 路由、三组 speedup、六模型 registry、247/247 大超胞筛选、top-1 历史输运连接与 unseen claim-level record；10 项自动验收通过，报告为 `REPLAY_VERIFIED` | DONE 只适用于 evidence-replay contract；top-1 AIMD/DFT 高保真仍 pending，unseen 仍不可数值验证，也不能声称重新计算论文 |
| `LOCAL_NUMERICAL_PARITY` | PARTIAL | 历史 MACE/DeepMD `target.msd` 输运脚本与 stdout 已作条件真实证据 parity；历史 247 候选 ID/数值规范化和排序已比较；两套 LASP/SSW archive 已由本地 wrapper 实际重放并经 Adapter check=`OK` | SQS、DIRECT、真实 LASP/SSW 数值、DFT、训练、RDF/局域结构、全部六模型 fresh prediction、总能→电压和 unseen transfer 尚无 numerical parity |
| `REAL_HPC_INTEGRATION` | BOUNDED_ACTIVE_LEARNING_ROUND_EXECUTED | 除既有真实 LASP、DeepMD 和 LAMMPS smoke 外，已按“CPU 集群仅运行 DFT、非 DFT training cluster 运行数据组装/训练/benchmark”的边界完成一次 Strategy A 有界 round：CPU VASP 的 400 K QUERY 与 SAFE spot 经 checker/collect=`OK`，600 K QUERY 因电子步未收敛为 `FAIL`；training cluster 上共同 CHGNet split、四个 seed 的 fresh fine-tune 和四份 immutable-audit fresh benchmark 均经 checker/collect=`OK`；最终本地 active-learning assessment=`OK` | 这是功能闭环而非科学收敛：插件 decision=`SCIENTIFIC_REVIEW_REQUIRED`、PES=`NOT_CONVERGED`、transport=`NOT_ESTABLISHED`；一份 QUERY DFT 失败，五项 audit gate 与三种温度覆盖均未通过，SAFE spot 为 false negative；仍未验证 GPU、多节点、生产规模或跨站点数值 parity |
| `PRODUCTION_SCALE_RERUN` | NOT_PERFORMED | 有意复用既有输出，避免无必要的昂贵计算 | 未重跑真实 LASP/SSW、>60k DFT 数据、AIMD、长 MLIP-MD、完整训练、247 项筛选或生产电压 DFT；本阶段不要求这些工作 |

## 真实有界主动学习 round

2026-08-20 的 Strategy A 验证使用四成员 CHGNet committee（seed 11/23/37/53）。
初始四条 canonical 标签与一条通过 checker 的 400 K QUERY 标签合并为五条训练数据；
确定性 seed 0 得到 train/validation/test=3/1/1，SAFE spot 按显式 policy 保持在训练集外。
四次 one-epoch fine-tune 与四次六结构 immutable-audit fresh inference 全部在非 DFT training cluster
完成；VASP static 只在 CPU 集群执行。两个选定 QUERY 中，400 K 成功，600 K 因达到
`NELM=700` 仍未电子收敛而保留为 `DFT_FAIL`，未伪造标签也未再次提交。

active-learning plugin 的最终 machine-readable decision 为
`SCIENTIFIC_REVIEW_REQUIRED`。最大跨 committee 成员的 audit error 为：energy
MAE/RMSE=0.0994958154/0.1002198332 eV/atom，force
MAE/RMSE=0.2249919450/0.3030649876 eV/Å，maximum atomic force
error=1.7365412545 eV/Å，五项均超过 policy。400/600/800 K 的
QUERY fraction 分别为 0.1/0.1/0.0，UNSAFE fraction 为 0.8/0.8/1.0。
SAFE spot 的 predicted/actual force error 为 0.0532529470/3.5769243713 eV/Å，
false-negative=1/1。累计有效 DFT 标签计数为 6，其中本轮新增成功标签 2；canonical
训练记录为 5。结论必须分别写为
`PES_ACTIVE_LEARNING_CONVERGENCE=NOT_CONVERGED` 与
`TRANSPORT_CONVERGENCE=NOT_ESTABLISHED`。

去标识的机器可读摘要见
`reports/active_learning_bounded_round_validation.json`。它不包含本地绝对路径、作业号、
模型权重、POTCAR 或集群凭据。

生产 HPC 不可用不会单独阻止 `SCIENTIFIC_INTEGRATION` 或
`MANUSCRIPT_REPRODUCTION` 完成。当前两者均已按本阶段边界完成；
`LOCAL_NUMERICAL_PARITY` 仍为 `PARTIAL` 的原因是上述缺失科学对照，而不是 HPC。

## 十一个科学插件的真实边界

| 插件 | 当前可执行能力 | 可重放证据 | 已有科学比较 | 当前边界 |
|---|---|---|---|---|
| `high-entropy-structure` | bundled、seeded `icet.generate_sqs_from_supercells` wrapper；显式 sublattice/count/cutoff/repeat/steps/output | 标准 generation manifest | 用真实 57 原子 prototype、ASE 3.28.0/icet 3.2 完成两次 100-step bounded smoke；结构/manifest byte-identical，成分正确 | `LOCAL_INTEGRATION_SMOKE_PASS` 不等于 production SQS 或历史 parity；历史脚本无 seed，外部 prototype 未复制 |
| `pes-sampling` | bundled、local-only MAML DIRECT runner；用户自备 LASP 的 local `shell=False` wrapper 与受控 `ssh-slurm` scheduled execute；已有 archive 的 normalize replay | DIRECT result manifest 与 sanitized local-smoke 路径/参数记录；LASP `allstr.arc`、可选 `best.arc`/`md.arc`、归一化结构和 bounded scheduled artifacts | 真实 MAML 2025.4.3 bounded smoke：4 个 CIF→2 个结构，adapter checker/collect=`OK`；两套历史 LASP archive 实际本地 replay；真实 LASP 3.6.0 NN 14-atom scheduled CPU smoke 在 fresh retry 后 14/14/14、checker/collect=`OK` | DIRECT smoke 未走 core run/approval/state lifecycle且无历史同输入 parity；不捆绑或实现 LASP；真实 LASP 仅为一个 tiny functional smoke，未建立 trajectory numerical parity、GPU/DFT/生产能力 |
| `dft-labeling` | 两个独立操作：bundled pymatgen `vasp-prepare` 生成 static/relax/AIMD 输入；`label` 支持 local user wrapper，static VASP 可向通用 `ssh-slurm` backend 提供 scientific stage/fetch/check/collect 合同 | `dft-input-manifest.json`、可收集的 POSCAR/INCAR/KPOINTS；标准标签与 raw-output 路径；远端 input/output/logs/completion allowlist | SI/历史单点参数只读 source audit；真实 pymatgen 生成 smoke；真实 CPU VASP bounded round 中两个 static label checker/collect=`OK`，另一个因电子未收敛正确为 `FAIL` | POTCAR 只从站点授权环境组装且永不 fetch/collect/入库；当前真实验证仅覆盖 large-cell static、单节点 128 MPI ranks、`NCORE=4`，无 VASP 数值 parity；relax/AIMD scheduler、array/continuation pending |
| `mlip-training` | DeepMD、M3GNet/MatGL、CHGNet、MACE 的 bundled train/finetune runner；local compatibility path 与通用 `ssh-slurm` contract | 标准 training result、cluster run report、模型 artifact 路径；真实 DeepMD 与 CHGNet scheduled learning evidence | 5 份历史源审计；真实 DeepMD 500 步 early-training 数值复现逐 reporting step 一致；非 DFT training cluster 上四个固定 seed 使用同一五记录 split 完成 CHGNet one-epoch fine-tune，均 checker/collect=`OK` | CHGNet 结果仅证明 bounded fine-tune 功能闭环，不是生产训练或数值 parity；M3GNet/MACE scheduled、GPU、多节点和生产规模未验证 |
| `ase-md` | 显式四框架模型的单温 NVT Langevin / isotropic MTK NPT；周期 JSON checkpoint、失败 salvage 和新 attempt 精确 restart | trajectory/index/thermo、checkpoint、final structure、result/cluster reports | fake scheduler 覆盖 NVT、NPT、checkpoint、TIMEOUT salvage 与 restart 身份 | 尚未在真实站点运行；不做轨迹拼接、温度序列或输运分析，NPT 必须由具体模型通过 finite-stress probe |
| `lammps-md` | 本地生成 DeepMD/MACE/MatGL CPU/GPU NVT/NPT deck；受控 scheduled execute；交替 binary restart、salvage 与 runtime-bound resume | prepared manifest、trajectory/final data/restart/log/result；restart runtime 路径/版本/参数记录；两条真实 CPU smoke 的匿名化报告 | 本地 fixture 覆盖 prepare、scheduled lifecycle、completion 与 restart deck；真实 cluster-a 上 `pair_style deepmd` 与 MatGL/GNNP `pair_style gnnp` 各完成一次 5-step NVT，exit 0、marker 和 checker/collect 均为 `OK` | 真实验证仅为 CPU functional smoke；MACE、native MatGL、GPU、NPT、binary restart、科学数值对照与生产 MD 未验证。GNNP 模型无 virial，不能把 smoke 中 pressure 当科学结果 |
| `mlip-benchmark` | `evaluate-fresh` 通过共享 model runtime 加载六个 exact family，生成 canonical prediction evidence，再复用唯一 MAE/RMSE/Pearson/ranking；`normalize-execute` 只重算 supplied pairs | 历史 DeepMD/CHGNet workbook、prepared JSON/CSV/XLSX、论文表；四份真实 CHGNet immutable-audit prediction evidence | fresh contract/injected-predictor 全链测试完成；三份 DeepMD 与一份 CHGNet 历史 source/output 为 `REPLAY_VERIFIED`；非 DFT training cluster 上四个 fine-tuned CHGNet member 分别对同一六结构 audit 做 fresh inference 并 checker/collect=`OK` | audit 数值未通过当前 policy，且不代表六模型 production parity；M3GNet benchmark output 和 DPA-2 raw pairs 为 `MISSING_SOURCE`；历史未声明单位不猜测 |
| `active-learning` | local deterministic `committee-evaluate`、`select-candidates`、`assess-round`，以及受控 scheduled committee inference；单模型 committee 与双模型独立校准风险并集共享同一实现 | 显式 committee predictions、calibration DFT、DIRECT、audit、spot-check、campaign/round lineage | Strategy A/B oracle contract 均可复算；一次 Strategy A 真实有界 continuation 完成 cumulative merge、共同 split、四 seed retrain、maximum-over-members audit 和 plugin assessment | 当前为离线有限 round；真实 round 如实得到 `SCIENTIFIC_REVIEW_REQUIRED`，不把评估节点 `OK`、预算耗尽、oracle replay 或部分 DFT 成功表述为 PES/transport 收敛 |
| `ionic-transport` | local-only formal runner：ASE/LAMMPS/VASP trajectory → pymatgen Structure/`DiffusionAnalyzer`；MSD → `get_diffusivity_from_msd`；Structure 可用时 `get_conversion_factor`；Arrhenius → `fit_arrhenius(linear)` | analysis manifest、runtime/source/result 路径、pymatgen API checker rerun；历史 stdout/integration manifest | trajectory D/σ/time、MSD-only D/σ、Arrhenius 均与 pymatgen public API parity；保留 MACE/DeepMD 历史 parity | formal 需要 `[transport]` 且无 NumPy fallback；极短轨迹不提供收敛输运数值；历史 N=7 仅限隔离 parity convention |
| `candidate-ranking` | legacy order normalization、显式单位、稳定排序/确定 tie-break、missing policy、top-k | candidate/metric-results/ranking manifests | 247/247 历史候选与 top-k 已本地兼容复核 | 不生成候选、不计算性质、不执行 MLIP/MD/DFT、不选择模型或替代高保真验证 |
| `electrochemical-voltage` | 默认 `compute-from-energies` 从严格 eV total-energy sequence 计算相邻平均电压；`replay-si-table-s11` 经标准 Adapter 生成并收集 metrics/ranking/provenance 三件套 | SI Table S11 的 18 行六模型结果 | S11 为 `REPLAY_VERIFIED`；总能公式尚无论文原始序列 parity | 不执行 DFT/MLIP；S11 replay 从已报告 voltage 开始，且明确 `model_execution=false` |

## 六模型论文证据与路由

registry 中登记：`deepmd-se_e2_a`、`deepmd-se_e2_r`、
`deepmd-se_atten_v2`、`deepmd-dpa2`、`m3gnet`、`chgnet`。没有模型带
`recommended_tasks` 提示，也没有核心品牌条件分支。当前真实证据产生：

| task | selected model | evidence criterion | scientific level |
|---|---|---|---|
| static PES | `deepmd-dpa2` | test energy RMSE=6.63 meV/atom、force RMSE=151 meV/Å；等权 minimize | REPLAY_VERIFIED |
| ionic transport | `deepmd-se_atten_v2` | Table S3 conductivity MAE=0.09599531 mS/cm；minimize | REPLAY_VERIFIED |
| electrochemical voltage | `chgnet` | Table S11 voltage MAE=0.182777777777778 V；minimize | REPLAY_VERIFIED |

这证明 static、transport、voltage 排名可以不同，并证明路由由 metrics/policy
产生。它不证明三个模型已经由 MLIPFlow 重新推理。

## 已完成的真实数值比较

- MACE Li10 dataset `11221`：历史 N=7 模式在 400/600/800 K 的两种 D 与 σ
  对历史 stdout 达到 rtol=1e-12；400 K linear-fit D 为
  9.261443310293172e-11 m²/s，σ 为 2.762778334022906 S/m。
- corrected N=10 保持 D 不变、σ 相对历史 N=7 精确乘 10/7；这是
  `EXPECTED_CARRIER_CONVENTION_DIVERGENCE`，不是“更改后的历史 ground truth”。
- DeepMD top candidate `6_3_8_4_7`：三个既有 `target.msd` 重现历史输出，
  legacy Arrhenius 的 300 K conductivity 为 18.420176685825222 S/m。
- 历史 Li10 筛选：247 个候选与 247 个结果完整对应；top-1 为
  `Mn6_Fe3_Ni8_Cu4_Zn7`，18.420176685825222 S/m。

这些测试在外部真实证据不可用的公共环境中可以 skip；因此文档记录的是已执行的
条件 evidence parity，不把 synthetic fallback 当科学验证。

另有 `reports/sqs_local_integration_smoke.json`：真实 ASE/icet wrapper 在同一 seed=23
下独立运行两次，57 原子结构与 generation manifest 逐字节一致。这是 bounded
integration smoke，不属于上述“历史数值 parity”，也不证明 100-step 搜索达到生产
SQS 质量。

`reports/ionic_md_local_integration_smoke.json` 记录真实历史 MD/analysis 源的本地闭环：
Li3YCl6、ASE 3.28.0、MACE 0.3.15、torch 2.10.0、CPU float64、seed=29，
400/600/800 K 各 10 production steps（1 fs、2-step interval、无 NPT/NVT）。标准
Adapter 的 plan、subprocess、check、collect 均成功，3 runs、0 failed。
structure 与外部模型均只按路径引用，未复制入仓库。
报告明确 `integration-smoke-only`；0.01 ps 每温度的输出不是收敛电导率、论文数值
parity 或 production MD。

## LASP/SSW 历史后处理与执行边界

`pes-sampling` 现在把三类行为明确拆成 `direct-select`、
`lasp-ssw-execute` 和 `lasp-ssw-normalize-replay`：

- `lasp-ssw-execute` 要求显式 `lasp_version`、ARC 输入结构、`lasp.in` 和所需辅助文件。
  local 路径由用户提供 LASP executable，始终 `shell=False`；local MPI 只接受显式、
  basename 为 `mpirun`/`mpiexec` 的普通文件路径与 `-np N`。`ssh-slurm`
  路径则由 site-owned template 提供 executable/MPI/site knowledge，project 不嵌这些绝对
  信息。真实 LASP 3.6.0 NN 14-atom CPU smoke 在 job `redacted` 完成 14/14/14 结构并经
  bounded fetch、scientific checker/collect 到 `OK`；这是功能闭环，不是 numerical parity。
- `lasp-ssw-normalize-replay` 不执行 LASP。它解析现有 `allstr.arc`，先按
  `energy_max_ev` 过滤，再按 accepted order 应用 stride；可单独导出 `best.arc` 中的
  AIMD seed 候选，并记录 `md.arc` 清单和来源路径。
- 只读初始代表目录 `Li10M7P8S32/run/11113` 的小型文本输入已核对并在临时本地目录
  实际运行 wrapper：`allstr.arc` 6 frame，
  `energy<=0` 接受 5，stride 1 选中 5，`best.arc` 2 frame；解析得到 6/5/5/2，
  wrapper rc=0、Adapter check=`OK`。其 `lasp.in` 记录 `SSW.SSWsteps=4`、
  `SSW.Temp=200`。
- 重采样代表目录的 `allstr.arc`、`best.arc`、`md.arc` 和 `lasp.in` 也以只读方式复制到
  临时目录并实际 replay：66 generated、25 energy accepted、9 selected、1 best、17 md，
  wrapper rc=0、Adapter check=`OK`；这与历史 25 个重编号输入和 9 个单点目录一致。
  该阶段记录 `SSW.SSWsteps=6`，后续 MD 温度段为 500→800 K。

上述计数与参数属于 **历史 post-processing replay**。历史源未记录随机种子，统一标为
`HISTORICAL_PARAMETER_UNKNOWN`，绝不反推或补写。它证明解析、过滤、顺序和选择链可重放，
不证明 LASP 的 SSW 数值、势函数、轨迹或论文结论已由 MLIPFlow 重新计算。LASP/SSW
采样、LASP 势训练、以及 DFT→LASP `TrainStr`/`TrainFor` 导出仍是三个独立能力。
sanitized 证据写入 `reports/lasp_ssw_historical_replay.json`，状态为
`HISTORICAL_POSTPROCESS_REPLAY_PASS`；报告不含原始结构、远端实际路径或凭据，临时源副本
已在审计后删除。

## 训练源审计结论

`reports/training_source_audit_summary.json` 固定了 5 份小型历史入口/最终配置的
只读审计，共发现 1 个 critical、13 个 high、2 个 medium issue：

- DeepMD `se_atten_v2` 最终配置：descriptor/fitting/training seeds=1/1/10、
  float64、300,000 steps、train/validation systems=747/747；配置文件及两个
  system-ID 列表路径已记录，但没有标准结果 manifest；
- MACE scratch：`--E0s=average` 后缺续行，之后 flags（含 seed）没有进入实际 trainer
  命令；
- MACE fine-tune：seed=3、CPU、float64 可提取，但没有标准结果 manifest；
- CHGNet：路径依赖、没有显式 seed、有 top-level execution；
- M3GNet：split=[0.8,0.05,0.15]、random_state=42、shuffle=false，但混用
  legacy M3GNet 与 MatGL API，并有 top-level execution；
- 四个入口都没有 MLIPFlow 标准 training result contract。

这是 source audit，不是训练执行、模型质量 parity 或 framework compatibility 证明。

## 轻量开源数据边界

采用 wrap-before-rewrite。仓库只保留小型派生表、标准 manifest 和来源 locator；
不复制原始文档、模型 checkpoint、完整训练集、AIMD/MD 轨迹、VASP
输出集合、WAVECAR、CHGCAR、POTCAR、私有集群配置或凭据。外部用户需自行提供：

| artifact type | purpose | supply mechanism |
|---|---|---|
| model/checkpoint | fresh prediction、MD 或 training continuation | project/site 配置中的本地 artifact path |
| labeled reference/prediction pairs | metric-only execute 与 fresh parity | benchmark input path + unit/split/model/dataset records |
| trajectory / `target.msd` | transport post-processing | explicit input paths + sampling/carrier/volume conventions |
| prototype + composition manifest | SQS execution | project input path + exact counts/cutoffs/repeat/seed |
| LASP executable/input/archive | SSW execute or historical normalization | user-owned executable + explicit version；`input.arc`/`lasp.in`/auxiliary files or `allstr.arc`；source ID、paths、selection policy、unknown-seed acknowledgement |
| training dataset/config | train/fine-tune | user-owned paths + framework parameters/seed |
| DFT engine/pseudopotential reference | labeling | licensed site environment；二进制数据不进入仓库 |
| site profile | optional HPC validation | 用户本地 `~/.mlipflow/site.yaml`：named cluster、SSH alias、remote template/work roots；不含 credential value |

## 发布/论文主张前的剩余门槛

公开 GitHub 的技术审计已完成：全量测试、ruff、Python 3.9 编译、reproduction
`--check`、repository hygiene、隐私/大文件扫描、wheel/sdist 资产闭包与离线安装均已
通过；README 的科学边界也已同步。尚未完成的是发布治理而非代码：项目所有者需确认
紧凑论文转录与派生 top-k 的再分发许可/正式引用，并把 `pyproject.toml`、
`CITATION.cff`、`SECURITY.md` 中的通用贡献者、仓库和私密联系占位信息替换为最终值。

在论文中声称“validated agentic workflow”前，除科学缺口外还必须验证 agent 本身：
冻结 LLM/agent 版本、系统提示与 Skill 版本，在相同 fixture 上保存端到端决策和审批审计
记录，并覆盖只读零副作用、危险操作拒绝/批准绑定、失败后停止与可重复路由等正反例。
科学上仍需：若主张历史/生产 SQS 质量则补结构统计或历史对照；若主张 LASP/SSW
科学数值能力则在授权环境执行真实程序并完成同输入对照；若把 MACE integration
smoke 提升为论文数值验证，则使用与主张匹配的模型/结构和可信 baseline，完成采样收敛
与同口径比较；完成至少一个训练 CLI/config/result extraction parity；取得 top
candidate 的 AIMD/DFT 高保真比较；取得 unseen/transfer 的最小数值表并验证。真实
HPC 仅在论文声称远端自治执行时才必须验证。
