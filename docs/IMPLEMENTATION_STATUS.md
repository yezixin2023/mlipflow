# 实现状态

最后更新：2026-08-17。

## 六个独立状态轴

这些状态不能合并成一个含糊的“完成度”。`DONE` 表示该轴的本阶段目标已完成；
`PARTIAL` 表示已有实质实现/证据，但仍缺规范要求的闭环；
`EXTERNAL_VALIDATION_PENDING` 表示等待站点外部验证；`NOT_PERFORMED` 表示明确
没有执行，且本阶段不要求执行。

| 状态轴 | 状态 | 已完成 | 尚未完成/不可声称 |
|---|---|---|---|
| `SOFTWARE_CORE` | DONE | 配置/schema、DAG、SQLite attempt lineage、审批摘要、plugin discovery；多 cluster local site control plane；remote template selection/fingerprinting；确定性 CPU/GPU rendering；fresh remote workspace；SSH-SLURM stage/submit/monitor/fetch/check/collect 状态机；evidence-gated routing 与只读命令语义均有软件测试 | 该状态不代表任何科学算法、远端模板或真实集群已验证 |
| `SCIENTIFIC_INTEGRATION` | DONE | 六模型 benchmark 归一化；真实 SQS/icet 薄封装；pymatgen `vasp-prepare` 与 DFT label contract；四框架 bundled training runner 与通用 scheduled contract；scheduled LASP；ASE NVT/NPT checkpoint restart；LAMMPS prepare/execute/binary restart；既有轨迹输运后处理、247 候选 normalization/ranking、电压 post-process 与论文证据 replay | DONE 只表示本阶段代码/契约/本地 fixture 集成完成；新 scheduled 能力除一次 DeepMD CPU 训练外尚未真实集群验证，不能据此声称生产数值 parity |
| `MANUSCRIPT_REPRODUCTION` | DONE | 紧凑真实论文证据可重建 static/transport/voltage 路由、三组 speedup、六模型 registry、247/247 大超胞筛选、top-1 历史输运连接与 unseen claim-level record；10 项自动验收通过，报告为 `REPLAY_VERIFIED` | DONE 只适用于 evidence-replay contract；top-1 AIMD/DFT 高保真仍 pending，unseen 仍不可数值验证，也不能声称重新计算论文 |
| `LOCAL_NUMERICAL_PARITY` | PARTIAL | 历史 MACE/DeepMD `target.msd` 输运脚本与 stdout 已作条件真实证据 parity；历史 247 候选 ID/数值规范化和排序已比较；两套 LASP/SSW archive 已由本地 wrapper 实际重放并经 Adapter check=`OK` | SQS、DIRECT、真实 LASP/SSW 数值、DFT、训练、RDF/局域结构、全部六模型 fresh prediction、总能→电压和 unseen transfer 尚无 numerical parity |
| `REAL_HPC_INTEGRATION` | SCIENTIFIC_PROGRAM_VERIFIED_ON_ONE_SITE | synthetic multi-cluster `site.yaml`、fake template library/backend 覆盖 resolution、attempt path、stage、scheduler reconciliation、fetch 与 scientific check/collect；同一个真实 SSH-SLURM 站点上先完成 CPU tiny smoke，**随后由 MLIPFlow 端到端执行一次真实 DeePMD-kit 训练**（job `27356151`、500 步 fresh training、scheduler `COMPLETED`、MLIPFlow `OK`），并与历史 run 的前 6 个 reporting step 逐位一致 | 已验证的科学程序只有 DeePMD `dp train` 一个，且限于 CPU、单节点、单进程、`se_e2_a`、一个数据集、500 步 bounded run；VASP/LAMMPS 仍未在调度器上运行；GPU、多节点、cancellation、排队/容错、生产规模训练与该站点之外的集群同样未验证 |
| `PRODUCTION_SCALE_RERUN` | NOT_PERFORMED | 有意复用既有输出，避免无必要的昂贵计算 | 未重跑真实 LASP/SSW、>60k DFT 数据、AIMD、长 MLIP-MD、完整训练、247 项筛选或生产电压 DFT；本阶段不要求这些工作 |

生产 HPC 不可用不会单独阻止 `SCIENTIFIC_INTEGRATION` 或
`MANUSCRIPT_REPRODUCTION` 完成。当前两者均已按本阶段边界完成；
`LOCAL_NUMERICAL_PARITY` 仍为 `PARTIAL` 的原因是上述缺失科学对照，而不是 HPC。

## 十个科学插件的真实边界

| 插件 | 当前可执行能力 | 可重放证据 | 已有科学比较 | 当前边界 |
|---|---|---|---|---|
| `high-entropy-structure` | bundled、seeded `icet.generate_sqs_from_supercells` wrapper；显式 sublattice/count/cutoff/repeat/steps/output | 标准 generation manifest | 用真实 57 原子 prototype、ASE 3.28.0/icet 3.2 完成两次 100-step bounded smoke；结构/manifest byte-identical，成分正确 | `LOCAL_INTEGRATION_SMOKE_PASS` 不等于 production SQS 或历史 parity；历史脚本无 seed，外部 prototype 未复制 |
| `pes-sampling` | bundled、local-only MAML DIRECT runner；用户自备 LASP 的 local `shell=False` wrapper 与受控 `ssh-slurm` scheduled execute；已有 archive 的 normalize replay | DIRECT result manifest；LASP `allstr.arc`、可选 `best.arc`/`md.arc`、归一化结构和 bounded scheduled artifacts | DIRECT adapter/runner contract 测试；两套历史 archive 实际本地 replay；fake executable local/scheduled contract 测试通过 | DIRECT 不暴露 seed 控制且无历史同输入 numerical parity；不捆绑或实现 LASP；真实授权 LASP 和站点 scheduler 尚未执行 |
| `dft-labeling` | 两个独立操作：bundled pymatgen `vasp-prepare` 生成 static/relax/AIMD 输入；`label` 支持 local user wrapper，static VASP 可向通用 `ssh-slurm` backend 提供 scientific stage/fetch/check/collect 合同 | `dft-input-manifest.json`、可收集的 POSCAR/INCAR/KPOINTS；标准标签与 raw-output hashes；远端 input/output/logs/completion allowlist | SI/历史单点参数只读 source audit；真实 pymatgen 生成 smoke；synthetic site + fake template/backend 完成 fresh stage/submit/reconcile/bounded fetch/pinned check E2E | POTCAR 只从 `PMG_VASP_PSP_DIR` 组装；远端可上传但永不 fetch/collect/入库；真实 VASP/HPC 尚未运行、无数值 parity；relax/AIMD scheduler、array/continuation pending |
| `mlip-training` | DeepMD、M3GNet/MatGL、CHGNet、MACE 的 bundled train/finetune runner；local compatibility path 与通用 `ssh-slurm` contract | 标准 training result、cluster run report、模型 artifact identity；一次真实 DeepMD scheduled learning curve | 5 份历史源审计；一次真实 DeepMD 500 步 early-training 数值复现逐 reporting step 一致 | 真实 scheduled 验证仅覆盖单节点 CPU DeepMD fresh train；其他框架、fine-tune、GPU、多节点和生产规模未验证 |
| `ase-md` | 显式四框架模型的单温 NVT Langevin / isotropic MTK NPT；周期 JSON checkpoint、失败 salvage 和新 attempt 精确 restart | trajectory/index/thermo、checkpoint、final structure、result/cluster reports | fake scheduler 覆盖 NVT、NPT、checkpoint、TIMEOUT salvage 与 restart 身份 | 尚未在真实站点运行；不做轨迹拼接、温度序列或输运分析，NPT 必须由具体模型通过 finite-stress probe |
| `lammps-md` | 本地生成 DeepMD/MACE/MatGL CPU/GPU NVT/NPT deck；受控 scheduled execute；交替 binary restart、salvage 与 runtime-bound resume | prepared manifest、trajectory/final data/restart/log/result；restart runtime identity | 本地 fixture 覆盖 prepare、scheduled lifecycle、completion 与 restart deck/identity | 尚未运行真实 LAMMPS；需要匹配模型接口的站点 build，restart 只保证 pinned runtime 下 state continuity，不保证跨平台 bitwise identity |
| `mlip-benchmark` | `evaluate-fresh` 通过共享 model runtime 加载六个 exact family，生成 canonical prediction evidence，再复用唯一 MAE/RMSE/Pearson/ranking；`normalize-execute` 只重算 supplied pairs | 指纹固定的 DeepMD/CHGNet workbook、prepared JSON/CSV/XLSX、论文表 | fresh contract/injected-predictor 全链测试完成；三份 DeepMD 与一份 CHGNet 历史 source/output 为 `REPLAY_VERIFIED` | 真实六模型 production-scale fresh parity 未执行；M3GNet benchmark output 和 DPA-2 raw pairs 为 `MISSING_SOURCE`；历史 CHGNet/DeepMD 未声明的物理单位不猜测 |
| `ionic-transport` | local-only formal runner：ASE/LAMMPS/VASP trajectory → pymatgen Structure/`DiffusionAnalyzer`；MSD → `get_diffusivity_from_msd`；Structure 可用时 `get_conversion_factor`；Arrhenius → `fit_arrhenius(linear)` | analysis manifest、runtime/source/result SHA/size、pymatgen API checker rerun；历史 stdout/integration manifest | trajectory D/σ/time、MSD-only D/σ、Arrhenius 均与 pymatgen public API parity；保留 MACE/DeepMD 历史 parity | formal 需要 `[transport]` 且无 NumPy fallback；极短轨迹不提供收敛输运数值；历史 N=7 仅限隔离 parity convention |
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
Adapter 的 plan、subprocess、check、collect 均成功，3 runs、0 failed。1269-byte
structure SHA-256 为
`b283f4c4ddbc161214847fcd7d69948b7a453ea5f74d0d32bb6d37ec817916d5`；
79,462,305-byte model SHA-256 为
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`，未复制入仓库。
报告明确 `integration-smoke-only`；0.01 ps 每温度的输出不是收敛电导率、论文数值
parity 或 production MD。

## LASP/SSW 历史后处理与执行边界

`pes-sampling` 现在把三类行为明确拆成 `direct-select`、
`lasp-ssw-execute` 和 `lasp-ssw-normalize-replay`：

- `lasp-ssw-execute` 要求用户显式提供 LASP executable、`lasp_version`、ARC 输入结构、
  `lasp.in` 和所需辅助文件；运行始终为 local、`shell=False`，可以直接调用；MPI 只接受
  显式、可指纹化且 basename 为 `mpirun`/`mpiexec` 的普通文件路径与 `-np N`。fake executable 的 plan→execute→check→collect contract smoke
  已通过，但没有运行真实授权 LASP，也没有 scheduler/HPC 支持声明。
- `lasp-ssw-normalize-replay` 不执行 LASP。它解析现有 `allstr.arc`，先按
  `energy_max_ev` 过滤，再按 accepted order 应用 stride；可单独导出 `best.arc` 中的
  AIMD seed 候选，并记录 `md.arc` 清单。所有源文件和导出结构均绑定 SHA-256。
- 只读初始代表目录 `Li10M7P8S32/run/11113` 的小型文本输入已核对并在临时本地目录
  实际运行 wrapper：`allstr.arc` 6 frame，
  `energy<=0` 接受 5，stride 1 选中 5，`best.arc` 2 frame；解析得到 6/5/5/2，源哈希
  一致，wrapper rc=0、Adapter check=`OK`。其 `lasp.in` 记录 `SSW.SSWsteps=4`、
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
  system-ID 列表均有 SHA-256，但没有标准结果 manifest；
- MACE scratch：`--E0s=average` 后缺续行，之后 flags（含 seed）没有进入实际 trainer
  命令；
- MACE fine-tune：seed=3、CPU、float64 可提取，但没有标准结果 manifest；
- CHGNet：路径依赖、没有显式 seed、有 top-level execution；
- M3GNet：split=[0.8,0.05,0.15]、random_state=42、shuffle=false，但混用
  legacy M3GNet 与 MatGL API，并有 top-level execution；
- 四个入口都没有 MLIPFlow 标准 training result contract。

这是 source audit，不是训练执行、模型质量 parity 或 framework compatibility 证明。

## 轻量开源数据边界

采用 wrap-before-rewrite。仓库只保留小型派生表、标准 manifest、来源 locator 和
SHA-256；不复制原始文档、模型 checkpoint、完整训练集、AIMD/MD 轨迹、VASP
输出集合、WAVECAR、CHGCAR、POTCAR、私有集群配置或凭据。外部用户需自行提供：

| artifact type | purpose | supply mechanism |
|---|---|---|
| model/checkpoint | fresh prediction、MD 或 training continuation | project/site 配置中的本地 artifact path + 可选 SHA-256 |
| labeled reference/prediction pairs | metric-only execute 与 fresh parity | benchmark input path + unit/split/model/dataset fingerprints |
| trajectory / `target.msd` | transport post-processing | explicit input paths + sampling/carrier/volume conventions |
| prototype + composition manifest | SQS execution | project input path + exact counts/cutoffs/repeat/seed |
| LASP executable/input/archive | SSW execute or historical normalization | user-owned executable + explicit version；`input.arc`/`lasp.in`/auxiliary files or `allstr.arc`；source ID、SHA-256、selection policy、unknown-seed acknowledgement |
| training dataset/config | train/fine-tune | user-owned paths + dataset/config fingerprints |
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
