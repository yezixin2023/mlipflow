# Phase 0 代码库盘点

## 范围、方法与边界

盘点日期：2026-08-10；LASP/SSW 补充审计：2026-08-11。

只读检查了三类来源：

1. 同级 `taskflow` 的 CLI、配置、skills 与文档；
2. 用户授权的远端 `<MLP_ROOT>` 目录和代表性脚本；
3. 工作区只读副本 `../claw` 中与采样、MACE 微调和扩散分析有关的脚本与技能文档。

远端检查没有上传、修改、提交或取消作业。初始盘点主动跳过模型权重、大型 MD/AIMD
轨迹、POTCAR、WAVECAR、CHGCAR、OUTCAR、`vasprun.xml`、训练大数据和海量结构，只抽样
文本脚本与小配置。后续 LASP/SSW 补充审计经授权只读复制了两套小型 BIOSYM 文本
archive（`allstr.arc`、`best.arc`、可选 `md.arc`）与 `lasp.in` 到临时本地目录，用于
实际运行 normalization wrapper；没有读取模型/POTCAR、执行远端程序或保留原始结构。
仓库只保存 sanitized 报告、portable locator、指纹和 manifest。

## 复用分级

| 等级 | 含义 | 处理原则 |
|---|---|---|
| A | 可直接复用的稳定接口 | 仅加校验、manifest 与测试 |
| B | 逻辑完整但缺少标准 CLI/配置 | 包装命令行、移除硬编码 |
| C | 可做薄适配器的研究脚本 | 固定 cwd/argv，解析成标准 JSON/CSV |
| D | 职责混杂或科学参数硬编码 | 保留算法证据，先拆分与验证再实现 |
| E | 重复、过时、危险或误导 | 不迁移，只在文档中留痕 |

## Taskflow 参考实现

| 来源 | 能力 | 分类 | MLIPFlow 处理 |
|---|---|---:|---|
| `versions/v1.0/tf` | 插件发现、状态采集、SSH/SLURM、生成、提交、JSON CLI | B/D | 只借鉴边界；重新实现为可测试的分层包 |
| `skill/*/skill.yaml` | 声明步骤、判据、模板、依赖和可选 DAG | B | 提炼为有 JSON Schema 的 `plugin.yaml` |
| `AGENTS.md` | LLM 监督、安全门控和巡检规则 | C | 拆成九个仓库 Agent Skills；安全约束下沉至代码 |
| 远端内嵌 collector | 单次 SSH 批量读取调度器/文件状态 | B | 后续作为只读 transport 优化；v1 先以可测接口实现 |

详细取舍见 `TASKFLOW_REFERENCE.md`。

## 远端 MLP 目录概览

`<MLP_ROOT>` 是用户在本地配置的外部数据根；真实主机、账户和绝对个人路径不写入可发布仓库。主要区域包括 `3-Element`、`4-Element`、`DeepMD`、`MACE`、`CHGnet`、`M3Gnet`、`Nequip`、`MTPs`、`LASP`、`LaspNN_in_lammps`、多种 Li–M–P–S 原型、`supply`、`Voltage` 和 `Band-Gap`。

### 结构生成与组分设计

| 代表路径（相对 MLP 根） | 输入 → 输出 | 依赖/问题 | 等级 | 目标插件 |
|---|---|---|---:|---|
| `3-Element/Li8/replace_sqs_6.5-4.5.py` | CONTCAR/CIF、整数组分、cutoff → SQS CIF/POSCAR | ASE、icet、pymatgen、mendeleev；无 CLI；位点硬编码 | C/D | `high-entropy-structure` |
| `Li10M7P8S32/SQS/*/replace_sqs.py` | 原型结构与组分 → SQS 结构 | 多个近重复版本 | C/E | `high-entropy-structure` |
| `Li10M7P8S32/SQS/7-5/gen_val_data.py` | 随机组分 → 2×2×2 验证结构 | 未固定随机种子 | B/C | `high-entropy-structure` |
| `Li10M7P8S32/SQS/supercell/replace_sqs.py` | 原型 → 2×2×1、约 2460 个候选 | 规模较大，无标准 manifest | C/D | `composition-screening` |

迁移要求：随机种子、原型指纹、组分约束、超胞、cutoff 和输出结构哈希必须进入 manifest；不得静默改变原子占位规则。

### PES 采样与 DFT 标注

| 代表路径 | 输入 → 输出 | 依赖/问题 | 等级 | 目标插件 |
|---|---|---|---:|---|
| `3-Element/*/sub_aimd_*.sh` | 初始结构 → 分温区 AIMD 轨迹 | 建目录、拷贝、提交和数值计算混在一起 | D | `pes-sampling` |
| `Li24M12P16S64/data/sub_aimd.slurm` | 0–800 K 分段 AIMD → CONTCAR/轨迹 | VASP、核数、路径硬编码 | D | `pes-sampling` |
| `Li24M12P16S64/data/extract.sh` | 轮询 CONTCAR → 每 100 次更新的快照 | 无明确超时/完成协议 | C/D | `pes-sampling` |
| `4-Element/Li8/scf/{single/INCAR,single/KPOINTS,sub_1.slurm}` | 快照 → VASP 单点输入/标签 | 参数与调度耦合；INCAR/KPOINTS 已只读审计并 SHA-256 固定，POTCAR 未读取 | D | `dft-labeling.vasp-prepare` 的 `manuscript-static-v1` 历史模板证据 + 独立 `label` |
| `*/vasp2lasptrain.py` | OUTCAR/CONTCAR → `TrainStr.txt`/`TrainFor.txt` | 约万份重复副本；解析规则可复用 | B/C/E | `dft-labeling` 的 LASP 训练数据导出；不是 SSW sampler |
| `Li10M7P8S32/run/{sub.slurm,single.py}` | 结构 → LASP SSW archive → 能量过滤/抽样 → 单点输入 | LASP、`pos2arc_`、绝对路径和 scheduler 耦合；历史 seed 未记录 | D | 已提炼为 `pes-sampling` 的 `lasp-ssw-execute`/`lasp-ssw-normalize-replay` local contract；不复用调度脚本 |
| `Li10M7P8S32/rerun/test_data.py` | VASP 输出 → 完成/失败清单 | 只读判据有价值 | C | `dft-labeling` checks |

LASP/SSW 定向只读审计没有执行远端程序或修改源文件。初始代表目录
`Li10M7P8S32/run/11113` 的 `allstr.arc` 含 6 frame，历史 `energy<=0` 规则接受 5，
accepted-order stride 1 仍选中 5，`best.arc` 含 2 frame；本地 normalization 得到相同
6/5/5/2/0，wrapper rc=0、Adapter check=`OK`，源 SHA-256 已核对。重采样代表目录的
`allstr.arc`/`best.arc`/`md.arc`/`lasp.in` 临时副本也实际运行 wrapper，得到
generated/accepted/selected/best/md=66/25/9/1/17，wrapper rc=0、Adapter check=`OK`；
这与历史 `single.py` 先排除正能量并重编号为 25 个 `output/input-*.arc`、随后每 3 个
accepted frame 取 1 并形成 9 个单点目录一致。初始 `lasp.in` 为
`SSW.SSWsteps=4`、`SSW.Temp=200`；重采样为 `SSW.SSWsteps=6`，后续 MD 温度段为
500→800 K。源中没有可恢复的随机种子，统一记录为 `HISTORICAL_PARAMETER_UNKNOWN`。
`reports/lasp_ssw_historical_replay.json` 以 `HISTORICAL_POSTPROCESS_REPLAY_PASS` 保存
sanitized 证据，不含原始结构、远端实际路径或凭据；临时副本已删除。这些结论只验证
archive 解析和历史后处理选择链，不是 LASP/SSW 科学数值 parity。

### 数据装配

| 代表路径 | 输入 → 输出 | 问题 | 等级 | 目标插件 |
|---|---|---|---:|---|
| `DeepMD/sub_AIMD_Li8_4ele.py` | OUTCAR/OSZICAR → DeepMD npy train/test | 绝对路径；随机拆分无 seed | D | `dft-labeling` / dataset assembly |
| `3-Element/sub_AIMD_Li{8,10,16}_3ele.py` | 多目录 VASP 输出 → DeepMD npy | 抽样边界与完成判据需复核 | D | `dft-labeling` |
| `Nequip/data/ase*`, `Nequip/data/npz*` | OUTCAR → extxyz/npz | 元素与 shift/scale 配置可能不一致 | D | dataset assembly |
| `CHGnet` 数据提取脚本 | OUTCAR/vasprun/CONTCAR → JSON | 各目录复制、路径硬编码 | D | dataset assembly |

未在抽样范围内找到完整、通用且生产可用的 DeepMD `input.json + dp train/freeze/test` 入口；当前证据更偏数据转换。首版不可把“发现数据转换脚本”误报为“已支持完整 DeepMD 训练”。

### 训练与模型导出

| 代表路径 | 输入 → 输出 | 依赖/问题 | 等级 | 目标适配器 |
|---|---|---|---:|---|
| `MACE/train/run.slurm` | train/test extxyz → MACE model/log | `mace_run_train`；一处续行符疑似缺失 | B/D | `mlip-training.mace` |
| `MACE/finetune/para-0/run.slurm` | foundation model + dataset → 微调模型 | CPU 资源硬编码 | B | `mlip-training.mace` |
| `MACE/evaluate/get_lammps_model.py` | MACE model → `lammps.pt` | 清晰单用途 | A/B | `model-export.mace-lammps` |
| `CHGnet/code/train_model_Li8.py` | merged JSON → checkpoint | 冻结层/5 epoch/CPU/路径硬编码 | D | `mlip-training.chgnet` |
| `M3Gnet/all_train/train.py` | CHGNet 风格 JSON → model/predictions | 混用 matgl/m3gnet，训练与评估耦合 | D | `mlip-training.m3gnet` |
| `Nequip` 训练配置 | extxyz/npz → NequIP model | 本地复制的上游源码不应迁移 | C/D/E | 可选 `mlip-training.nequip` |
| `LASP` train dirs | `TrainStr/TrainFor` + `lasp.in` → `.pot` | 专有可执行程序 | C | 可选 `mlip-training.lasp`；与已实现的 SSW sampling wrapper 分开，当前未启用 |
| `MTPs` train dirs | cfg/训练集 → MTP potential | 接口不统一 | C/D | 可选 `mlip-training.mtp` |

所有框架适配器必须记录框架版本、入口 argv、环境/profile、数据集指纹、随机种子、超参数、模型指纹和日志引用。仓库不包含模型权重。

### 静态基准

| 代表路径 | 能力 | 等级 | 目标插件 |
|---|---|---:|---|
| `CHGnet/cal_rmse/cal_new.py` | 能量/力/应力 Pearson、MAE、RMSE，按原子数处理 | C | `mlip-benchmark` 薄适配器 |
| `M3Gnet/cal_test/cal.py` | 能量/力/应力误差与表格输出 | C | `mlip-benchmark` 薄适配器 |

这是最适合先包装的远端功能。首版统一输出 `metrics.json` 和长表 CSV，并保存能量归一化、应力单位、样本/分量计数和数据 split 定义。Excel 仅作为可选派生产物。

### MD、扩散与离子电导

| 代表路径 | 输入 → 输出 | 依赖/问题 | 等级 | 目标插件 |
|---|---|---|---:|---|
| `supply/**/pos2lammps.py` | POSCAR → CIF/LAMMPS data | ASE/pymatgen；已有位置参数 | B | `ionic-transport` converter |
| `supply/**/write_lammps.py` | 结构 → LAMMPS data | 复制 ASE 内部实现 | E | 直接调用 ASE API，不迁移 |
| `supply/Li8/sub.slurm` | 候选结构 + DeepMD → 400/600/800 K LAMMPS | 硬编码模型/路径，含清目录操作 | D | `ionic-transport` execution |
| `supply/code/Li{8,10}/get_MSD*.py` | `target.msd` → logD/扩散系数 | 固定跳过行、粒子数、体积、拟合窗 | D | `ionic-transport` analysis |
| `supply/code/Li10/.../get_sigma.py` | 三温扩散 → Arrhenius 300 K 电导 | 缺拟合诊断和单位 manifest | D | `ionic-transport` analysis |
| 各模型目录 MD SLURM | 模型 + 结构 → trajectory/MSD | 后端路径与科学参数耦合 | D | backend + `ionic-transport` |

MSD 时间轴、体积、Li 数量、扩散维度、拟合窗、平衡段和电荷载流子假设在统一实现前必须用原始输入或论文复核。

### 组分筛选与描述符

| 代表路径 | 能力 | 等级 | 目标插件 |
|---|---|---:|---|
| `supply` 组分 CSV 与批量目录 | 枚举候选并在多温度评估 | D | `composition-screening` |
| `supply/code/Li10/.../get_order.py` | 参考位点比较 → 组成/占位 Excel | C/D | `descriptor-analysis` |

未发现统一 top-k 排序器或结构化候选 manifest。首版排序必须显式列出约束、指标方向、缺失值策略和稳定 tie-break。

### 电化学电压

| 代表路径 | 能力 | 等级 | 目标插件 |
|---|---|---:|---|
| `Voltage/GGA+U/sub_Li{8,10,16}.slurm` | Li 含量序列结构的 VASP+U 优化 | D | `electrochemical-voltage` DFT backend |
| `Voltage/GGA+U/*/is_opt.sh` | 检查优化完成 | C | completion check |
| `Voltage/GGA+U/cp_Li8.sh` | 收集/移动结构 | D | artifact collector |

目录证据支持 Li8 的 8/6/4/2/0、Li10 的 10/8/6/4/2/0、Li16 的 16/12/8/4/0 序列，但尚未发现通用电压差公式或曲线生成器。首版只实现有测试公式的后处理，并要求显式金属 Li 参考能、每步 Li 数和能量单位。

### 不迁移或仅保留证据

- 上游项目源码副本（如完整 NequIP/LAMMPS 代码）；
- `write_lammps.py` 这类复制依赖库内部代码的文件；
- MTP 目录中名为 `calculate_ab_initio_ef`、实际使用 Farkas EAM/LAMMPS 的脚本，不得标为 DFT；
- 模型、训练大数据、轨迹、VASP 大输出和专有 POTCAR；
- 含 `rm`、`mv`、无限轮询或隐式提交的原始流程脚本不得直接由只读或回放模式调用。

## 工作区 claw 盘点

工作区发现 `../claw` 副本，包括 MACE 微调、VASP 静态标注、DIRECT 采样、LAMMPS/ASE/AIMD MD、扩散分析以及旧 skills。只定向读取了文本脚本和说明；没有读取或复制模型、轨迹、POTCAR 和大 VASP 结果。

| 路径（相对 `../claw`） | 能力与输入输出 | 复用判断 | 目标映射与风险 |
|---|---|---:|---|
| `direct_sampling/direct.py` | MAML DIRECT/Birch 代表性采样 CLI；POSCAR/CIF/vasprun/XDATCAR/ASE/LAMMPS → POSCAR、`manifest.csv`、PCA/FCS 图 | B | `pes-sampling`；需去绝对默认路径，禁止默认 `rmtree`，补版本/输入哈希与失败 manifest |
| `finetune_mace/collect.py` | 检查 VASP 完整性、SCF/离子收敛和 E/F/stress；固定 seed 划分 → train/valid/all extxyz、坏样本清单 | B | `dft-labeling` dataset assembly；改为无顶层副作用的 CLI，参数化阈值和是否取终态 |
| `finetune_mace/{finetune.sh,convert.sh}` | extxyz/foundation model → MACE 微调模型/LAMMPS 导出 | C | `mlip-training.mace` + model export；CUDA、路径、资源迁至 backend profile |
| `MD/ase_MD/ase_md_only_multi_calc.py` | MACE/CHGNet/M3GNet/EMT/LJ，NPT/NVT → trajectory/log/metadata | D | `ionic-transport` MD adapter；关键参数、device/dtype/seed 必须显式，禁止覆盖轨迹，增加恢复点 |
| `MD/lammps_MD/generate_md.py` | 结构/模型/多温度/seed/超胞 → MACE-LAMMPS data 与 NPT/NVT/MSD 输入 | B/C | `ionic-transport`；去除 `Li type 1`、framework `2 3` 和 `data.LYC` 假设 |
| `MD/lammps_MD/submit_mace_cond.sh` | 模块加载并调用绝对路径 LAMMPS | E | 只把资源意图迁入用户 backend profile，不复用脚本文本 |
| `MD/AIMD/generate_input.py` | pymatgen MITMDSet → NVT/NPT VASP 输入 | D | 已提炼为 `dft-labeling.vasp-prepare` 的显式 `aimd` contract；POTCAR 只由用户 `PMG_VASP_PSP_DIR` 运行时组装且禁止 collect/入库 |
| `diffusion_analysis/ionic_conductivity.py` | ASE/LAMMPS/VASP/MSD 多源；漂移、多时间原点、MSD、D、Nernst–Einstein、Arrhenius → CSV/JSON/HTML | C | `ionic-transport` 首选薄适配器；显式拟合窗、维度、体积、电荷、Haven ratio、replica 聚合与外推警告 |
| `skill/{direct-sampling,mace-sft,ase-md,diffusion_analysis}/SKILL.md` | 旧 Agent 操作说明 | D | 只提炼领域决策，重新编写九个 MLIPFlow Skills |

旧 skill 与代码存在明显漂移：DIRECT 文档调用不存在的 `run_direct_sampling.py` 和错误参数名；MACE 文档调用不存在的 `run_mace_sft.py` 并错误描述标签/硬件；ASE MD 文档调用不存在的 `run_ase_md_multi.py`，声称的多结构并行和全参数 CLI 也不在实际脚本中。因此这些文档不能作为接口事实来源。

科学注意项：扩散脚本默认全段拟合，`D = slope / 6` 只适用于三维各向同性；Nernst–Einstein 默认忽略离子相关；分段 Arrhenius 的每段点数和 300 K 远外推需要可信区间/警告；多个 seed 不应被当作独立温度点。首版必须在 manifest 中明确这些假设。

排除规模约为：DIRECT 目录 2.3 GB、MACE 微调 403 MB、扩散目录 707 MB、旧 skill 样例 104 MB。MLIPFlow 不复制其中的 `.venv`、权重、zip、轨迹、`vasprun.xml`、OUTCAR、POTCAR、WAVECAR、CHGCAR、备份或 `.DS_Store`。

## 跨代码库的关键缺口

1. 没有统一的 run/attempt/job/artifact 状态模型。
2. 没有机器验证的 project/plugin/result schema。
3. 训练、评估、调度和文件操作普遍耦合。
4. 随机种子、单位、split、拟合窗和结构/数据指纹经常缺失。
5. 没有统一模型 registry，也没有由 benchmark 证据驱动的 task-aware routing。
6. 没有严格的 replay 契约；很多“分析”脚本会删文件、生成新轨迹或提交作业。
7. 没有通用电压后处理和 production DeepMD 训练入口的充分证据。
8. 科研环境和 SLURM 资源硬编码，无法跨本地/集群复用。

## Phase 0 结论

可行路线是 **wrap before rewrite**：先保留原脚本为外部工具证据，通过 argv、工作目录、输入输出 schema 和回放收集器建立薄适配层；只有在接口和基准对齐后，才重写 SQS、数据装配、MSD/电导和电压等数值逻辑。模型优劣不得写死在核心代码，必须来自版本化 benchmark artifact。
