# 科学验证记录

最后更新：2026-08-11。

## 状态口径

本表只使用规范允许的状态：`PASS`、`FAIL`、`REPLAY_VERIFIED`、
`EXTERNAL_VALIDATION_PENDING`、`NOT_TESTABLE_WITH_AVAILABLE_DATA`。

- `PASS`：同一真实 fixture 已分别经过历史实现和 MLIPFlow 路径，并实际比较数值。
- `REPLAY_VERIFIED`：既有可信结果被只读导入，转录、归一化、透明 reduction 和来源
  哈希已校验；没有重新运行产生预测的科学程序。
- `EXTERNAL_VALIDATION_PENDING`：代码接口存在，但仍需要科学依赖、模型、数据、引擎
  或站点环境才能作真实比较。
- `NOT_TESTABLE_WITH_AVAILABLE_DATA`：当前没有足够原始证据来建立比较。

通用 CLI、schema、manifest 或 synthetic unit test 通过，均不单独构成科学 `PASS`。

## 对照表

| plugin | original implementation | fixture | quantity compared | original result | MLIPFlow result | tolerance | status | notes |
|---|---|---|---|---|---|---|---|---|
| `high-entropy-structure` | 历史 Li10M7P8S32 icet/SQS 脚本（未暴露 seed） | 真实只读 57 原子 Li10Zn7P8S32 prototype；ASE 3.28.0、icet 3.2；1×1×1、1 candidate、100 steps、seed=23、cutoffs=[7,5] Å | wrapper 执行、成分/原子数、两次 deterministic output；另验证论文 2×2×1/228 原子 manifest 语义 | 历史输出无确定 seed，不能建立历史 structure parity | 两次真实 wrapper smoke 的 structure/manifest 逐字节一致；57 原子、7 金属位，Zn3Fe1Cu1Ni1Mn1 正确 | 新执行两次 byte exact；没有历史结构比较容差 | EXTERNAL_VALIDATION_PENDING | `reports/sqs_local_integration_smoke.json` 记录 `LOCAL_INTEGRATION_SMOKE_PASS`；它仅证明 bounded wrapper/seed/write/manifest 闭环，不是 production SQS，也不是历史结构 parity；外部 prototype 只固定摘要、未复制 |
| `pes-sampling` — DIRECT | MAML DIRECT `direct.py` | 当前无可分享的历史输入/选择 manifest 对 | 被选结构 ID/索引、cluster assignment | 无可比较结果 | adapter 仅完成 argv/安全边界 | 未定义 | NOT_TESTABLE_WITH_AVAILABLE_DATA | source CLI 没有 seed，且没有执行真实 DIRECT fixture |
| `pes-sampling` — LASP/SSW historical normalization | 历史 LASP `allstr.arc`/`best.arc`/`md.arc` 与 `single.py`/`single_extract` 选择链 | 两个只读代表目录的小型 LASP 文本 archive/配置被复制到临时本地目录；审计后不保留原始结构 | ARC frame 解析、能量过滤、accepted-order stride、导出顺序、输入参数、源 SHA-256 与 Adapter completion | 初始链 6 frame→`energy<=0` 接受 5→stride 1 选 5，`best.arc` 为 2；重采样链 66 frame→过滤后 25→每 3 个 accepted frame 取 1，`best.arc` 为 1、`md.arc` 为 17 | 两套临时副本均实际运行 `lasp-ssw-normalize-replay`，wrapper returncode=0、Adapter check=`OK`；初始计数 generated/accepted/selected/best/md=6/5/5/2/0，重采样为 66/25/9/1/17；参数和源哈希核对一致 | source SHA-256、frame count/order、energy filter、accepted-order stride 与选中结构 bytes exact | REPLAY_VERIFIED | `reports/lasp_ssw_historical_replay.json` 记录 `HISTORICAL_POSTPROCESS_REPLAY_PASS`；只验证已有 archive 后处理，不执行 LASP、不验证 SSW 数值；seed=`HISTORICAL_PARAMETER_UNKNOWN` |
| `pes-sampling` — LASP/SSW execute contract | 用户自备 LASP executable + ARC/`lasp.in`/auxiliary inputs | 有界 fake executable local fixture | 显式 `lasp_version`、输入 staging、direct 或显式可指纹化 `mpirun`/`mpiexec` 路径加 `-np` argv、`shell=False`、return code、archive normalize、manifest/check/collect | 没有真实授权 LASP 结果 | fake executable 的 local plan→execute→check→collect contract smoke 通过 | contract exact；无科学数值容差 | EXTERNAL_VALIDATION_PENDING | adapter/contract 已实现，但 fake program 不是 LASP；真实 LASP、scientific numerical parity、scheduler/HPC 均未验证 |
| `dft-labeling` | 历史 VASP/AIMD 标注脚本 | 当前无小型、已完成、可分享的同输入 DFT fixture | energy/force/stress、收敛、dataset shape | 无可比较结果 | 严格 result contract 已实现，但未运行 DFT | 未定义 | NOT_TESTABLE_WITH_AVAILABLE_DATA | 未运行 VASP；没有 POTCAR/OUTCAR/vasprun.xml 被纳入仓库 |
| `mlip-training` — DeepMD | 历史 `deepmd-se_atten_v2` 最终 `input.json` | SHA-256 固定的只读最终配置 | fresh-train effective config/result extraction | descriptor=`se_atten_v2`；descriptor/fitting/training seeds=1/1/10；float64；300,000 steps；train/validation systems=747/747，两个 system-ID 列表各有摘要；没有标准 training result | `reports/training_source_audit_summary.json` 保存相同字段、配置摘要与缺失 contract | exact config-field/source-digest audit；未执行数值比较 | EXTERNAL_VALIDATION_PENDING | 仅 source/config audit，未加载 DeepMD、未训练、未建立 numerical parity；没有声称 DeepMD fine-tune、freeze/test 或 checkpoint parity |
| `mlip-training` — MACE scratch | 历史 SLURM shell source | 只读 source audit | 有效 trainer argv、seed、输出 contract | `--E0s=average` 后缺续行；后续 flags/seed 不属于实际命令 | auditor 检出 missing continuation、absolute paths、missing effective seed | exact source parse | EXTERNAL_VALIDATION_PENDING | 源审计不是训练 parity；历史命令本身需要修复后才能作 tiny smoke/结果提取 |
| `mlip-training` — MACE fine-tune | 历史 foundation-model fine-tune shell source | 只读 source audit | operation、seed/device/precision 与输出 contract | seed=3、device=cpu、float64、6 epochs；无标准 result manifest | auditor 能提取这些字段并报告缺失 contract | exact source parse | EXTERNAL_VALIDATION_PENDING | 未加载 foundation model、未 fine-tune |
| `mlip-training` — CHGNet | 历史 Python trainer source | 只读 source audit | split、seed、device、输出 contract | train/val=0.8/0.2、CPU、5 epochs；无显式 seed，存在 top-level execution | auditor 检出 path/seed/top-level/result-contract 问题 | exact source parse | EXTERNAL_VALIDATION_PENDING | 未训练；静态审计不等于 CLI/config parity 已闭合 |
| `mlip-training` — M3GNet | 历史 Python trainer source | 只读 source audit | split、seed、API family、输出 contract | [0.8,0.05,0.15]、random_state=42、shuffle=false；混用 M3GNet/MatGL API | auditor 提取 split/seed 并检出 mixed API/top-level/result-contract 问题 | exact source parse | EXTERNAL_VALIDATION_PENDING | 未训练；需固定框架版本和可运行入口 |
| `mlip-benchmark` — `deepmd-se_e2_a` | 历史 DeepMD metrics XLSX | 原始只读 workbook | test energy/force RMSE 与 Pearson r | 0.007190632639714127 eV/atom、0.998383122897441；0.1766770059878328 eV/Å、0.9564072093736768 | 归一化记录逐值相同，N=13,024 / 2,673,726 | workbook scalar exact | REPLAY_VERIFIED | 只重放 workbook；没有运行模型或重新生成预测 |
| `mlip-benchmark` — `deepmd-se_e2_r` | 历史 DeepMD metrics XLSX | 原始只读 workbook | 同上 | 0.01303367441338188、0.9946815122841708；0.2756778043936335、0.8901533484123095 | 归一化记录逐值相同 | workbook scalar exact | REPLAY_VERIFIED | 同上 |
| `mlip-benchmark` — `deepmd-se_atten_v2` | 历史 DeepMD metrics XLSX | 原始只读 workbook | 同上 | 0.00790596295510387、0.9980456192199905；0.1856936082906573、0.9517273261398933 | 归一化记录逐值相同 | workbook scalar exact | REPLAY_VERIFIED | 同上 |
| `mlip-benchmark` — `deepmd-dpa2` | SI Table 3 | 紧凑、SHA-256 固定的表格转录 | test energy/force RMSE | 6.63 meV/atom；151 meV/Å | replay/routing 使用相同值 | exact transcription and deterministic reduction | REPLAY_VERIFIED | 没有原始预测对，因此不是脚本级 metric parity |
| `mlip-benchmark` — `m3gnet` | SI Table 3；历史计算脚本 | 紧凑表格转录 | test energy/force RMSE | 15.84 meV/atom；212 meV/Å | replay 使用相同值 | exact transcription | REPLAY_VERIFIED | 真实 output workbook 未取得；不能称历史脚本 parity |
| `mlip-benchmark` — `chgnet` | 历史 E/F/S metrics XLSX | 原始只读 workbook | test energy/force/stress MAE、RMSE、Pearson r | energy RMSE=0.0080861044340676 eV/atom；force RMSE=0.1889819035175336；stress RMSE=12.46909130171059 | 归一化记录逐值相同 | workbook scalar exact | REPLAY_VERIFIED | source 没可靠声明 force/stress 单位；MLIPFlow 明确写 `source-unit-unspecified` 而非猜测 |
| `mlip-benchmark` — execute-from-pairs | 历史模型预测/参考对 | 当前只有 synthetic pair test，没有全部六模型真实 pair fixture | MAE、RMSE、Pearson r | 无真实统一 fixture | metric-only execute 路径可运行 | 未建立科学容差 | NOT_TESTABLE_WITH_AVAILABLE_DATA | generic 测试证明算法接口，不证明论文模型的新评测 |
| `ionic-transport` — MACE historical parity | 历史 MACE Li10 `target.msd`、script 与 stdout | dataset `11221`，400/600/800 K | 两种 D 与 conductivity | 400 K linear-fit D=9.261443310293172e-11 m²/s，σ=2.762778334022906 S/m | legacy mode 与历史 stdout 数值相同 | rtol=1e-12 | PASS | 历史脚本错误使用 N=7；此 PASS 只证明 bug-compatible parity，绝不把 N=7 当 ground truth。corrected N=10 保持 D 不变、σ 精确乘 10/7，并标为预期分歧 |
| `ionic-transport` — DeepMD top candidate | 历史 `6_3_8_4_7` 的三个 `target.msd`、`get_MSD_Li10.py`、`get_sigma.py` 与 stdout | 400/600/800 K，Li40 大超胞 carrier convention | D、σ、legacy Arrhenius 300 K extrapolation | 400 K linear-fit D=9.720444190727505e-10 m²/s；mean-MSD Arrhenius σ(300 K)=18.420176685825222 S/m | adapter parity path 重现相同值 | σ rtol=1e-14；其它记录 exact/rtol=1e-12 | PASS | 输入均只读；没有运行 MD。`6_3_8_4_7` 对应筛选 top-1 `Mn6_Fe3_Ni8_Cu4_Zn7` |
| `ionic-transport` — MD handoff | 历史 `ase_md_only_multi_calc.py` 与 `ionic_conductivity.py` | Li3YCl6；MACE 0.3.15 + 外部 model；ASE 3.28.0、torch 2.10.0、CPU float64、seed=29；400/600/800 K 各 10 steps（1 fs、每 2 步写帧），无 NPT/NVT | model load → 3 trajectories/metadata → NumPy multi-origin MSD/D/σ → 3 点 Arrhenius → manifest/check/collect | 两个真实历史源均实际执行，rc=0、3 runs/0 failed；没有独立科学 baseline | 标准 Adapter：plan READY、subprocess rc=0、check OK、collect OK，`scientific_use=integration-smoke-only` | artifact existence/SHA 与 contract exact；未定义科学数值容差 | EXTERNAL_VALIDATION_PENDING | `reports/ionic_md_local_integration_smoke.json` 固定代码、输入、运行时和输出摘要；真实 MLIP 已加载，但每温度仅 0.01 ps，没有独立数值对照、采样收敛性、模型质量或 production 主张，故不是科学 parity |
| `composition-screening` | 历史 Li10 candidate files + conductivity files | 247 个候选的只读结果；元素顺序 `Zn,Fe,Cu,Ni,Mn` 与 metric ID 顺序不同 | candidate identity、247/247 coverage、deterministic top-k | top-1=`Mn6_Fe3_Ni8_Cu4_Zn7`，18.420176685825222 S/m | normalizer + screen 得到相同 top-1 与稳定排序 | ID exact；metric scalar exact | PASS | 没有重跑 247 个 SQS/MD；这里只比较历史结果规范化与排序 |
| `composition-screening` — top validation link | 历史筛选结果与 `6_3_8_4_7` transport artifacts | top-1 composition mapping | top candidate ID 与输运 evidence linkage | 两者指向同一 composition | provenance 可追踪到 top-1 的 transport parity | ID exact + evidence SHA-256 | REPLAY_VERIFIED | 仅连接 MLIP 输运证据；没有 AIMD/DFT 高保真 top-candidate 对照 |
| `composition-screening` — high-fidelity validation | 期望的 top candidate AIMD/DFT comparison | 当前无可分享的对应数值表 | ranking agreement / error vs high fidelity | 无 | 无 | 未定义 | EXTERNAL_VALIDATION_PENDING | 这是当前 top-candidate 科学主张的明确缺口 |
| `electrochemical-voltage` — SI replay | SI Table S11 | 9 个体系×2 个阶段=18 行 strict long CSV | 每模型 signed/absolute error、MAE/RMSE/max、ranking | CHGNet MAE=0.182777777777778 V、RMSE=0.23313086453749532 V、max=0.56 V | 标准 Adapter 的 `replay-si-table-s11` 重算所有 18×6 个误差并收集三件套；CHGNet 排名第一 | scalar abs≤1e-15；row set exact | REPLAY_VERIFIED | `mode=replay`、`model_execution=false`、`dft_execution=false`、`recomputed_from_total_energies=false`；不是总能重算 |
| `electrochemical-voltage` — total-energy formula | 历史 total-energy voltage calculation | 当前未取得同一论文总能序列 | adjacent average voltage | 无可比较原始序列 | adapter 可从显式 eV total-energy series 计算 | 未定义 | NOT_TESTABLE_WITH_AVAILABLE_DATA | synthetic formula test 不是论文 parity；SI replay 从已报告 voltage 开始 |
| manuscript routing | SI Tables 3/S3/S11 + project policy | 六模型 registry、27 条 benchmark records | task-specific selected model | static=`deepmd-dpa2`；transport=`deepmd-se_atten_v2`；voltage=`chgnet` | 路由得到相同三项，并保存 policy/evidence/routing digests | selected IDs exact | REPLAY_VERIFIED | registry 没有 `recommended_tasks` 提示；赢家由指标、方向和权重产生 |
| timing reduction | SI Tables S8–S10 | prototype I/II/III timing entries | AIMD time / MLIP time | 论文近似 116、114、173 | 精确重算 115.59219357117627、113.28807549327995、173.07966457023062 | exact from source seconds | REPLAY_VERIFIED | 这是历史 timing evidence reduction，不是当前机器的性能 benchmark |
| unseen / transfer validation | 论文中的 Li24M12(PS4)16 / unseen claim | 仅 claim-level provenance；无逐结构 reference/prediction/error table | quantitative error / ranking preservation | 原始数值不可用 | claim 可被机器读取地表示，但不能重算 | 未定义 | NOT_TESTABLE_WITH_AVAILABLE_DATA | “已表示”不等于“已验证”；不得纳入 numerical parity 结论 |
| RDF / bond / coordination validation | 论文结构比较 | 当前无可分享机器可读 source fixture | curve/distribution/local structure parity | 无 | 无 | 未定义 | NOT_TESTABLE_WITH_AVAILABLE_DATA | 需最小原始曲线或轨迹及相同分析约定 |

## 已建立与未建立的科学结论

已建立的真实脚本数值 parity 只有：历史 MACE/DeepMD 输运后处理，以及 247 项
历史筛选结果的 ID/数值规范化与排序。benchmark、voltage、timing 和路由属于
`REPLAY_VERIFIED`。LASP/SSW 的两套已有 archive 已由本地 wrapper 实际重放并经
Adapter check=`OK`，其 sanitized 记录为 `reports/lasp_ssw_historical_replay.json`；
该历史后处理选择链达到
`REPLAY_VERIFIED`，但真实 LASP/SSW 数值没有被验证。SQS、DIRECT、DFT、训练、
论文模型 MD 数值、top-candidate AIMD/DFT 高保真对照和 unseen quantitative
validation 均没有被科学验证；fake LASP executable 与真实 MACE model-loading bounded
smoke 只属于软件契约/集成 handoff 证据。
