# 论文基准审计

最后更新：2026-08-17。

## 审计口径

本文件只审计论文中实际出现的六个精确模型配置：

- `deepmd-se_e2_a`
- `deepmd-se_e2_r`
- `deepmd-se_atten_v2`
- `deepmd-dpa2`
- `m3gnet`
- `chgnet`

`EXECUTABLE` 表示本地确定性执行能力；具体是 metric-only 还是 fresh model
inference 必须由 mode 和 provenance 区分，不能只看该状态。
`REPLAYABLE` 表示：既有论文证据可被只读导入、校验、归一化并记录来源；它不
是一次新计算。每一行的 `current status` 严格使用以下一个枚举值：
`EXECUTABLE`、`REPLAYABLE`、`MISSING_ADAPTER`、`MISSING_SOURCE`、
`EXTERNAL_DEPENDENCY` 或 `NOT_APPLICABLE`。

归一化输出固定为：
`benchmark/metrics.json`、`benchmark/benchmark_summary.csv`、
`benchmark/model_ranking.json` 和 `benchmark/provenance.json`。`execute`
模式只从成对标量重新计算 MAE、RMSE 和 Pearson 相关系数；`replay` 模式读取
JSON/CSV/XLSX 历史结果，且在 provenance 中写入 `model_execution=false`。
`evaluate-fresh` 才会加载显式模型，在 versioned prediction evidence 上记录
`model_execution=true`，再复用同一套 MAE/RMSE/Pearson 与 ranking 实现。该实现
及 fake/injected predictor contract 已完成，不代表六个真实 checkpoint 已完成
production-scale fresh numerical parity。

## Static PES

| model | metric | original source | MLIPFlow implementation | mode | required inputs | current status | remaining work |
|---|---|---|---|---|---|---|---|
| `deepmd-se_e2_a` | test energy RMSE / Pearson r；force RMSE / Pearson r | 指纹固定的历史 DeepMD source + metrics XLSX；测试集 13,024 个结构、2,673,726 个力分量 | DeepMD workbook parser | replay | XLSX、精确模型名、任务/场景；源码确认 energy per-atom，但 source 未声明 energy/force 物理单位 | REPLAY_VERIFIED | source SHA `9502e8…74e2`，output SHA `372466…97f`；当前重放值分别为 0.007190632639714127、0.998383122897441、0.1766770059878328、0.9564072093736768；不是 fresh parity |
| `deepmd-se_e2_r` | test energy RMSE / Pearson r；force RMSE / Pearson r | 指纹固定的历史 DeepMD source + metrics XLSX；同一测试划分 | 同上 | replay | 同上 | REPLAY_VERIFIED | output SHA `7f354f…b67e`；当前重放值分别为 0.01303367441338188、0.9946815122841708、0.2756778043936335、0.8901533484123095 |
| `deepmd-se_atten_v2` | test energy RMSE / Pearson r；force RMSE / Pearson r | 指纹固定的历史 DeepMD source + metrics XLSX；同一测试划分 | 同上 | replay | 同上 | REPLAY_VERIFIED | output SHA `9f39ae…a296`；当前重放值分别为 0.00790596295510387、0.9980456192199905、0.1856936082906573、0.9517273261398933 |
| `deepmd-dpa2` | test energy RMSE；force RMSE | SI 中表题为 Table 3 的汇总表；历史 archive 未发现 family-specific output/raw pairs | prepared JSON/CSV replay + 通用排序器 | replay | 紧凑转录、来源文档 SHA-256、单位和样本语义 | REPLAYABLE / MISSING_SOURCE | 可重建论文舍入值 6.63 meV/atom、151 meV/Å；对更丰富的历史 regression 仍为 `MISSING_SOURCE`，不能重算相关系数或误差分布 |
| `m3gnet` | test energy RMSE；force RMSE | SI Table 3；历史评测 source SHA `2289d1…fce`，但只读清单未发现 benchmark output/workbook/pairs/log | prepared evidence replay | replay | 紧凑转录、来源文档 SHA-256、单位 | REPLAYABLE / MISSING_SOURCE | 可重放论文值 15.84 meV/atom、212 meV/Å；脚本级历史数值 regression 明确为 `MISSING_SOURCE` |
| `chgnet` | test energy/force/stress MAE、RMSE、Pearson r | 指纹固定的历史 source + `efs_metrics_by_split.xlsx` | CHGNet workbook parser | replay | XLSX、任务/场景；源码确认 energy-per-atom 字段 | REPLAY_VERIFIED | source SHA `7f6e1d…df0`，output SHA `ddcf67…0c8`；test structures=9,753、force scalars=1,992,795、stress scalars=87,777。energy/force/stress 单位及 stress 顺序/符号均保留 source-unspecified；不从框架常识猜测 |
| 六个模型（精确 ID 如上） | energy/force/stress 的 MAE、RMSE、Pearson r | 用户提供的 reference/prediction pairs | `benchmark_wrapper.py execute` | execute（metric-only） | CSV/JSON/XLSX 成对数值、模型、task、scenario、split、target、unit | EXECUTABLE | 该路径不会加载模型或生成预测；真实新评测仍依赖外部模型、数据与预测生成器 |
| 六个模型（精确 ID 如上） | fresh energy/force/stress MAE、RMSE、Pearson r + ranking | canonical labeled dataset + explicit model artifact | `evaluate-fresh` + shared `mlipflow.science.model_runtime` | fresh inference | model/dataset fingerprints、显式 task/scenario/split、units/conventions；框架依赖按需安装 | EXECUTABLE | contract/injected-predictor regression 完成；尚未对六个真实 checkpoint 与 production dataset 完成 historical numerical parity，因此不能声称 external scientific validation complete |
| 三个 DeepMD 配置 | energy/force MAE、stress metrics | 已审计 DeepMD workbook 未报告这些量 | 无可导入数值 | 无 | 原始预测对或可信汇总 | NOT_APPLICABLE | 它们不是当前已连接论文表/workbook 的已报告指标；若新增论文主张，需先提供来源 |
| `deepmd-dpa2`、`m3gnet` | energy/force MAE/相关系数、stress metrics | 当前紧凑论文证据只含 energy/force RMSE | 无可导入数值 | 无 | 原始预测对或可信汇总 | NOT_APPLICABLE | 当前论文路由不使用这些缺失量，不能从模型名称推断支持 |

## Structural validation

| model | metric | original source | MLIPFlow implementation | mode | required inputs | current status | remaining work |
|---|---|---|---|---|---|---|---|
| 六个模型 | RDF | 论文中的结构比较，但当前已审计紧凑证据没有可分享的 RDF 数值表/原始曲线 | 无 | 无 | 参考/预测轨迹、物种对、bin/range 约定、原始脚本或可信曲线 | MISSING_SOURCE | 先取得最小可分享原始输出；之后再以薄适配器复用历史算法 |
| 六个模型 | bond distributions | 当前已审计证据没有可机器读取的键长分布 | 无 | 无 | 参考/预测结构、键定义、bin 约定、历史结果 | MISSING_SOURCE | 不得从论文图片或模型名称反推数值 |
| 六个模型 | coordination / local-structure metrics | 当前已审计证据没有可机器读取的配位/局域结构结果 | 无 | 无 | 邻居定义、截断、结构与历史结果 | MISSING_SOURCE | 获得源证据后才能判断还需哪一个最小 adapter |

## Transport validation and screening

| model | metric | original source | MLIPFlow implementation | mode | required inputs | current status | remaining work |
|---|---|---|---|---|---|---|---|
| `deepmd-se_atten_v2` | AIMD 对比的 conductivity MAE | SI Table S3，10 个成分 | manuscript evidence replay + registry/routing | replay | Table S3 紧凑转录、单位来源、文档与证据 SHA-256 | REPLAYABLE | MAE=0.09599531 mS/cm；这是已报告电导率的误差重放，不是新 MD |
| `deepmd-se_e2_a` | AIMD 对比的 conductivity MAE | SI Table S3 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.19909484 mS/cm |
| `deepmd-se_e2_r` | AIMD 对比的 conductivity MAE | SI Table S3 | 同上 | replay | 同上 | REPLAYABLE | MAE=1.1047 mS/cm |
| `deepmd-dpa2` | AIMD 对比的 conductivity MAE | SI Table S3 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.35609381 mS/cm |
| `m3gnet` | AIMD 对比的 conductivity MAE | SI Table S3 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.2332 mS/cm |
| `chgnet` | AIMD 对比的 conductivity MAE | SI Table S3 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.307299216 mS/cm |
| `deepmd-se_atten_v2` | MSD、两种 diffusion coefficient、Nernst–Einstein conductivity、Arrhenius activation energy | 历史 `target.msd`、`get_MSD_Li10.py`、`get_sigma.py` 与历史 stdout | `plugins/ionic-transport/adapter.py` 的只读 manuscript parity path | execute（post-process only） | 三个温度的既有 `target.msd`、采样 profile、体积、载流子数、Arrhenius convention | EXECUTABLE | 已能从既有轨迹产物重算；该行没有启动 MD。生产 MD 和新候选预测仍是外部工作 |
| 六个模型（取决于 calculator 支持） | tiny MD → trajectory → MSD/D/σ/Arrhenius handoff | 历史 `ase_md_only_multi_calc.py` 与 `ionic_conductivity.py` 接口 | `operation=md-smoke-and-analyze` 的有界、fresh-output、`shell=false` argv 编排 | execute（integration-smoke only） | 两个经审阅脚本、本地结构、本地模型或明确允许的内置 calculator、ASE/框架依赖、2–4 个温度和显式 seed | EXTERNAL_DEPENDENCY | 已用 MACE 0.3.15 + 外部 79,462,305-byte model/Li3YCl6 做真实 MLIP model-loading smoke，标准 Adapter 全闭环；该 model 不是此审计六个精确 benchmark 配置的 checkpoint，且每温度仅 10 步，因此不是模型 parity 或生产执行 |
| 其余五个模型 | 原始 MSD / diffusion / Arrhenius 曲线的脚本级 parity | 当前证据只有 Table S3 的最终电导率汇总 | 无原始轨迹可输入 parity path | 无 | 每模型原始轨迹或 MSD、历史脚本输出 | MISSING_SOURCE | Table S3 重放不能代替原始轨迹级数值验证 |
| `deepmd-se_atten_v2` | 247 个 Li10 大超胞候选的 300 K conductivity ranking / top-k | 历史候选清单与 conductivity 结果；候选 ID 的元素顺序与 metric ID 顺序不同 | `plugins/candidate-ranking/normalize_legacy.py` + `rank.py` + adapter 独立复核；原 evidence 的 `composition-screening` identity 保持不变 | replay / deterministic rank | 候选与结果的紧凑派生表、显式 `S/m`、排序方向、缺失策略 | REPLAYABLE | top-1 为 `Mn6_Fe3_Ni8_Cu4_Zn7`，18.420176685825222 S/m；候选生成和生产输运推理不在此重放中 |
| `deepmd-se_atten_v2` | top-candidate transport traceability | top-1 ID 与历史 `6_3_8_4_7` target.msd/stdout | composition ID 映射 + ionic parity record | replay + post-process | 候选映射、三个温度 MSD、历史输出 | REPLAYABLE | 已连接到同一候选的历史 MLIP 输运证据；这不是 AIMD/DFT 高保真验证，后者仍缺 |
| `deepmd-se_atten_v2` | top-candidate high-fidelity AIMD/DFT error 与 validation-ranking agreement | 当前没有与 `Mn6_Fe3_Ni8_Cu4_Zn7` 匹配的 AIMD trajectory、DFT reference 或 validation ranking table | 无可比较输入 | 无 | 精确结构 ID、AIMD/DFT reference、MLIP prediction、单位和 ranking rule | MISSING_SOURCE | 历史 MLIP post-processing parity 不能替代 predictive/high-fidelity parity；不得声称 top-k 高精度验证已完成 |

## Efficiency and generalization

| model | metric | original source | MLIPFlow implementation | mode | required inputs | current status | remaining work |
|---|---|---|---|---|---|---|---|
| `deepmd-se_atten_v2` | AIMD/MLIP wall-time acceleration | SI Tables S8–S10 | reproduction reduction | replay | 每个 prototype 的原始时间条目与来源哈希 | REPLAYABLE | 精确重算比值为 115.59219357117627、113.28807549327995、173.07966457023062；不是新性能测试 |
| 其余五个模型 | 相对 AIMD 的同口径 speedup | 当前连接的论文证据未对它们作同口径时序比较 | 无 | 无 | 同硬件/结构/步数定义的 timing evidence | NOT_APPLICABLE | 不可把 `deepmd-se_atten_v2` 的 speedup 外推给其他模型 |
| `deepmd-se_atten_v2` | large-supercell screening | 228 原子 Li10 SQS 候选空间的 247 项历史筛选结果 | screening replay | replay | 紧凑 top-k/摘要、候选数量、超胞语义、来源哈希 | REPLAYABLE | 可重放排序逻辑；未重新生成 247 个 SQS，也未运行 247 次生产 MD |
| 六个模型 / transfer claim | unseen Li24M12(PS4)16 / cross-prototype numerical validation | 论文文字主张可被记录，但当前没有可分享的逐结构预测/参考/误差表 | 仅 claim-level provenance record | claim-only | 原始 787-config（或对应最终版本）参考/预测摘要及单位、划分、来源哈希 | MISSING_SOURCE | claim 被“表示”不等于数值验证；在取得最小派生误差表前不得称为 parity |

## Electrochemical prediction

| model | metric | original source | MLIPFlow implementation | mode | required inputs | current status | remaining work |
|---|---|---|---|---|---|---|---|
| `deepmd-se_e2_a` | voltage MAE vs DFT | SI Table S11，18 个相邻 Li 阶段 | 标准 Adapter 的 `operation=replay-si-table-s11`，以 `shell=false` argv 调 bundled `manuscript_replay.py` | replay | 严格 15 列 long CSV、V 单位、来源文档 SHA-256 | REPLAYABLE | MAE=2.61944444444444 V；从已报告电压重算误差，不从总能重算电压 |
| `deepmd-se_e2_r` | voltage MAE vs DFT | SI Table S11 | 同上 | replay | 同上 | REPLAYABLE | MAE=1.19944444444444 V |
| `deepmd-se_atten_v2` | voltage MAE vs DFT | SI Table S11 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.666666666666667 V |
| `deepmd-dpa2` | voltage MAE vs DFT | SI Table S11 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.59 V |
| `m3gnet` | voltage MAE vs DFT | SI Table S11 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.536111111111111 V |
| `chgnet` | voltage MAE / RMSE / max absolute error vs DFT | SI Table S11 | 同上 | replay | 同上 | REPLAYABLE | MAE=0.182777777777778 V、RMSE=0.23313086453749532 V、max=0.56 V、N=18 |
| model-independent post-process | adjacent average Li intercalation voltage | 已完成并标为 `OK` 的 Li-content total-energy sequence | `plugins/electrochemical-voltage/adapter.py` + `mlipflow.science.voltage` | execute（post-process only） | 严格递增 Li 序列、eV 总能、formula-unit 数、Li reference energy | EXECUTABLE | 尚未用论文原始总能序列作 parity；不会执行 DFT 或 MLIP |

## 结论边界

当前重放证据足以让路由从指标中选择：static PES → `deepmd-dpa2`、ionic
transport → `deepmd-se_atten_v2`、voltage → `chgnet`。这些选择来自
versioned evidence、方向和 policy，不是核心代码中的模型品牌条件分支。

本审计不支持以下说法：六个真实模型 checkpoint 已完成历史同口径重新推理、RDF/局域结构已重算、
unseen transfer 已获数值验证、生产 MD/DFT/训练已运行，或任一重放结果是新科学
计算。
