# 论文工作流复现

最后更新：2026-08-11。

## 复现定义

`examples/high_entropy_sulfide_reproduction/` 是真实论文证据的紧凑、只读
`evidence-replay`，不是生产计算复跑。它不会执行 MLIP、MD、AIMD、DFT、训练、
结构优化、SQS 或 LASP/SSW，也不会访问网络或 scheduler。

`REPLAY_VERIFIED` 在这里表示：

1. 小型转录/派生证据的 schema、单位、行数与 SHA-256 已校验；
2. MAE、speedup、排序和 routing 等透明 reduction 被重新计算；
3. 每个决定保留 source document/artifact → compact evidence → normalized metric →
   policy → route 的来源链；
4. 相同证据在不同本地目录产生相同结果，篡改会使检查失败。

它不表示底层模型预测已独立验证，也不替代原始轨迹、训练集或生产计算。

## 运行方式

从 example 目录执行：

```bash
PYTHONPATH=../../src python3 reproduce.py --check
```

`--check` 只在内存/临时位置重建并核对已提交 artifacts；不会改动证据。
维护者可用 `--write` 显式重生成已提交的 normalized outputs/reports，之后必须重新
审计 diff 与测试。

固定输出包括：

- `benchmark/metrics.json`
- `benchmark/benchmark_summary.csv`
- `benchmark/model_ranking.json`
- `benchmark/provenance.json`
- `reports/manuscript_reproduction_summary.json`
- `reports/manuscript_reproduction_summary.md`
- 顶层 `reports/manuscript_reproduction_summary.json/.md` 索引

自动验收位于 `tests/test_manuscript_reproduction.py`，不运行生产科学程序。

## 证据包

原始 Word 文档不进入 example；只记录 basename、size、read-only 标记和摘要：

| source | SHA-256 | repository inclusion |
|---|---|---|
| main manuscript | `22bfc6c405bc972bca86c141a14239ad4a91718b086e9418784a51ab0aa67e9b` | 不包含原文 |
| supporting information | `0e421d4236d8c9389eddd4e61f9503ada6ff0fd07f70f5bd1c32eea35214e732` | 不包含原文 |

紧凑证据及其用途：

| evidence | purpose | evidence level |
|---|---|---|
| `table_s2_model_implementation.csv` | 六模型/训练数据与模式 inventory | source-document transcription |
| `table_3_pes_rmse.csv` | static PES test energy/force RMSE | source-document transcription |
| `table_s3_ionic_conductivity.csv` | 10 个成分的 AIMD/六模型 conductivity | source-document transcription |
| `table_s8_runtime.json`、`table_s9_runtime.json`、`table_s10_runtime.json` | prototype I–III 的 AIMD/MLIP timing | source-document transcription |
| `table_s11_voltage.csv`、`table_s11_voltage_long.csv` | 9 个体系×2 stage 的 DFT/六模型 voltage | source-document transcription |
| `large_supercell_screening_top10.json` | 247/247 历史 Li10 大超胞候选的 deterministic top-10 | pinned local result artifact；REPLAY_VERIFIED |
| `top_candidate_transport_link.json` | top-1 ID 到历史 `6_3_8_4_7` MSD/stdout parity 的来源连接 | read-only historical post-process parity；高保真例外见下文 |
| `unseen_transfer_claim.json` | 论文对 unseen Li24M12(PS4)16 与 composition transfer 的文字主张 | claim-level document evidence；非 numerical evidence |
| `transcription_provenance.json` | 每个转录文件、source location 和 digest | provenance index |
| `benchmark_records.json` | 从上述表透明派生的 27 条 routing records | prepared replay input |

仓库不包含原始文档、完整 247 候选 manifest、完整轨迹、模型、训练数据或 DFT
输出。大文件只以 portable locator、size 和 SHA-256 指向用户可提供的外部 artifact。

训练只读审计另见 `reports/training_source_audit_summary.json`：它固定 5 份历史
入口/最终配置，包括 `deepmd-se_atten_v2` 的最终配置摘要（seeds=1/1/10、
float64、300,000 steps、747/747 systems）。该报告明确
`training_executed=false`、`source_code_executed=false`、
`numerical_parity_established=false`，不属于论文结果的新训练复现。

`reports/sqs_local_integration_smoke.json` 另记录一次真实 ASE/icet bounded smoke：
外部只读 57 原子 prototype、seed=23、1 candidate、100 steps；两次独立输出结构与
manifest 逐字节一致且成分正确。它不由 reproduction example 执行，外部 prototype
也未复制，因此只证明真实 wrapper/seed/write handoff，不是 production SQS 或历史
structure parity。

`reports/ionic_md_local_integration_smoke.json` 另记录
`operation=md-smoke-and-analyze` 的真实 MLIP bounded handoff：ASE 3.28.0、MACE
0.3.15 + 外部 model、Li3YCl6 在 400/600/800 K 各 10 steps。三个
trajectory/metadata 经当时外部审阅的 `ionic_conductivity.py` 生成 MSD/D/σ/Arrhenius，标准
Adapter 的 plan→execute→check→collect 全部成功；外部 79,462,305-byte model 只固定
SHA-256，没有复制入仓库。reproduction example 不调用该 smoke；0.01 ps/温度的轨迹
也没有独立科学 baseline 或收敛性，所以只证明 integration handoff，科学 parity 状态
仍为 `EXTERNAL_VALIDATION_PENDING`。

该报告固定的是历史 smoke 当时的 adapter/source digest，不能冒充当前代码重跑。当前 `ionic-transport` 已把 runner 正式打包、把核心关系收敛到 `mlipflow.science.transport`，并增加 analysis manifest 与 checker 独立重算；本轮 deterministic local regression 不改变旧 smoke 的科学等级。

## LASP/SSW 对历史数据生成的贡献与边界

独立于 reproduction example 的只读源码/小文本审计表明，LASP/SSW 曾是历史结构与
后续单点数据生成链的上游采样环节：SSW 产生 `allstr.arc`，历史脚本排除正能量结构、
按接受顺序抽样并生成后续单点输入；`best.arc` 则提供候选结构。这个结论补充了数据
lineage，但不证明仓库中的全部论文训练数据都由这一条链生成。

代表性初始目录 `Li10M7P8S32/run/11113` 的源哈希已核对：`allstr.arc` 含 6 frame，
`energy<=0` 接受 5，accepted-order stride 1 选中 5，`best.arc` 含 2 frame；临时本地
副本实际运行 `lasp-ssw-normalize-replay` 后得到 generated/accepted/selected/best/md=
6/5/5/2/0，wrapper rc=0、Adapter check=`OK`。代表性重采样目录的
`allstr.arc`/`best.arc`/`md.arc`/`lasp.in` 也被只读复制到临时目录并实际 replay，得到
66/25/9/1/17，wrapper rc=0、Adapter check=`OK`；前 66/25/9 与历史正能量过滤、
25 个重编号输入及每 3 个 accepted frame 取 1 的 9 个单点目录一致。参数证据还区分了初始
`SSW.SSWsteps=4`、`SSW.Temp=200`，重采样 `SSW.SSWsteps=6`，以及后续 MD 的
500→800 K 温度段。历史随机种子未记录，严格标为
`HISTORICAL_PARAMETER_UNKNOWN`。

因此这里只把 archive 解析、能量过滤、accepted-order stride、计数、顺序与 provenance
记为 **历史 post-processing `REPLAY_VERIFIED`**。sanitized 报告
`reports/lasp_ssw_historical_replay.json` 的状态为
`HISTORICAL_POSTPROCESS_REPLAY_PASS`；它不含原始结构、远端实际路径或凭据，临时副本已
在审计后删除。`reproduce.py --check` 不执行 LASP；
另行通过的 fake executable local contract smoke 也只证明 `shell=False` staging、直接或
显式可指纹化 `mpirun`/`mpiexec` 路径加 `-np` argv、结果采集和失败边界。真实授权 LASP 未执行，SSW
科学数值 parity、scheduler/HPC 均仍 pending。LASP/SSW 采样、LASP 势训练、以及
DFT→LASP `TrainStr`/`TrainFor` 导出是三个独立能力，不能相互代替完成声明。

## 从证据到任务路由

六个精确模型均登记在 registry：`deepmd-se_e2_a`、`deepmd-se_e2_r`、
`deepmd-se_atten_v2`、`deepmd-dpa2`、`m3gnet`、`chgnet`。registry 中没有
`recommended_tasks`；脚本也没有按 task 写死赢家。

| task | evidence-derived winner | criterion |
|---|---|---|
| static PES | `deepmd-dpa2` | test energy RMSE=6.63 meV/atom、force RMSE=151 meV/Å；等权 minimize |
| ionic transport | `deepmd-se_atten_v2` | Table S3 conductivity MAE=0.09599531 mS/cm；minimize |
| electrochemical voltage | `chgnet` | Table S11 voltage MAE=0.182777777777778 V；minimize |

由此可重建论文的重要工作流逻辑：static 最优模型不必等于 transport 最优模型，
transport 最优模型也不必等于 voltage 最优模型。每条 route 保存 registry、policy、
prepared benchmark、source table 和 source document 的摘要链。

## 可重建的 timing 结论

| prototype | exact AIMD/MLIP ratio | manuscript approximation |
|---|---:|---:|
| I | 115.59219357117627 | 116 |
| II | 113.28807549327995 | 114 |
| III | 173.07966457023062 | 173 |

这些比值由历史秒数重新相除；不是当前机器的新性能测试，也不能外推给另外五个
模型。

## 大超胞筛选与 top candidate

`large_supercell_screening_top10.json` 固定了 11 个候选源文件、11 个 metric 源文件
及三个完整外部 result artifact 的摘要。校验结论为：

- candidate_count=247、evaluated_count=247、missing=0、unique=247；
- 每个候选含 28 个金属位，源顺序 `Zn,Fe,Cu,Ni,Mn` 被显式转换为 canonical
  `Mn,Fe,Ni,Cu,Zn`；
- metric=`ionic_conductivity_300k_s_per_m`，unit=`S/m`，maximize，top_k=10，
  tie-break=`candidate_id-ascending`；
- top-1=`Mn6_Fe3_Ni8_Cu4_Zn7`，18.420176685825222 S/m。

该 artifact 重放已经完成的 ranking。它没有重新生成 247 个 SQS，也没有执行
247 次模型推理或 MD。

现行实现为 `candidate-ranking`。artifact 中原 `composition-screening` plugin ID、
旧 locator 与 SHA-256 作为历史 provenance 原样保留；`reproduce.py` 固定校验这些
legacy 记录，并用现行 `rank.py` 对紧凑 top-10 再做兼容排序检查。

top-1 的 canonical composition 与历史目录 token `6_3_8_4_7` 形成精确身份映射。
三个既有 `target.msd` 经历史 sampling/Arrhenius convention 后，对历史 stdout 达到
`EXACT_NUMERICAL_PARITY`，300 K mean-MSD conductivity 同样为
18.420176685825222 S/m。这证明筛选值与同一历史 MLIP transport artifact 的
post-processing 链接正确。

它**不**是 top candidate 的 AIMD/DFT 高保真验证：当前没有匹配的 AIMD trajectory
或 DFT reference artifact，所以 high-fidelity 状态保持
`EXTERNAL_VALIDATION_PENDING`。

## Unseen / transfer 边界

`unseen_transfer_claim.json` 将论文文字中的下列内容结构化并固定来源：

- unseen system：Li24M12(PS4)16（expanded formula Li6M3(PS4)4）；
- 论文称该体系不在训练集中，并报告 787 个 AIMD configurations；
- 论文定性描述 energy accuracy 和 force errors；
- 论文定性描述三个掺杂成分的 DeepMD/AIMD room-temperature conductivity
  comparison 与相对排序。

当前没有逐 configuration reference/prediction energy/force，也没有三个成分 ID 及
成对 conductivity 值。故该项只满足“在验收中有来源地表示”，数值 parity 为
`NOT_TESTABLE_WITH_AVAILABLE_DATA`，整体仍是 `EXTERNAL_VALIDATION_PENDING`。

## 论文级自动验收覆盖

| required check | result | precise boundary |
|---|---|---|
| 1. multiple MLIPs registered | REPLAY_VERIFIED | 六个精确模型，且 registry 无推荐 task hint |
| 2. static and transport rankings differ | REPLAY_VERIFIED | 分别选择 DPA-2 与 se_atten_v2；由证据/策略产生 |
| 3. transport route selects se_atten_v2 | REPLAY_VERIFIED | Table S3 MAE 与完整 decision provenance |
| 4. voltage route selects CHGNet | REPLAY_VERIFIED | Table S11 18-row errors 与完整 decision provenance |
| 5. large-supercell screening replay | REPLAY_VERIFIED | 247/247、top-10、单位、规则、输入/实现 digests 均校验；不运行筛选计算 |
| 6. top candidate validation evidence connected | BOUNDED_CONNECTION | top-1 ↔ `6_3_8_4_7` historical MLIP post-process parity 已连接；AIMD/DFT 高保真仍 `EXTERNAL_VALIDATION_PENDING` |
| 7. unseen/transfer represented | CLAIM_LEVEL_ONLY | claim/source digest 已连接；numerical parity=`NOT_TESTABLE_WITH_AVAILABLE_DATA` |
| 8. every decision has provenance | REPLAY_VERIFIED | evidence、policy、registry、routing 和 source digests 均存在，证据篡改被拒绝 |

因此，自动验收可以通过其明示的 evidence-replay contract；它没有把第 6、7 项的
外部科学缺口提升为验证完成。

## 可复现结论与不可复现结论

当前可以从真实证据重建：

- 六模型 static PES 排名以及 DPA-2 的该 policy 最优结论；
- 六模型 AIMD-conductivity 误差排名以及 se_atten_v2 的 transport route；
- 六模型 voltage 误差排名以及 CHGNet route；
- prototype I–III 的历史 timing reductions；
- 247 个大超胞候选的历史 top-10、top-1 与同一历史 transport result 的连接；
- LASP/SSW 历史 archive 的解析、过滤、抽样计数与数据生成 provenance；
- unseen/transfer 文字主张的存在、范围和当前证据缺口。

当前不能从仓库独立复现：

- 任一模型的新 energy/force/stress prediction；
- RDF、bond distribution、coordination/local-structure 曲线；
- 训练、SQS、真实 LASP/SSW、AIMD、DFT 或生产 MLIP-MD；
- top-1 对 AIMD/DFT 的 predictive accuracy；
- unseen 787 configurations 的逐结构误差或三成分 transfer 数值；
- 任一真实集群运行。

## 外部 artifact 接入

若用户希望把 replay 提升为新执行/parity，应通过项目配置提供明确本地路径与可选
SHA-256，而不是把大文件复制进仓库：

| artifact | purpose | minimum metadata |
|---|---|---|
| reference/prediction pairs | fresh benchmark metric execution | model、dataset、split、target、unit、normalization、SHA-256 |
| model checkpoint | prediction/MD | family/config、framework version、elements、training provenance、SHA-256 |
| trajectory / MSD | transport | timestep/time unit、MSD unit、temperature、carrier count、volume、sampling profile |
| LASP/SSW executable or archive | fresh SSW execution / historical normalization | explicit LASP version、executable/input/auxiliary SHA-256、`lasp.in`、`allstr.arc`、可选 `best.arc`/`md.arc`、selection policy、seed status |
| matched AIMD/DFT top-candidate result | high-fidelity validation | exact composition/structure ID、method、unit、paired metric、source digest |
| unseen/transfer compact table | numerical generalization parity | split definition、787-config aggregation或逐项值、三 composition IDs、reference/prediction values、units、source digest |

模型权重、完整数据集、轨迹、VASP 大文件、POTCAR、私有 site config 与凭据仍不得
进入开源仓库。
