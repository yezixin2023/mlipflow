# MLIPFlow 迁移与实现计划

## 总原则

- 新实现只存在于 `mlipflow/`，不修改原 taskflow、远端 MLP 或工作区 claw。
- LLM 只负责观察、解释、规划建议和经授权调用；数值结果由确定性插件产生。
- 先包装、后重写；先回放、后执行；先结构化证据、后自动路由。
- 大数据、轨迹、模型、POTCAR 和私钥不进入仓库，只存路径引用和必要元数据。
- 每一阶段更新 `docs/IMPLEMENTATION_STATUS.md`，未验证能力不得写成“已支持”。

## 目标架构

```text
Agent Skills (.agents/skills)       人类/LLM 的监督说明
              │
              ▼
CLI + Query/Command services        严格 CQRS、计划展示与布尔批准
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
 state     plugin     registry/routing
 SQLite    contract   benchmark evidence
     │        │        │
     └────────┼────────┘
              ▼
 local / SLURM / SSH+SLURM backends
              │
              ▼
 外部科研程序、已有文件与只读回放产物
```

## Phase 0：盘点与边界（当前阶段）

交付：

- `TASKFLOW_REFERENCE.md`：参考架构、可复用点和安全反例；
- `CODEBASE_INVENTORY.md`：taskflow、远端 MLP 和 claw 的选择性盘点；
- 本文件：来源到插件的迁移路径与阶段门；
- `IMPLEMENTATION_STATUS.md`：逐阶段诚实状态。

退出条件：主要能力均有来源证据、A–E 分类、数据排除和已知科学缺口。

## Phase 1：核心契约与只读安全

实现：

- 正常 Python 包、`mlipflow` CLI 和稳定 JSON envelope；
- `project.yaml`、`plugin.yaml`、run manifest 与 model registry schemas；
- SQLite 显式状态：`PREP READY SUBMITTED PENDING RUNNING OK FAIL WAIT BLOCKED STOPPED`；
- run、attempt、dependency、artifact、event 和 submission intent；
- `list/status/json/inspect/logs/route/doctor` 严格只读；
- `init/run/advance/retry/stop` 显式写入；昂贵/外部操作先 dry-run，再用布尔 `--approve`；
- 配置优先级和来源追踪。

阶段门：只读命令在空目录、已有项目和只读数据库上均零文件/进程/提交副作用；非法状态迁移被拒绝。

## Phase 2：插件框架、manifest 与回放

实现统一 `plugin.yaml` 和适配器接口：

```python
validate(context) -> diagnostics
plan(context) -> execution_plan        # 纯函数
prepare(context, plan) -> prepared_run
check(context) -> completion_result
collect(context) -> plugin_result
replay(context) -> plugin_result
```

首批计算插件目录：

1. `high-entropy-structure`
2. `pes-sampling`
3. `dft-labeling`
4. `mlip-training`
5. `mlip-benchmark`
6. `ionic-transport`
7. `candidate-ranking`
8. `electrochemical-voltage`

可选辅助：`descriptor-analysis`、`model-export`。

所有外部命令必须为 argv 列表并使用 `shell=False`；回放只解析/引用已有产物，不启动数值程序、不提交、不复制大文件。

阶段门：每个插件能被 schema 校验和发现；回放 fixture 生成有效 result manifest；dry-run 零写入。

## Phase 3：执行后端

接口分为：

- `Scheduler`: submit/query/cancel/logs；
- `Transport`: stage-in/stage-out/stat/read-tail/run-readonly；
- `Backend`: plan/submit/query/cancel/fetch/logs。

按序实现：

1. Local backend；
2. fake SLURM 与 fake SSH 测试后端；
3. SLURM backend；
4. SSH+SLURM backend。

仓库只引用用户本地 `~/.mlipflow/site.yaml` 中的 named cluster profile；profile 再引用
`~/.ssh/config` alias，并把持久 `remote_template_root` 与 per-run `work_root` 分开。
workflow node 只给 scientific contract 与抽象 resources，不接收完整 sbatch。job ID 必须
由提交回执保存，不允许靠工作目录猜测；远端 staging 只发生于显式已批准命令。

阶段门：synthetic multi-cluster site config、fake template library/backend 可完整覆盖
profile resolution、deterministic rendering、attempt workspace、stage、pending/running/
completed/failed/cancelled、fetch/check/collect；重启后能恢复。真实 cluster bootstrap 与
验证保持为独立站点步骤。

## Phase 4：逐能力包装

### 4.1 静态 benchmark（优先）

来源：CHGNet `cal_new.py` 与 M3GNet `cal.py`。先做薄适配器，输出统一 `metrics.json`/CSV；建立 E/F/S 单位、样本计数和 per-atom 约定测试。

### 4.2 MACE 训练/微调与导出

来源：远端 MACE train/finetune、工作区 claw 的微调脚本、MACE→LAMMPS 导出。先包装框架 CLI，不重写训练算法；修复/拒绝原 SLURM 续行错误。

### 4.3 DFT 数据装配

来源：DeepMD/CHGNet/NequIP 提取脚本与 `vasp2lasptrain.py`。先统一 VASP 完成判据、结构/能量/力/应力记录和确定性 split；再按 DeepMD npy、extxyz、JSON、LASP 文本导出。这里的 LASP `TrainStr`/`TrainFor` 是标注数据导出格式，不是 LASP/SSW 采样执行。

### 4.4 PES 采样

已把 sampler、snapshot policy 和 DFT labeling 拆开。当前 `pes-sampling` 有三个显式操作：插件内置、local-only 且继续调用 MAML DIRECT 的 `direct-select`；对用户自备 LASP 的 local/`ssh-slurm` `lasp-ssw-execute`；以及只解析现有 `allstr.arc`、可选 `best.arc`/`md.arc` 的 `lasp-ssw-normalize-replay`。DIRECT 不再接受外部脚本路径并拒绝复用已有输出目录。LASP execute 要求显式 LASP 版本与输入文件；local MPI 仅接受显式、basename 为 `mpirun`/`mpiexec` 的普通可执行文件路径与 `-np` 数量，scheduled 路径则由 site template 提供 executable/MPI。历史 normalization 的 frame/过滤/stride 链已经按来源路径和记录字段重放；一个真实 LASP 3.6.0 NN 14-atom CPU tiny case 已完成 scheduled functional closure（job `redacted`，14/14/14，bounded fetch + checker/collect=`OK`）。AIMD、MLP-MD、生产 LASP 和完整科学数值 parity 仍待外部验证。

### 4.5 SQS 与组分枚举

保留 icet/pymatgen/ASE 算法，外置位点映射与组分约束，强制 seed 并记录原型路径。大规模枚举只在 dry-run 估算后批准。

### 4.6 离子输运

先包装轨迹/MSD 分析，再增加执行。参数化平衡段、时间单位、维数、粒子选择、体积、拟合窗与不确定度；多温度 Arrhenius 输出完整回归诊断。

### 4.7 电压

先做确定性后处理，输入为 Li 含量、总能、参考能和 formula units；对相邻组成输出平均电压及符号/单位验证。DFT 结构松弛作为独立上游节点。

### 4.8 模型后端扩展

DeepMD、M3GNet、CHGNet、MACE 为计划中的四个主后端。NequIP、LASP、MTP 仅在依赖与许可条件满足时启用。此处 LASP 指势函数训练后端，与 `pes-sampling` 的 LASP/SSW 外部采样薄封装不同；后者的存在不表示 LASP 势训练已启用。首版不得声称远端已有通用 DeepMD 训练实现。

## Phase 5：模型 Registry 与 task-aware routing

Registry 每条模型记录：

- artifact 路径、family、版本和元素覆盖；
- 支持任务（静态 E/F/S、MD、离子输运、结构优化、电压代理等）；
- 训练数据版本和已知限制；
- benchmark run 与指标；
- 环境/后端兼容性。

Routing 先按元素、任务、场景、限制和关键指标完整性过滤，再按项目策略中的指标方向/权重评分。输出候选、淘汰原因、指标贡献和 policy 参数。平局依次使用覆盖率、验证样本数和 model id，保证可复现。

论文示例中 DeepMD 可因离子输运 benchmark 排名胜出、CHGNet 可因电压相关静态任务胜出，但这些结论只能来自示例 registry/benchmark artifact，核心代码不硬编码。

## Phase 6：Agent Skills 与示例

用仓库级 `.agents/skills/` 提供八个独立 Agent Skills：

- `$mlip-workflow`
- `$high-entropy-structure`
- `$pes-sampling`
- `$dft-labeling`
- `$mlip-training`
- `$mlip-benchmark`
- `$ionic-transport`
- `$candidate-ranking`

Skills 只指导观察、判断、调用、风险确认和解释结果，不实现数值算法。计算插件始终在 `plugins/`。

未来 `$mlip-workflow` 可直接编排 `electrochemical-voltage` plugin 的确定性电压后处理与回放能力，不要求存在独立的 voltage Agent Skill。

`examples/high_entropy_sulfide` 使用小型、可分享 fixture 和 replay manifest 演示完整链路；不复制远端数据或模型。

## Phase 7：测试、开源清理与发布门

必测项：

- 状态转换、retry 新 attempt、事务和崩溃恢复；
- 所有只读命令零副作用、无 daemon/自动推进；
- dry-run 零写、显式布尔批准；
- schema、配置优先级和重复插件检测；
- 外部 argv 注入防护和结构化错误；
- fake local/SLURM/SSH；
- replay 不执行、不复制大文件；
- manifest、原子写、凭据脱敏和 provenance；
- routing 的指标方向、缺失值、场景、稳定 tie-break；
- 小型科学公式和单位 smoke tests。

开源门：无凭据/主机私钥/绝对个人路径/专有 POTCAR/模型权重/大数据；LICENSE、README、贡献说明、依赖与外部程序许可清楚；示例可离线 replay。

## 已知待决策项

- 状态库首版使用 SQLite；如果以后需要共享多用户数据库，另做迁移层，不在 v1 过早引入服务。
- YAML 解析优先 PyYAML；JSON 是 YAML 子集，核心 fixtures 保持可由 stdlib JSON 读取，以便最小环境诊断。
- 普通计算记录路径、参数、软件版本、seed 和输出；不建立项目自定义内容身份体系。
- 远端科学脚本只能作为来源引用，不作为运行时依赖；用户可在本地配置 adapter command/profile。
