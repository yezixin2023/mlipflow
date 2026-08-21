# 架构

## 一句话边界

Agent 解释科研意图并产生科学任务和抽象资源需求；MLIPFlow 持久化并执行显式计划；
插件封装确定性科学能力；cluster profile 与远端模板提供 site-specific execution
knowledge；backend 负责组合、执行和状态协调。Agent 不猜 host、partition、module、
executable、模板或远端 work root。

## 命令/查询分离

```text
查询：list status json inspect logs route doctor
  └─ 只加载 project/plugin/registry manifest
  └─ SQLite mode=ro + query_only
  └─ 不导入 adapter，不调用 backend，不写 event

命令：init run advance retry stop
  └─ 明确写入入口
  └─ expensive/external run 需要 execution approval
  └─ advance/retry/stop 以命令本身表达继续或停止意图
  └─ 外部程序始终 argv + shell=False
```

`--dry-run` 不建库、不建 attempt 目录、不 staging、不提交。local 计划不联网；
`ssh-slurm` 计划会按显式 local site profile 通过 SSH **只读** 获取所需远端模板并展示
渲染脚本。执行模式的计划会加载所选 Python adapter，因此
插件与构建脚本一样属于受信代码；查询命令除 `doctor` 的本地 site 校验外不 import
adapter 或访问 backend。批准摘要展示实际执行实现、输入路径、执行参数、资源、backend、
staged 文件以及最终执行的脚本。

## Agent-facing 输出

query/planning service 始终返回一个详细内部对象，执行、审计和展示共用这一个 source of
truth。CLI 默认只做只读投影：`status` 显示 node/state/attempt/output roles，`inspect`
显示节点语义与 plugin 摘要，`route` 显示选择、评分、指标和简短淘汰原因，dry-run 显示
将运行什么、inputs/resources 和是否需要审批。默认 text 使用命令专用的行式渲染；默认
JSON 使用同一紧凑投影，不包含完整 manifests、raw configs 或内部 plan。

`--audit` 不重新规划，也不维护第二份 plan model；它只显示同一详细对象，包括 artifact
provenance、完整 plugin manifest、routing contributions/evidence verification 和 adapter/HPC
plan。

## 审批：显式布尔确认

plan `schema_version` 为 3。需要审批的计算先用 `run NODE --dry-run --audit` 展示计划；
用户确认后以 `run NODE --approve` 执行。MLIPFlow 不为 plan、approval、输入、脚本或模板
生成摘要或内容编号。执行时把当次计划写入 attempt 目录，后续 scheduler observation、
bounded fetch 与科学 `check/collect` 继续读取这份普通 JSON 记录。

### Adapter-authored 字段的可移植化

adapter 天然以绝对路径思考：八个插件全部输出 `cwd`，部分还输出绝对 argv、staged
`source` 与引用文件名的诊断信息。这些保留在 audit plan 中；核心在记录前按已知根
改写路径，在执行前还原：

```text
<project>/.mlipflow/runs/n/attempt-1/out  ->  {ATTEMPT_DIR}/out
<project>/prepared/POSCAR                 ->  {PROJECT_ROOT}/prepared/POSCAR
<plugins>/dft-labeling/helper.py           ->  {PLUGIN_DIR}/helper.py
```

`portable.to_portable()` 在 `make_run_plan` 中改写，`portable.to_runtime()` 在
`_execute_ready`、`_scheduled_contract` 与科学 checker 上下文中还原，因此 adapter
始终收到本机绝对路径，argv 与工作目录完全不变——科学行为不受影响。审批看到的是可移植
描述，执行看到的是本机实现。

不在任何已知根内的路径保持原样：例如 `resources.python_executable` 属于项目显式声明的
站点配置，在任何使用同一份 `project.yaml` 的机器上都相同，改写它反而会隐藏审批内容。

测试覆盖 BLOCKED、local READY 与 ssh-slurm READY 等 plan 形态，并确认仓库内路径可在
运行前正确还原。

## 持久状态

状态库为项目目录下 `.mlipflow/state.sqlite3`。主要实体：

- `step_runs`：每个节点每个 attempt 的不可覆盖记录；
- `dependencies`：DAG 边；
- `artifacts`：URI 与角色；
- `events`：状态转换审计；
- `submission_intents`：获批计划及消费时间；
- `metadata`：state schema 版本。状态库归属直接由 attempt rows 的 project id 判定。

READY attempt 在真正执行前绑定当时的完整 node snapshot；提交后的观察、fetch、checker 和
stop 都使用该 snapshot。历史 attempt 因此保持不可变，而当前 `project.yaml` 中无关节点或
未来 attempt 的修改不会使整个状态库失效。

```text
WAIT ──依赖完成──> READY ──本地──> RUNNING ──检查──> OK
 │                         └───────────────> FAIL
 ├─上游失败──> BLOCKED
 └────────────────────────────────────────> STOPPED

READY ──提交──> SUBMITTED ──队列──> PENDING ──调度──> RUNNING
FAIL/STOPPED ──retry──> 新 attempt 的 READY（旧 attempt 保留）
```

SLURM `COMPLETED` 只是调度事实。远端 `completion.json` 必须至少记录
project/node/attempt 和成功退出状态；fetch 仅允许计划声明的路径并执行大小上限检查，之后仍须由固定
adapter 执行科学 `check/collect`。`dft-labeling.label` 的 static VASP 合同是首个接入该
通用 lifecycle 的科学插件；其余内置 adapter 仍只支持 local。

## HPC 三层配置

```text
本地 ~/.mlipflow/site.yaml
  = named cluster selection/control plane
  = backend + SSH config alias + template root + work root

远端 <remote_template_root>/
  = persistent site-specific template library
  = Slurm skeleton + module/environment + launcher/executable knowledge

远端 <work_root>/<project>/<node>/attempt-XXXX/
  = ephemeral per-run workspace
  = rendered scripts + input/output/logs/completion
```

`site.yaml` 不属于 project 或仓库。一个 project node 只写 `backend: ssh-slurm`、
可选的 `backend_profile: <name>` 和 `cpus/gpus/memory/walltime`；省略 profile 时，core
按各 profile 当前满足资源请求的空闲节点数选择集群，显式 profile 始终优先。项目级 `backend_profiles`、
`parameters.submit_script` 与 `parameters.remote_cwd` 已被拒绝。真实 cluster bootstrap、
模板安装和凭据配置是独立站点过程，不属于通用 workflow。

## 模板解析与远端 workspace

新 `scheduled_execution schema_version=3` 同时提供安全 `template_family` 和
`execution_model`。核心先选择 `slurm/<execution-model>/cpu.sbatch` 或
`slurm/<execution-model>/gpu.sbatch`，再由 `template_family` 选择例如 `vasp/run.sh`。
模板只能使用固定占位符：

```text
PROJECT_ID NODE_ID ATTEMPT RUN_DIR INPUT_DIR OUTPUT_DIR LOG_DIR
CPUS GPUS MEMORY WALLTIME
```

渲染器只做精确 `{{NAME}}` 替换和换行规范化，不支持表达式、include、
循环或 arbitrary code templating。模板缺失、变量未知/不完整、资源缺失或 profile 不存在
都会在 staging 前明确失败。

execution model 固定 `CPUS` 语义：

- `single-python`：`CPUS` 是一个 Python 进程的线程预算；模板必须含
  `--ntasks=1` 与 `--cpus-per-task={{CPUS}}`。
- `mpi`：`CPUS` 是 MPI task/rank 数；模板必须含 `--ntasks={{CPUS}}`，且不得同时
  把 `CPUS` 用作 `cpus-per-task`。

MLIP training 与 ASE MD 使用前者；VASP、scheduled LAMMPS 与 LASP 使用后者。核心会在
staging 前拒绝映射错误的 site template，因此修复单进程 Python 不会改变真正的 MPI 布局。

attempt number 只来自 SQLite 中现有 attempt state。workspace 固定为
`<work_root>/<project-id>/<node-id>/attempt-XXXX/`，包含 `submit.sbatch`、`run.sh`、
`input/`、`output/`、`logs/stdout.log`、`logs/stderr.log` 与 `completion.json`。目录必须
fresh；retry 递增 attempt，永不覆盖旧目录。template root 与 work root 必须互不嵌套。

## SSH-SLURM 执行状态机

```text
resolve local profile
-> resolve remote templates
-> render deterministic execution plan/scripts
-> create fresh attempt workspace
-> stage reviewed inputs/scripts
-> submit and persist job ID
-> monitor scheduler
-> inventory/fetch outputs, logs and completion
-> plugin scientific check
-> plugin collect
-> OK
```

`run` 审批覆盖 resolution、render、stage 与 submit。scheduler terminal 后，普通
`advance` 按该 run 的 output allowlist 读取 inventory，执行 bounded fetch，再进入科学
检查；不产生第二次审批。

## 插件发现

查询侧只按排序后的 `plugins/*/plugin.yaml` 读取静态清单，校验 ID/API/后端/回放声明，不 import Python。只有显式执行路径才允许加载 adapter。科学插件不得硬编码主机、私钥、绝对个人路径或调度资源。

## 产物与回放

每个 attempt 写独立 `run-manifest.json`，记录路径、角色、参数、软件版本、seed、命令和
输出。大型外部 dataset/model 使用站点根目录下的安全相对路径；Adapter 产物必须是 fresh
attempt 内的普通文件。

回放只接受项目根内的普通 result manifest，要求显式 `OK`，并只引用其目录内明确列出的
普通文件；拒绝绝对路径、`..` 与符号链接，不复制数据或运行数值程序。

## 模型路由

核心不含“最佳模型”常量。路由过程：

1. 按元素、任务和 benchmark 场景过滤；
2. 缺少必需指标的模型被淘汰并给出原因；
3. 按项目 policy 的方向和权重归一评分；
4. 平局按元素覆盖、验证样本数、model id 稳定排序；
5. 默认输出 selected model、ranking/metrics 和简短淘汰原因；`--audit` 额外显示每项贡献与 evidence verification。

本地 benchmark evidence 记录普通文件路径；外部 URI 明确标为 external。路由只使用声明
的 task/scenario/metrics/policy，不把文件工程属性当成科学指标。

示例中的 DeepMD/CHGNet 选择来自示例 registry 的合成 benchmark fixture，不是核心偏好。

## 后端边界

- local：同步运行显式 argv、`shell=False`、白名单环境，并在固定 adapter check/collect 后判定科学状态；LASP/SSW 可直接运行；MPI 只接受显式且 basename 为 `mpirun`/`mpiexec` 的普通可执行文件路径与 `-np N`，这仍是 local execution，不是 scheduler backend；
- SLURM：保留 scheduler command abstraction，但 scientific node 不再以用户自备完整 sbatch 作为主执行合同；
- SSH+SLURM：只使用 site config 指向的 SSH alias、remote template library 和 work root；创建全新 attempt workspace、持久化 job ID、终态只读 inventory、bounded local fetch 和科学 checker。POTCAR 永不回收。

远端 staging/submit 只能由获批 run 触发；后续 fetch 只能读取该 run 已绑定的 allowlist 并进行 transport 校验。显式 `stop NODE` 表达 cancel 意图。查询若以后支持 live overlay，也只能驻留内存，不修改状态。
