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
  └─ run/advance/retry/stop 需要 plan digest
  └─ 外部程序始终 argv + shell=False
```

`--dry-run` 不建库、不建 attempt 目录、不 staging、不提交。local 计划不联网；
`ssh-slurm` 计划会按显式 local site profile 通过 SSH **只读** 获取所需远端模板，才能把
模板 identity 与渲染脚本纳入审批摘要。执行模式的计划会加载所选 Python adapter，因此
插件与构建脚本一样属于受信代码；查询命令除 `doctor` 的本地 site 校验外不 import
adapter 或访问 backend。批准摘要绑定项目配置、site digest、plugin manifest/adapter
源码、现有输入和远端模板指纹；任一**内容**改变后旧摘要失效。

## 审批身份：内容，不是文件系统状态

plan `schema_version` 为 2。审批身份只能由声明配置和文件**内容**导出：

```text
content identity（可进入 plan_digest）      observational metadata（永不进入）
  locator：project/plugin 相对路径            uri：绝对 file:// 路径
  size_bytes                                  mtime_ns
  content：SHA-256 / tree-SHA-256
  content_mode
```

`artifacts.content_identity()` 产出前者，`artifacts.fingerprint()` 产出后者并仍随
artifact 落盘供审计。二者共用同一个内容摘要，因此目录树身份在两侧都与 mtime 无关。

由此，同一份内容在 `touch`、`git clone` 到别处、`rsync` 到另一台机器后产生**相同**的
`plan_digest`；内容一改则必然改变。schema 1 曾把 `mtime_ns` 与绝对 `file://` URI 经
`implementation_fingerprint`/`input_fingerprints` 送入摘要，其后果是排队中的作业会因为
任何原地重写而永久无法 `advance`。schema 2 用 `implementation_identity`、
`input_identities` 与 `adapter_command_identities` 取代之。

超过阈值的大文件不读取内容，报告 `content: null` 与 `content_mode: "size-only"`：
metadata 哈希不是内容证据，不得冒充内容证据。

### Adapter-authored 字段的可移植化

adapter 天然以绝对路径思考：八个插件全部输出 `cwd`，部分还输出绝对 argv、staged
`source` 与引用文件名的诊断信息。这些都落在被签名的 `adapter_plan` /
`adapter_diagnostics` 里。核心因此在签名前按已知根改写，在执行前还原：

```text
<project>/.mlipflow/runs/n/attempt-1/out  ->  {ATTEMPT_DIR}/out
<project>/prepared/POSCAR                 ->  {PROJECT_ROOT}/prepared/POSCAR
<plugins>/dft-labeling/helper.py           ->  {PLUGIN_DIR}/helper.py
```

`portable.to_portable()` 在 `make_run_plan` 中改写，`portable.to_runtime()` 在
`_execute_ready`、`_scheduled_contract` 与 pinned checker 上下文中还原，因此 adapter
始终收到本机绝对路径，argv 与工作目录完全不变——科学行为不受影响。审批看到的是可移植
描述，执行看到的是本机实现。

不在任何已知根内的路径保持原样：例如 `resources.python_executable` 属于项目显式声明的
站点配置，在任何使用同一份 `project.yaml` 的机器上都相同，改写它反而会隐藏审批内容。

实测覆盖：8 个插件的 BLOCKED plan、2 个 local READY plan、1 个 ssh-slurm READY plan
共 11 种 plan 形态，machine-specific path leaves = 0，mtime leaves = 0。

## 持久状态

状态库为项目目录下 `.mlipflow/state.sqlite3`。主要实体：

- `step_runs`：每个节点每个 attempt 的不可覆盖记录；
- `dependencies`：DAG 边；
- `artifacts`：URI、角色、大小和指纹；
- `events`：状态转换审计；
- `submission_intents`：获批计划及消费时间。
- `metadata`：schema 版本及初始化时的完整项目配置摘要，防止 backend/profile/node 静默漂移。

```text
WAIT ──依赖完成──> READY ──本地──> RUNNING ──检查──> OK
 │                         └───────────────> FAIL
 ├─上游失败──> BLOCKED
 └────────────────────────────────────────> STOPPED

READY ──提交──> SUBMITTED ──队列──> PENDING ──调度──> RUNNING
FAIL/STOPPED ──retry──> 新 attempt 的 READY（旧 attempt 保留）
```

SLURM `COMPLETED` 只是调度事实。远端 `completion.json` 必须至少绑定
project/node/attempt、成功退出状态，并在批准与 fetch 间保持相同指纹；之后仍须由固定
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
`backend_profile: <name>` 和 `cpus/gpus/memory/walltime`；项目级 `backend_profiles`、
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

渲染器只做精确 `{{NAME}}` 替换、换行规范化与 SHA-256 绑定，不支持表达式、include、
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
-> resolve/fingerprint remote templates
-> render deterministic execution plan/scripts
-> create fresh attempt workspace
-> stage approved inputs/scripts
-> submit and persist job ID
-> monitor scheduler
-> inventory/fetch outputs, logs and completion
-> plugin scientific check
-> plugin collect
-> OK
```

第一次 `run` 审批覆盖 resolution、render、stage 与 submit。scheduler terminal 后，
第二次 `advance` 审批绑定远端输出 inventory；fetch 后再次校验 identity，再进入科学检查。

## 插件发现

查询侧只按排序后的 `plugins/*/plugin.yaml` 读取静态清单，校验 ID/API/后端/回放声明，不 import Python。只有显式执行路径才允许加载 adapter。科学插件不得硬编码主机、私钥、绝对个人路径或调度资源。

## 产物与回放

每个 attempt 写独立 `run-manifest.json`。小文件和小目录树默认 SHA-256；超过阈值的大文件记录 URI、size、mtime 和 metadata fingerprint，避免默认扫描多 GB 数据。大目录树退化为 `tree-structure-sha256`（路径+size+symlink 目标，不含 mtime），仍是稳定的结构身份。artifact 记录保留 URI 与 mtime 供审计，但这些字段属于 observational metadata，不参与任何审批摘要。Adapter 产物必须是 fresh attempt 内的普通文件；外部引用必须走显式 URI+fingerprint 契约。

回放只接受项目根内的普通 result manifest，要求显式 `OK`，并只为其目录内明确列出的普通文件建立引用和指纹；拒绝绝对路径、`..` 与符号链接，不复制数据或运行数值程序。

## 模型路由

核心不含“最佳模型”常量。路由过程：

1. 按元素、任务和 benchmark 场景过滤；
2. 缺少必需指标的模型被淘汰并给出原因；
3. 按项目 policy 的方向和权重归一评分；
4. 平局按元素覆盖、验证样本数、model id 稳定排序；
5. 输出 policy digest、每项贡献、完整候选和淘汰原因。

本地 benchmark evidence 默认必须通过完整 SHA-256；外部 URI 或仅 size/metadata 验证的证据会被排除，除非项目 policy 明确降级授权。

示例中的 DeepMD/CHGNet 选择来自示例 registry 的合成 benchmark fixture，不是核心偏好。

## 后端边界

- local：同步运行显式 argv、`shell=False`、白名单环境，并在固定 adapter check/collect 后判定科学状态；LASP/SSW 可直接运行；MPI 只接受显式、可指纹化且 basename 为 `mpirun`/`mpiexec` 的普通可执行文件路径与 `-np N`，这仍是 local execution，不是 scheduler backend；
- SLURM：保留 scheduler command abstraction，但 scientific node 不再以用户自备完整 sbatch 作为主执行合同；
- SSH+SLURM：只使用 site config 指向的 SSH alias、remote template library 和 work root；创建全新 attempt workspace、逐文件 SHA-256 staging、持久化 job ID、终态只读 inventory、第二次审批、fresh local fetch 和 pinned scientific checker。POTCAR 永不回收。

远端 staging、fetch、cancel 均只能由获批命令触发。查询若以后支持 live overlay，也只能驻留内存，不修改状态。
