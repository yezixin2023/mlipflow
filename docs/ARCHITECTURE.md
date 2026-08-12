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
源码、现有输入和远端模板指纹；任一内容改变后旧摘要失效。

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

CPU 任务选择 `slurm/cpu.sbatch`，GPU 任务选择 `slurm/gpu.sbatch`；插件只提供安全
`template_family`，例如 `vasp` 选择 `vasp/run.sh`。模板只能使用固定占位符：

```text
PROJECT_ID NODE_ID ATTEMPT RUN_DIR INPUT_DIR OUTPUT_DIR LOG_DIR
CPUS GPUS MEMORY WALLTIME
```

渲染器只做精确 `{{NAME}}` 替换、换行规范化与 SHA-256 绑定，不支持表达式、include、
循环或 arbitrary code templating。模板缺失、变量未知/不完整、资源缺失或 profile 不存在
都会在 staging 前明确失败。

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

每个 attempt 写独立 `run-manifest.json`。小文件和小目录树默认 SHA-256；超过阈值的大文件/目录记录 URI、size、mtime 和 metadata fingerprint，避免默认扫描多 GB 数据。Adapter 产物必须是 fresh attempt 内的普通文件；外部引用必须走显式 URI+fingerprint 契约。

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
