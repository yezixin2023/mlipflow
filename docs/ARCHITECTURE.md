# 架构

## 一句话边界

Agent 解释科研意图和结构化结果；MLIPFlow 持久化并执行显式计划；插件封装确定性科学能力；backend 负责本地、SLURM 或 SSH+SLURM 计算。

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

`--dry-run` 不建库、不建 attempt 目录、不 SSH、不提交。执行模式的计划会加载所选 Python adapter，因此插件与构建脚本一样属于受信代码；查询命令永远不 import adapter。批准摘要绑定项目配置、plugin manifest/adapter 源码、现有输入和 adapter argv 中的本地文件；任一内容改变后旧摘要失效。

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

SLURM `COMPLETED` 只是调度事实。完成清单必须绑定 project/node/run/attempt/plugin/plan digest，并在批准与落库间保持相同指纹。`dft-labeling.label` 的单结构 static SSH-SLURM 合同会在第二次 `advance` 审批后执行 allowlisted fetch，再运行固定 adapter 的 `check/collect`；其余内置 adapter 仍只支持 local。

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
- SLURM：保存 `sbatch` 回执中的 job ID，再按 ID 查询/取消；当前用于显式用户脚本及身份绑定 completion；
- SSH+SLURM：只使用 SSH config alias；不在仓库保存连接凭据。DFT static 窄合同要求已存在的 `remote_root`、全新 identity 派生目录、basename-only staging、上传后逐文件 SHA-256、持久化 job ID、终态输出只读 inventory、第二次审批、fresh local fetch 和 pinned scientific checker。POTCAR 永不回收。

远端 staging、fetch、cancel 均只能由获批命令触发。查询若以后支持 live overlay，也只能驻留内存，不修改状态。
