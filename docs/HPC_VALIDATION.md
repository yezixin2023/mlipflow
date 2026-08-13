# HPC 验证状态

最后更新：2026-08-12。

```text
REAL_HPC_INTEGRATION = SCHEDULER_CONTRACT_VERIFIED_ON_ONE_SITE
```

一个真实 SSH-SLURM 集群上已完成一次 **CPU tiny smoke**，通用生命周期
`resolve → render → fresh attempt workspace → stage → sbatch → squeue/sacct →
bounded fetch → completion 身份校验 → plugin check/collect` 全链路到达 `OK`。

该 smoke 的科学内容为零：远端脚本只对 staged 输入重算 SHA-256 并回传，因此
`check` 能够证明输入字节完整到达计算节点、作业确实在计算节点执行、输出字节完整
取回——除此之外不证明任何事情。

**这一条不能被读作**：production ready、任何科学程序（VASP/LAMMPS/DeePMD 等）已
验证、GPU 已验证、多节点已验证、排队/容错行为已验证，或该站点之外的任何集群已
验证。它只说明通用 scheduler 合同在一个真实站点上可以走通。

## 已做与未做

| 项目 | 当前事实 |
|---|---|
| local site control plane | 多 cluster `site.yaml` parsing、profile selection、缺失/危险配置失败均有 synthetic fixture 测试；本阶段未读取真实 `~/.mlipflow/site.yaml` |
| remote template library | CPU/GPU Slurm 模板选择、program-family `run.sh`、固定变量合同、模板 identity 与不完整模板失败均有 fake library 测试 |
| remote workspace | `<work_root>/<project>/<node>/attempt-XXXX`、attempt 递增、fresh directory、input/output/logs/completion 分层与逐文件 SHA-256 staging 有 mock 测试 |
| core scheduler lifecycle | resolve → render → stage → submit → persist job ID → monitor → inventory/fetch → check → collect 状态逻辑有本地 fixture/mock 测试 |
| scientific adapters | 默认只声明 `local`；`dft-labeling.label` static VASP 是首个接入通用 `ssh-slurm` 合同的插件。LASP/SSW execute 仍为 local；scheduler `COMPLETED` 不会自动变成科学 `OK` |
| SSH/remote access | 已在一个真实站点执行只读探索与受控写入；写入全部限制在本次新建的独立 MLIPFlow root 内 |
| remote staging | 真实站点已验证：fresh attempt workspace、逐文件 SHA-256 staging |
| real `sbatch` submission | 已执行一次 CPU tiny job，取得并持久化 job ID |
| real queue monitoring | 已通过 `squeue`/`sacct` 观察到 terminal `COMPLETED` |
| real cancellation | 未执行 |
| remote result fetch + local re-check | 真实站点已验证：bounded allowlist 抓取、二次审批绑定 inventory、fetch 后指纹复核、completion 身份字段校验、pinned plugin `check`/`collect` |
| scientific program on scheduler | **未验证**。已完成的 smoke 不加载任何 module，也不调用任何科学程序 |
| GPU / multi-node | **未验证**。smoke 为单节点单核 CPU |
| production scientific job | 未提交 |

因此仍不应使用“生产可用”“科学程序已验证”“GPU/多节点已验证”等表述。可以说明
的是：通用 scheduler 合同已在一个真实站点完成一次端到端 CPU tiny smoke。

## 三层站点合同

用户本地 `~/.mlipflow/site.yaml` 只保存 named cluster control plane：

- SSH config 中已有的 profile 别名；仓库不保存 hostname、用户名、跳板或密钥；
- 明确、受限且互不嵌套的 `remote_template_root` 与 `work_root`；
- backend 当前必须是 `ssh-slurm`。

远端 template library 保存站点执行知识：

- `slurm/cpu.sbatch` 与 `slurm/gpu.sbatch`；
- `<program>/run.sh`；
- partition/account/QoS/GPU directive；
- module/environment 初始化、launcher 和科学程序路径/版本。

project/workflow 只保存 `backend_profile` 和抽象资源 `cpus/gpus/memory/walltime`。
实际 run workspace 由 work root、project、node 与持久 attempt number 确定。不得把完整
sbatch、module、executable 或远端路径塞回 node parameters。

## 后续真实集群验证仍需检查

- SSH alias、host key 与认证在目标用户环境中工作；
- template root 只读、work root 可写，且共享文件系统语义符合预期；
- template 中 scheduler 命令、job-ID 解析和站点 launch 约定正确；
- partition/account/QoS、CPU/GPU/内存/walltime 映射符合站点策略；
- 一个无生产意义、秒级完成的 tiny command；
- completion manifest 的 project/node/attempt 与退出状态身份字段；
- 允许回收的文件清单、单文件/总大小上限和 SHA-256；
- 远端退出码、scheduler terminal state 与本地固定版本 scientific checker 的三重
  成功条件。

凭据只由 SSH agent、用户 SSH config 或站点安全设施管理。MLIPFlow 配置不接受
密码、私钥文本或 token，也不应把 private cluster profile 发布到 GitHub。

## 获得明确授权后的无害测试

真实测试必须在提交前再次向用户展示精确对象、命令、资源、远端目录和预期文件，
并得到明确批准。建议顺序为：

1. 只读检查 local site profile、远端 template root 与 work root；
2. 审查被选中的模板及渲染结果；
3. 在一个全新的、精确命名的 attempt 目录 stage 极小输入；
4. 对基础 backend 可提交秒级无害 job；对已单独批准的 DFT smoke，可提交一个单结构 static VASP job；
5. 持久化 job ID 与已批准的 plan digest；
6. 轮询至 terminal state，但不把 `COMPLETED` 直接升级为科学 `OK`；
7. 只回收 allowlist 中的小文件并校验大小与 SHA-256；
8. 在本地运行同一固定 plugin 的 `check`/`collect`；
9. 保存匿名化测试记录，不保存站点地址或凭据。

任何真实提交、取消、清理或覆盖都不应由文档审计自动触发。

已完成的真实 CPU tiny smoke 把状态从 `EXTERNAL_VALIDATION_PENDING` 提升为
`SCHEDULER_CONTRACT_VERIFIED_ON_ONE_SITE`，其边界严格限于：一个站点、单节点、
单核、无 module、无科学程序。本地 fake LASP executable、fake VASP 输出和受限
MPI argv 测试仍然不能改变 HPC 状态。

同一 smoke 暴露的一个通用缺陷已修复：`render_template` 原先在替换后拒绝任何残留
的 `{{` 或 `}}`，使得合法的 shell/JSON（`"${value}}"`、`{"a":{"b":1}}`）无法写入
模板；现在只拒绝真正未解析的 `{{ ... }}` 占位符语法。

## 与科学完成度的关系

`REAL_HPC_INTEGRATION = EXTERNAL_VALIDATION_PENDING` 不阻止：

- 历史 benchmark/voltage/timing evidence 的 `REPLAY_VERIFIED`；
- 既有 `target.msd` 的本地输运 post-processing parity；
- 历史 247 候选结果的本地筛选重放；
- LASP/SSW 已有 archive 的本地解析、能量过滤和 accepted-order stride 重放；
- 从真实证据重建 task-aware model routing。

它会阻止的主张是：生产级远端执行、真实调度容错、远端 artifact 回收闭环、站点
性能或生产科学任务已经通过验证。
