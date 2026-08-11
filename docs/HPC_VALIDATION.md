# HPC 验证状态

最后更新：2026-08-11。

```text
REAL_HPC_INTEGRATION = EXTERNAL_VALIDATION_PENDING
```

这不是项目阻塞项。按照本阶段规范，科学证据重放和本地数值 parity 可以在没有
生产集群的情况下完成。

## 已做与未做

| 项目 | 当前事实 |
|---|---|
| core scheduler abstraction | 已有 local、SLURM、SSH+SLURM 边界；job ID、状态、取消与 identity-bound completion manifest 逻辑有本地 fixture/mock 测试 |
| scientific adapters | 内置科学 adapter 当前只声明 `local`；LASP/SSW execute 可直接运行，MPI 时要求显式、可指纹化的 `mpirun`/`mpiexec` 普通文件路径与 `-np`，仍不提交 scheduler；没有把一次 scheduler `COMPLETED` 自动当作科学 `OK` |
| SSH/remote inspection | 只进行过用户授权的只读源码/小证据检查，包括 LASP/SSW archive、参数和历史选择链审计；这不构成 staging、submission、真实 LASP 执行或 fetch 验证 |
| remote staging | 未验证完整闭环 |
| real `sbatch` submission | 未执行 |
| real queue monitoring | 未执行 |
| real cancellation | 未执行 |
| remote result fetch + local scientific re-check | 未验证完整闭环 |
| production scientific job | 未提交，也不属于本阶段要求 |

因此，不应使用“集群已接通”“真实 SLURM 已验证”“远端工作流已跑通”或“生产
可用”等表述。只读 SSH 成功也不能替代一次受控的 stage → submit → monitor →
fetch → scientific check 测试。

## 后续最小集群验证所需站点字段

这些值应放在用户控制的 site/project 配置中，不得硬编码进 plugin 或仓库：

- SSH config 中已有的 profile 别名；仓库不保存 hostname、用户名、跳板或密钥；
- 明确、受限的 remote work root，以及 local staging/fetch allowlist；
- scheduler 命令和 job-ID 解析约定；
- partition、account、QoS、walltime、CPU/GPU、内存等资源字段；
- module/environment 初始化方式和科学程序版本；
- 一个无生产意义、秒级完成的 tiny command；
- completion manifest 的远端相对路径，以及 project/node/run/attempt/plugin/
  plan digest 身份字段；
- 允许回收的文件清单、单文件/总大小上限和 SHA-256；
- 远端退出码、scheduler terminal state 与本地固定版本 scientific checker 的三重
  成功条件。

凭据只由 SSH agent、用户 SSH config 或站点安全设施管理。MLIPFlow 配置不接受
密码、私钥文本或 token，也不应把 private cluster profile 发布到 GitHub。

## 获得明确授权后的无害测试

真实测试必须在提交前再次向用户展示精确对象、命令、资源、远端目录和预期文件，
并得到明确批准。建议顺序为：

1. 只读检查 site profile 与 remote work root 是否可用；
2. 在一个全新的、精确命名的 attempt 目录 stage 极小输入；
3. 提交一个不加载模型、不运行 DFT/MD/训练的秒级 job；
4. 持久化 job ID 与已批准的 plan digest；
5. 轮询至 terminal state，但不把 `COMPLETED` 直接升级为科学 `OK`；
6. 只回收 allowlist 中的小文件并校验大小与 SHA-256；
7. 在本地运行同一固定 plugin 的 `check`/`collect`；
8. 保存匿名化测试记录，不保存站点地址或凭据。

任何真实提交、取消、清理或覆盖都不应由文档审计自动触发。本阶段没有执行上述
步骤，所以状态保持为 `EXTERNAL_VALIDATION_PENDING`。本地 fake LASP executable
contract smoke 和受限 MPI argv 测试均不能改变这一 HPC 状态。

## 与科学完成度的关系

`REAL_HPC_INTEGRATION = EXTERNAL_VALIDATION_PENDING` 不阻止：

- 历史 benchmark/voltage/timing evidence 的 `REPLAY_VERIFIED`；
- 既有 `target.msd` 的本地输运 post-processing parity；
- 历史 247 候选结果的本地筛选重放；
- LASP/SSW 已有 archive 的本地解析、能量过滤和 accepted-order stride 重放；
- 从真实证据重建 task-aware model routing。

它会阻止的主张是：生产级远端执行、真实调度容错、远端 artifact 回收闭环、站点
性能或生产科学任务已经通过验证。
