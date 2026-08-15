# HPC 验证状态

最后更新：2026-08-14。

```text
REAL_HPC_INTEGRATION = SCIENTIFIC_PROGRAM_VERIFIED_ON_ONE_SITE
```

同一个真实 SSH-SLURM 集群上，通用生命周期
`resolve → render → fresh attempt workspace → stage → sbatch → squeue/sacct →
bounded fetch → completion 身份校验 → plugin check/collect` 已经走通两次：

1. 一次 **CPU tiny smoke**，科学内容为零：远端脚本只对 staged 输入重算 SHA-256
   并回传，因此 `check` 只能证明输入字节完整到达计算节点、作业确实在计算节点
   执行、输出字节完整取回。
2. 一次 **真实 DeePMD-kit 训练**（job `27356151`，500 步 fresh training，
   scheduler `COMPLETED`，MLIPFlow `OK`）。这是第一次有真实科学程序在调度器上
   由 MLIPFlow 端到端执行并通过科学 check：`dp train` 正常结束、实际完成步数等于
   请求步数、learning curve 全部有限、checkpoint 存在、dataset/config/version
   身份三项均与已批准 plan 一致。

**这一条仍不能被读作**：production ready、VASP/LAMMPS 等其它科学程序已验证、
GPU 已验证、多节点已验证、排队/容错行为已验证，或该站点之外的任何集群已验证。
已验证的科学程序只有一个：CPU、单节点、单进程、TensorFlow 后端的 DeePMD-kit
`dp train`。

## 已做与未做

| 项目 | 当前事实 |
|---|---|
| local site control plane | 多 cluster `site.yaml` parsing、profile selection、缺失/危险配置失败均有 synthetic fixture 测试；本阶段未读取真实 `~/.mlipflow/site.yaml` |
| remote template library | CPU/GPU Slurm 模板选择、program-family `run.sh`、固定变量合同、模板 identity 与不完整模板失败均有 fake library 测试 |
| remote workspace | `<work_root>/<project>/<node>/attempt-XXXX`、attempt 递增、fresh directory、input/output/logs/completion 分层与逐文件 SHA-256 staging 有 mock 测试 |
| core scheduler lifecycle | resolve → render → stage → submit → persist job ID → monitor → inventory/fetch → check → collect 状态逻辑有本地 fixture/mock 测试 |
| scientific adapters | static `dft-labeling.label`、四框架 `mlip-training`、scheduled LASP、ASE MD 与 LAMMPS execute 已接入通用 `ssh-slurm` 合同；真实调度器仍只验证过 DeepMD CPU 训练，scheduler `COMPLETED` 不会自动变成科学 `OK` |
| SSH/remote access | 已在一个真实站点执行只读探索与受控写入；写入全部限制在本次新建的独立 MLIPFlow root 内 |
| remote staging | 真实站点已验证：fresh attempt workspace、逐文件 SHA-256 staging |
| real `sbatch` submission | 已执行 CPU tiny job 与一次真实 DeePMD 训练 job，均取得并持久化 job ID |
| real queue monitoring | 已通过 `squeue`/`sacct` 观察到 terminal `COMPLETED` |
| real cancellation | 未执行 |
| remote result fetch + local re-check | 真实站点已验证：bounded allowlist 抓取、二次审批绑定 inventory、fetch 后指纹复核、completion 身份字段校验、pinned plugin `check`/`collect` |
| scientific program on scheduler | **已验证一个**：DeePMD-kit v3.0.0b1（commit `4c565b9`、TensorFlow 2.15.0、float64、CPU）`dp train` 在 job `27356151` 上完成 500 步 fresh training，scientific check/collect 到 `OK`。VASP/LAMMPS 等仍未在调度器上运行 |
| training numerical reproduction | **已验证一次**：与历史 run `split_train/se_e2_a/para0` 的前 6 个 reporting step（0/100/…/500）在 lcurve 全部 7 列上逐位相同，相对差 0.0 |
| GPU / multi-node | **未验证**。两次真实执行均为单节点 CPU |
| production scientific job | 未提交。DeePMD 训练为 500 步 bounded validation run，不是生产训练 |

因此仍不应使用“生产可用”“GPU/多节点已验证”等表述。可以说明的是：通用 scheduler
合同已在一个真实站点完成端到端 CPU tiny smoke，并且已经承载一次真实 DeePMD
训练，其早期优化轨迹可与历史 run 逐步对照。

## 三层站点合同

用户本地 `~/.mlipflow/site.yaml` 只保存 named cluster control plane：

- SSH config 中已有的 profile 别名；仓库不保存 hostname、用户名、跳板或密钥；
- 明确、受限且互不嵌套的 `remote_template_root` 与 `work_root`；
- backend 当前必须是 `ssh-slurm`。

远端 template library 保存站点执行知识：

- `slurm/single-python/{cpu,gpu}.sbatch` 与 `slurm/mpi/{cpu,gpu}.sbatch`；
- `<program>/run.sh`；
- partition/account/QoS/GPU directive；
- module/environment 初始化、launcher 和科学程序路径/版本。

project/workflow 只保存 `backend_profile` 和抽象资源 `cpus/gpus/memory/walltime`。
实际 run workspace 由 work root、project、node 与持久 attempt number 确定。不得把完整
sbatch、module、executable 或远端路径塞回 node parameters。

schema v3 的 execution model 使 `resources.cpus` 不再含糊：single Python job 固定为
`--ntasks=1 --cpus-per-task={{CPUS}}`；MPI job 固定为 `--ntasks={{CPUS}}`，且不能把
同一个 `CPUS` 再用于 `cpus-per-task`。站点验证必须分别覆盖两套模板。

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

随后的真实 DeePMD 训练把状态提升为 `SCIENTIFIC_PROGRAM_VERIFIED_ON_ONE_SITE`，
其边界同样严格：一个站点、单节点、单进程 CPU、一个科学程序（DeePMD-kit `dp
train`）、一个 descriptor（`se_e2_a`）、一个数据集（180+180 Li10 systems）、
500 步 bounded run。证据见
[`reports/deepmd_early_training_reproduction.json`](../reports/deepmd_early_training_reproduction.json)。

同一 smoke 暴露的一个通用缺陷已修复：`render_template` 原先在替换后拒绝任何残留
的 `{{` 或 `}}`，使得合法的 shell/JSON（`"${value}}"`、`{"a":{"b":1}}`）无法写入
模板；现在只拒绝真正未解析的 `{{ ... }}` 占位符语法。

真实训练暴露的一个科学陷阱记录在此：DeePMD 的 `LearningRateExp.build()` 会从
`numb_steps` 反推 `decay_rate`，并在 `decay_steps >= numb_steps` 时把 `decay_steps`
替换为 `numb_steps // 100 + 1`。因此**只缩短 `numb_steps` 就会改变学习率
schedule**——把 300000 步的衰减压缩进 500 步，step 100 的 lr 会从历史的
`1.0e-03` 变成 `1.4e-04`，早期优化轨迹随之偏离，损失不再可比。截断复现必须同时
补偿 LR schedule，并显式记录这一补偿。

## 与科学完成度的关系

HPC 状态轴与科学状态轴彼此独立。即使 HPC 仍处于早期验证阶段，也不阻止：

- 历史 benchmark/voltage/timing evidence 的 `REPLAY_VERIFIED`；
- 既有 `target.msd` 的本地输运 post-processing parity；
- 历史 247 候选结果的本地筛选重放；
- LASP/SSW 已有 archive 的本地解析、能量过滤和 accepted-order stride 重放；
- 从真实证据重建 task-aware model routing。

它会阻止的主张是：生产级远端执行、真实调度容错、远端 artifact 回收闭环、站点
性能或生产科学任务已经通过验证。
