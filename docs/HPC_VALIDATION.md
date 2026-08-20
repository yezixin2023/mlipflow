# HPC 验证状态

最后更新：2026-08-18。

```text
REAL_HPC_INTEGRATION = SCIENTIFIC_PROGRAM_VERIFIED_ON_ONE_SITE
```

同一个真实 SSH-SLURM 集群上，通用生命周期
`resolve → render → fresh attempt workspace → stage → sbatch → squeue/sacct →
bounded fetch → completion 字段校验 → plugin check/collect` 已经承载以下有界验证：

1. 一次 **CPU tiny smoke**，科学内容为零：远端脚本读取 staged 输入并回传预期结果，
   因此 `check` 只能证明作业在计算节点执行并且声明的输出被取回。
2. 一次 **真实 DeePMD-kit 训练**（job `redacted`，500 步 fresh training，
   scheduler `COMPLETED`，MLIPFlow `OK`）。这是第一次有真实科学程序在调度器上
   由 MLIPFlow 端到端执行并通过科学 check：`dp train` 正常结束、实际完成步数等于
   请求步数、learning curve 全部有限、checkpoint 存在，且 dataset/config 路径与版本
   均符合执行记录。
3. 一次 **DeepMD + LAMMPS CPU 5-step NVT functional smoke**（job `redacted`，
   scheduler `COMPLETED`，MLIPFlow `OK`）。真实 `pair_style deepmd` 完成 5/5 步，
   写出 trajectory、`final.data`、`final.restart`，随后才打印批准的 completion marker；
   bounded fetch 后 checker 复核 LAMMPS 版本、model/input 路径、参数与输出语义。
4. 一次 **MatGL/M3GNet GNNP + LAMMPS CPU 5-step NVT functional smoke**
   （最终成功 job `redacted`，scheduler `COMPLETED`，MLIPFlow `OK`）。真实
   `pair_style gnnp` 加载声明路径下的 MatGL model directory 并完成 5/5 步；
   前一 fresh attempt 的失败被保留并分类为站点 Python 环境缺失，修正 canonical
   site template 后才在 attempt 2 成功。
5. 一次 **LASP 3.6.0 NN 14-atom CPU SSW functional smoke**（最终成功 job
   `redacted`，12 MPI ranks、0 GPU，scheduler `COMPLETED`，MLIPFlow `OK`）。前一
   attempt 暴露 ARC 兼容名称选择 bug 并保留为 `FAIL`；通用修复后，相同科学输入、
   potential、LASP 参数与资源在 fresh attempt 生成/接受/选择 14/14/14 个结构，随后
   bounded fetch 与 scientific checker/collect 完整通过。

**这些记录仍不能被读作**：production ready、VASP 已验证、LAMMPS 科学精度或
平衡态已验证、LASP trajectory numerical parity 已验证、GPU/多节点/restart 已验证、
调度故障恢复已验证，或该站点之外的
任何集群已验证。LAMMPS 证据严格限于 CPU、单节点、单进程、5 步功能性执行；
它证明真实 pair style 与 MLIPFlow 生命周期工作，不产生材料科学结论。

## 已做与未做

| 项目 | 当前事实 |
|---|---|
| local site control plane | 多 cluster `site.yaml` parsing、profile selection、缺失/危险配置失败均有 synthetic fixture 测试；本阶段未读取真实 `~/.mlipflow/site.yaml` |
| remote template library | CPU/GPU Slurm 模板选择、program-family `run.sh`、固定变量合同与不完整模板失败均有 fake library 测试 |
| remote workspace | `<work_root>/<project>/<node>/attempt-XXXX`、attempt 递增、fresh directory、input/output/logs/completion 分层与逐文件 staging 有 mock 测试 |
| core scheduler lifecycle | resolve → render → stage → submit → persist job ID → monitor → inventory/fetch → check → collect 状态逻辑有本地 fixture/mock 测试 |
| scientific adapters | static `dft-labeling.label`、四框架 `mlip-training`、scheduled LASP、ASE MD 与 LAMMPS execute 已接入通用 `ssh-slurm` 合同；真实调度器已验证 LASP NN CPU tiny SSW、DeepMD CPU 训练，以及 DeepMD 与 MatGL/GNNP 两条 LAMMPS CPU functional smoke；scheduler `COMPLETED` 仍不会自动变成科学 `OK` |
| SSH/remote access | 已在一个真实站点执行只读探索与受控写入；写入全部限制在本次新建的独立 MLIPFlow root 内 |
| remote staging | 真实站点已验证：fresh attempt workspace 与逐文件 staging |
| real `sbatch` submission | 已执行基础 CPU tiny job、一次真实 LASP CPU tiny job 的失败 attempt 与一次授权 fresh retry、一次真实 DeePMD 训练 job，以及两次最终成功的 LAMMPS CPU functional smoke；均取得并持久化 job ID，失败 attempt 也保留 lineage |
| real queue monitoring | 已通过 `squeue`/`sacct` 观察到 terminal `COMPLETED` |
| real cancellation | 未执行 |
| remote result fetch + local re-check | 真实站点已验证：bounded allowlist 抓取、completion project/node/attempt/exit 字段校验与 plugin `check`/`collect` |
| scientific program on scheduler | LASP 3.6.0 NN 在 job `redacted` 完成 14-atom tiny SSW 并经 bounded fetch + checker/collect 到 `OK`；DeepMD-kit v3.0.0b1 CPU `dp train` 在 job `redacted` 完成 500 步并到 `OK`；LAMMPS 2 Aug 2023 分别在 job `redacted` 真实执行 `pair_style deepmd`、在 job `redacted` 真实执行 `pair_style gnnp`，两者均完成 5/5 步并到 `OK`。VASP 尚未在调度器上运行 |
| training numerical reproduction | **已验证一次**：与历史 run `split_train/se_e2_a/para0` 的前 6 个 reporting step（0/100/…/500）在 lcurve 全部 7 列上逐位相同，相对差 0.0 |
| GPU / multi-node | **未验证**。这里记录的 LASP、训练和 LAMMPS 验证均为单节点 CPU |
| production scientific job | 未提交。LASP 为 14-atom/10-step tiny smoke，DeePMD 训练为 500 步 bounded validation run，LAMMPS 为 5 步 functional smoke，均不是生产任务 |

因此仍不应使用“生产可用”“GPU/多节点已验证”等表述。可以说明的是：通用 scheduler
合同已在一个真实站点完成端到端 CPU tiny smoke，承载一次真实 LASP SSW、一次真实
DeePMD 训练，并完成两种真实 LAMMPS MLIP interface 的五步功能性闭环。训练的早期
优化轨迹可与历史 run 逐步对照；LASP 与 LAMMPS functional smoke 只证明执行合同，
不证明 SSW numerical parity、模型准确性或 MD 收敛。

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
- completion manifest 的 project/node/attempt 与退出状态字段；
- 允许回收的文件清单及单文件/总大小安全上限；
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
5. 持久化 job ID 与该 attempt 的已批准 plan JSON；
6. 轮询至 terminal state，但不把 `COMPLETED` 直接升级为科学 `OK`；
7. 只回收 allowlist 中且不超过安全上限的小文件；
8. 在本地运行同一固定 plugin 的 `check`/`collect`；
9. 保存匿名化测试记录，不保存站点地址或凭据。

任何真实提交、取消、清理或覆盖都不应由文档审计自动触发。

已完成的真实 CPU tiny smoke 把状态从 `EXTERNAL_VALIDATION_PENDING` 提升为
`SCHEDULER_CONTRACT_VERIFIED_ON_ONE_SITE`，其边界严格限于：一个站点、单节点、
单核、无 module、无科学程序。后续真实 LASP smoke 已独立提升相应科学程序的
functional-integration 证据；本地 fake LASP executable、fake VASP 输出和受限 MPI
argv 测试本身仍然不能改变 HPC 状态。

随后的真实 DeePMD 训练把状态提升为 `SCIENTIFIC_PROGRAM_VERIFIED_ON_ONE_SITE`，
其边界同样严格：一个站点、单节点、单进程 CPU、一个科学程序（DeePMD-kit `dp
train`）、一个 descriptor（`se_e2_a`）、一个数据集（180+180 Li10 systems）、
500 步 bounded run。证据见
[`reports/deepmd_early_training_reproduction.json`](../reports/deepmd_early_training_reproduction.json)。

2026-08-16 的真实 LASP CPU tiny smoke 证明同一生命周期能够承载 site-owned LASP
executable 与 MPI launcher。attempt 1 暴露通用 ARC canonicalization bug 并保留；修复后
仅有的一次授权 retry 使用相同 14-atom NN 输入、potential、LASP 参数和 12-rank CPU
资源，在 job `redacted` 得到 14/14/14 结构，经 terminal observation、bounded fetch、
scientific checker/collect 到 `OK`。匿名化参数、输出路径与 retry lineage 见
[`reports/lasp_cpu_tiny_hpc_smoke.json`](../reports/lasp_cpu_tiny_hpc_smoke.json)。该记录
不含 potential 内容或 site 私有路径，也不建立 SSW numerical parity。

2026-08-16 的两次 LAMMPS CPU functional smoke 进一步证明了同一生命周期可以承载
真实 `pair_style deepmd` 和 AdvanceSoft `pair_style gnnp`/MatGL interface。两者均从
`lammps-prepare` 产生 portable deck，以 site-owned model/interface path 注入执行，
随后经过 scheduler terminal observation、bounded fetch 和 scientific checker/collect
到 `OK`。模型、input manifest、LAMMPS executable 版本和输出路径记录在匿名化报告
[`reports/lammps_cluster_cpu_functional_smokes.json`](../reports/lammps_cluster_cpu_functional_smokes.json)
中。该报告不保存绝对模型路径或权重。

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

它会阻止的主张是：生产级远端执行、真实调度故障恢复、站点性能或生产科学任务
已经通过验证。远端 artifact bounded fetch/check 闭环本身已经在上述真实任务中完成，
但这不自动提升任何科学结论。
