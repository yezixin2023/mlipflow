# MLIPFlow Agent 操作规范

本文件只适用于当前 `mlipflow/` 仓库。Agent 是监督者；MLIPFlow 与其计算插件才是确定性执行层。

## 不可跨越的边界

1. 数值工作必须通过 MLIPFlow 计算插件或用户明确指定的外部科学程序完成。不要让 LLM 估算、补写或臆造能量、力、扩散系数、电导率、电压或模型排名。
2. 优先调用 `mlipflow`，不要绕过状态库直接 `sbatch`、`scancel`、删除运行目录或覆盖模型。
3. 以下命令严格只读，可直接运行：

   ```text
   mlipflow list
   mlipflow status
   mlipflow json
   mlipflow inspect
   mlipflow logs
   mlipflow route
   mlipflow doctor
   ```

   它们不得提交、拉取、重试、取消、修改输入、写状态或触发自动推进。
4. `init/run/advance/retry/stop` 会改变状态或外部系统。`run/advance/retry/stop` 必须先执行 `--dry-run`，向用户展示计划、规模、后端、资源、输出和覆盖风险，再以完全匹配的 `--approve sha256:...` 执行。
5. 以下行为总要单独获得明确批准：取消作业、删除或清理数据、覆盖模型/数据集、破坏性重跑、远端 staging、DFT/AIMD、长时间 MD、训练/微调和大规模筛选。
6. 不要把密码、私钥路径、token、POTCAR、模型权重或私有大数据写入仓库。SSH 后端只引用用户 `~/.ssh/config` 中的 profile 名。
7. `retry` 必须创建新 attempt 并保留旧结果。不要用 `rm` 模拟 retry。
8. 调度器显示 `COMPLETED` 不等于科学结果 `OK`；只有插件完成判据和输出 schema 均通过才可标为 `OK`。
9. 不确定的科学参数、单位、拟合窗、随机种子、数据 split 或参考能必须报告并等待，禁止猜测。
10. Python adapter 是受信代码，`run --dry-run` 也会加载它；不要对来源不明的第三方插件生成计划。除 `dft-labeling.label` 已实现的受控单结构 static SSH-SLURM 合同外，内置 adapter 只支持 local；不得手工改 manifest 或绕过核心 staging/fetch/check 限制。

## 建议监督流程

1. 运行 `mlipflow json` 获取持久状态。
2. 用对应仓库 Skill 理解任务和科学注意事项。
3. `mlipflow inspect NODE` 检查插件契约、输入和已知限制。
4. 如涉及模型选择，运行 `mlipflow route --task ...`，读取候选、淘汰原因和 benchmark 指标贡献；不要凭图或模型品牌选择。
5. 对写操作先运行 `--dry-run`，逐项向用户说明，再请求批准。
6. 执行后按退出码和结构化 JSON 判断，不把非零退出码描述为成功。
7. 对 FAIL 先读 `logs` 和 manifest；诊断后提出 retry 或参数修正，不能自动重试。

## 回放模式

回放只允许读取、验证、指纹化并引用现有小型 manifest/产物。不得启动数值程序、提交作业、复制大型数据或修改来源文件。回放结果必须明确标注为“既有结果的结构化采集”，不能称为重新计算或独立科学验证。

## Skills 与 plugins

- `.agents/skills/`：教 Agent 何时使用能力、需要什么输入、怎样解释结果和何时请示；不含主数值实现。
- `plugins/`：声明确定性计算单元、依赖、后端、完成判据、重试和回放契约。

不要把 DeepMD、CHGNet、单个 MSD 拟合、SLURM 或 OUTCAR 解析各自提升为顶层 Agent Skill。
