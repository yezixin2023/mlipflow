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
4. `init/run/advance/retry/stop` 会改变状态或外部系统。启动声明为需审批的昂贵或外部计算前，先用 `run --dry-run` 审查执行语义，再用 `--approve` 表示明确批准。scheduler observation、bounded fetch、check/collect 不需要第二次审批；`retry` 只创建 fresh attempt；显式 `stop NODE` 本身就是取消意图。
5. 以下行为总要单独获得明确意图：删除或清理数据、覆盖模型/数据集、破坏性重跑、远端 staging、DFT/AIMD、长时间 MD、训练/微调和大规模筛选。显式 `stop NODE` 可取消该节点的已知作业，但不得扩大到其他作业。
6. 不要把密码、私钥路径、token、POTCAR、模型权重或私有大数据写入仓库。SSH 后端只引用用户 `~/.ssh/config` 中的 profile 名。
7. `retry` 必须创建新 attempt 并保留旧结果。不要用 `rm` 模拟 retry。
8. 调度器显示 `COMPLETED` 不等于科学结果 `OK`；只有插件完成判据和输出 schema 均通过才可标为 `OK`。
9. 不确定的科学参数、单位、拟合窗、随机种子、数据 split 或参考能必须报告并等待，禁止猜测。
10. Python adapter 是受信代码，`run --dry-run` 也会加载它；不要对来源不明的第三方插件生成计划。`dft-labeling.label` static 已接入通用 SSH-SLURM profile/template/workspace 合同；其他内置 adapter 只支持 local。Agent 只提出科学任务与抽象资源，不猜 SSH host、partition、module、executable、template root 或 work root，也不得用 `submit_script`/`remote_cwd` 绕过核心 staging/fetch/check 限制。
11. 同一个实际站点必须复用其 canonical `remote_template_root` 和 `work_root`。不得按插件、框架、target、partition 或验证轮次创建 sibling template/work roots；缺少 template family 时，只能先核对已有内容，再在 canonical template root 下新增相应 family 或 scheduler 子目录，禁止静默覆盖已有模板。

## 建议监督流程

1. 运行 `mlipflow json` 获取持久状态。
2. 用对应仓库 Skill 理解任务和科学注意事项。
3. `mlipflow inspect NODE` 检查插件契约、输入和已知限制。
4. 如涉及模型选择，运行 `mlipflow route --task ...`，读取候选、淘汰原因和 benchmark 指标贡献；不要凭图或模型品牌选择。
5. 对需审批的 `run` 先运行 `--dry-run`，逐项说明执行语义、规模、后端、资源、实际输入与 staged scripts，再请求布尔批准。普通 `advance`/`retry` 和显式 `stop NODE` 不需要额外批准字段。
6. 执行后按退出码和结构化 JSON 判断，不把非零退出码描述为成功。
7. 对 FAIL 先读 `logs` 和 manifest；诊断后提出 retry 或参数修正，不能自动重试。

## 回放模式

回放只允许读取、验证并按路径引用现有小型 manifest/产物。不得启动数值程序、提交作业、复制大型数据或修改来源文件。回放结果必须明确标注为“既有结果的结构化采集”，不能称为重新计算或独立科学验证。

## Skills 与 plugins

- `.agents/skills/`：教 Agent 何时使用能力、需要什么输入、怎样解释结果和何时请示；不含主数值实现。
- `plugins/`：声明确定性计算单元、依赖、后端、完成判据、重试和回放契约。

不要把 DeepMD、CHGNet、单个 MSD 拟合、SLURM 或 OUTCAR 解析各自提升为顶层 Agent Skill。
