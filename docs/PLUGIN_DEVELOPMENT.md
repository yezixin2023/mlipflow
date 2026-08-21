# 计算插件开发

## 插件不是 Agent Skill

计算插件位于 `plugins/<id>/`，必须是确定、可验证的科学执行单元。`.agents/skills/` 只教 Agent 何时调用和怎样解释，不能替代插件实现。

## 最小目录

```text
plugins/my-plugin/
  plugin.yaml      # JSON 语法也是合法 YAML，便于最小环境静态读取
  adapter.py       # 仅显式执行路径才会 import
```

`plugin.yaml` 必须通过 `schemas/plugin.schema.json`，至少声明：

- `id/name/description/version/api_version`；
- `inputs/parameters/outputs`；
- Python/外部程序依赖，且第三方程序不打包；
- 实际验证过的后端能力（不能把核心存在的 backend 误写成插件已支持）；
- 完成判据和科学失败状态；
- retry 来源状态、新 attempt 策略和次数；
- 是否昂贵/破坏性/联网以及批准要求；
- replay 能力和行为。

不要用空清单暗示已实现。当前内置插件均有 `adapter-ready`/`implemented` 的本地薄适配器和契约测试；这不等于外部科学程序已经打包，也不等于真实数值 parity 已完成。只有 adapter、完成判据、fixture 和 parity test 都通过后，才能声明相应状态。

## Adapter 契约

```python
class Adapter:
    def validate(self, context): ...
    def plan(self, context): ...       # 纯函数，不写文件、不联网、不提交
    def prepare(self, context, plan): ...
    def check(self, context): ...      # 调度成功不等于科学成功
    def collect(self, context): ...
    def replay(self, context): ...     # 只解析/引用既有产物
```

查询命令不会 import adapter；`run --dry-run` 会 import 所选 adapter，所以插件是受信代码。Adapter 顶层及 `validate/plan` 不得创建目录、读取大数据、导入 GPU 框架、修改环境、访问网络或启动子进程。第三方插件在没有独立 sandbox 的当前版本中不应被当成不可信数据加载。

## 包装外部科学程序

遵守 wrap-before-rewrite：

1. 在 `plugin.yaml`、相关文档或 pull request 中记录来源、版本、许可证/再分发边界、修改范围和验证证据；
2. 先把硬编码路径/参数变成显式 CLI；
3. 通过 argv 列表调用，永远 `shell=False`；
4. 固定 cwd 和允许的环境变量，不继承/记录敏感变量；
5. 输入 schema 校验通过后才准备目录；
6. 进程退出 0 后仍运行科学完成判据；
7. 生成结构化 result manifest/JSON/CSV，Agent 不解析任意 stdout；
8. 与既有小样例输出做 parity test；
9. 原脚本含 `rm/mv`、无限轮询或隐式提交时必须拆分，不能原样调用。

内置 adapter 默认只允许 `local`。若要接入 `ssh-slurm`，新 adapter 只能声明科学合同：
`schema_version: 3`、明确的 `execution_model`、安全 `template_family`、显式 `staged_files` 和有单文件上限的
`fetch_outputs`。adapter 不得声明 host、partition、module、executable path、launcher、
remote work root 或完整 sbatch；这些属于用户本地 cluster profile 与远端 template library。
核心从持久 attempt state 解析 fresh workspace、渲染脚本、提交，并在普通 `advance` 中
bounded fetch，再加载 pinned plugin 执行 `check/collect`。不要仅在 manifest 中增加 backend；
缺少完整 scientific contract 的 scheduled adapter 会被核心拒绝。本地 `slurm` adapter
仍未开放。

`execution_model` 当前只能是：

- `single-python`：一个 Python 进程；`resources.cpus` 是该进程的线程预算。Slurm 必须为
  `--ntasks=1` 与 `--cpus-per-task={{CPUS}}`。
- `mpi`：`resources.cpus` 是 MPI task/rank 数。Slurm 必须为
  `--ntasks={{CPUS}}`，且不得再次把 `{{CPUS}}` 用作 `cpus-per-task`。

核心按 execution model 选择不同的 `slurm/<execution-model>/{cpu,gpu}.sbatch` 并在 staging
前检查上述映射。旧 schema v2 与未分型的 `slurm/{cpu,gpu}.sbatch` 不再支持。

LASP/SSW 是这条边界的一个具体例子：execute 只包装用户自备 executable，要求显式
版本、ARC/`lasp.in`/辅助输入与 fresh attempt，使用 `shell=False`；可选 MPI 只接受
显式的 `mpirun`/`mpiexec` 普通可执行文件路径与 `-np N` 的受限 argv，不等于 SLURM 支持。normalize-replay 只解析已有
archive。fake executable smoke 只能证明 contract，不能写成真实 LASP 或科学 parity。

## 科学结果要求

结果应按能力记录适用字段：单位、归一化方式、样本和分量数量、随机 seed、split、温度、时间步、平衡段、拟合窗、体积、移动粒子、DFT 设置、模型/数据集路径和软件版本。

常见错误：

- 把 scheduler `COMPLETED` 当成 `OK`；
- 接受未收敛 VASP 输出；
- 混淆 total/per-atom energy；
- 未声明 stress 符号/张量顺序；
- 以 `D=slope/6` 处理非三维各向同性扩散；
- 用 Nernst–Einstein 时不声明独立载流子/Haven ratio 假设；
- 让模型排名从图或 Agent 主观判断产生。

## Replay 规则

Replay 可以：

- 读取小型结果 manifest；
- 验证其中明确列出的现有文件；
- 记录来源路径、参数、软件版本和已有输出；
- 写当前项目的新 run manifest（仅在显式批准的 `run` 中）。

Replay 不可以：启动数值程序、提交作业、复制大数据/权重、修改来源产物，或把既有结果称为重新计算。

portable replay/result manifest 必须显式为 `OK`，使用相对、无 `..` 的普通文件引用，且
不得通过符号链接逃逸。远端 `completion.json` 只证明进程终止，必须至少绑定
`project_id/node_id/attempt/status/exit_code`；core 保留该 attempt 的 approved plan，
并且仍须执行科学 `check/collect`。

## 测试清单

- manifest/schema、重复 ID 和 API 版本；
- adapter 无 import-time 副作用；
- plan/dry-run 零写入；
- argv 注入、非零退出和缺失产物；
- 完成判据的成功/失败/不完整 fixture；
- manifest 通过 `run-manifest.schema.json`；
- retry 新 attempt、不覆盖；
- synthetic multi-cluster site config、fake template library、fresh workspace/stage plan、
  scheduler reconciliation 与 SSH argv/path 边界；
- 科学单位和已知小样例 parity。

运行：

```bash
python -m pytest tests/test_plugin_manifests.py
```
