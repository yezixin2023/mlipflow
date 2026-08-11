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

1. 在 `CODEBASE_INVENTORY.md` 中记录来源和 A–E 分类；
2. 先把硬编码路径/参数变成显式 CLI；
3. 通过 argv 列表调用，永远 `shell=False`；
4. 固定 cwd 和允许的环境变量，不继承/记录敏感变量；
5. 输入 schema 校验通过后才准备目录；
6. 进程退出 0 后仍运行科学完成判据；
7. 生成结构化 result manifest/JSON/CSV，Agent 不解析任意 stdout；
8. 与既有小样例输出做 parity test；
9. 原脚本含 `rm/mv`、无限轮询或隐式提交时必须拆分，不能原样调用。

内置 adapter 目前只允许 `local`。不要在 manifest 中声明 `slurm`/`ssh-slurm`：调度回收侧尚未重跑固定 adapter 的 `check/collect`，核心会拒绝 scheduled adapter。需要调度器时，先使用显式用户 SLURM 脚本和身份绑定的 completion contract；远端 staging 完成后再扩展插件能力。

LASP/SSW 是这条边界的一个具体例子：execute 只包装用户自备 executable，要求显式
版本、ARC/`lasp.in`/辅助输入与 fresh attempt，使用 `shell=False`；可选 MPI 只接受
显式、可指纹化的 `mpirun`/`mpiexec` 普通可执行文件路径与 `-np N` 的受限 argv，不等于 SLURM 支持。normalize-replay 只解析已有
archive。fake executable smoke 只能证明 contract，不能写成真实 LASP 或科学 parity。

## 科学结果要求

结果应按能力记录适用字段：单位、归一化方式、样本和分量数量、随机 seed、split、温度、时间步、平衡段、拟合窗、体积、移动粒子、DFT 设置引用、模型/数据集指纹和软件版本。

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
- 为小文件计算 SHA-256，为大文件记录轻量 metadata fingerprint；
- 写当前项目的新 run manifest（仅在显式批准的 `run` 中）。

Replay 不可以：启动数值程序、提交作业、复制大数据/权重、修改来源产物，或把既有结果称为重新计算。

结果 manifest 必须显式为 `OK`，使用相对、无 `..` 的普通文件引用，且不得通过符号链接逃逸。调度完成结果还必须包含 `project_id/node_id/run_id/attempt/plugin_id/plan_digest` 并与批准计划一致。

## 测试清单

- manifest/schema、重复 ID 和 API 版本；
- adapter 无 import-time 副作用；
- plan/dry-run 零写入；
- argv 注入、非零退出和缺失产物；
- 完成判据的成功/失败/不完整 fixture；
- manifest 通过 `run-manifest.schema.json`；
- retry 新 attempt、不覆盖；
- fake local、SLURM scheduler reconciliation 与 SSH argv/path 边界；
- 科学单位和已知小样例 parity。

运行：

```bash
python -m pytest tests/test_plugin_manifests.py
```
