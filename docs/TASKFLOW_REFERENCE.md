# Taskflow 参考架构与 MLIPFlow 取舍

## 调研范围

本文件只记录对同级原项目 `taskflow` 的只读分析。MLIPFlow 是全新的独立项目；不会导入、修改或依赖 taskflow 的内部文件。

## Taskflow 做了什么

Taskflow 是面向 VASP 与 SLURM 的多材料、多步骤工作流管理器。它把以下职责组合在一个单文件 Python CLI 中：

- 从本地材料目录、项目配置和 `skill/*/skill.yaml` 发现工作流；
- 生成远端输入，调用 `sbatch`，查询 `squeue`，取消或重跑作业；
- 依据调度器状态和输出文件判据推导步骤状态；
- 按依赖关系推进步骤，并将状态以表格或 JSON 暴露给 Agent；
- 用 `AGENTS.md` 约束 LLM 的观察、诊断、确认和操作边界。

核心不是 Python 包，而是 `versions/v1.0/tf` 单文件可执行程序。领域逻辑位于声明式 skill 清单、生成脚本、检查器和模板中。远端不安装 taskflow：本地程序将采集器、检查器和生成资源编码后经 SSH 传入远端 Python 进程。

## 值得保留的设计

1. **驱动与领域逻辑分离**：主程序负责状态、调度和数据移动，skill 负责步骤、输入生成和完成判据。
2. **结构化机器接口**：`tf json`、原子 CLI 命令和退出码适合 tool-using Agent。
3. **远端低侵入**：远端无需安装中央服务或数据库。
4. **批量状态采集**：按 `(host, work_dir)` 分组，一次 SSH 获取调度器与文件状态，避免每个材料独立连接。
5. **文件系统即科研事实来源**：结果仍保留为普通科研文件，不被专有数据库吞没。
6. **LLM 在监督层**：LLM 不进入数值内核；策略由自然语言规则约束。

## 不直接继承的设计

### 查询与写操作混合

当前 taskflow 的 `status` 可自动拉取并自动推进，且通用命令入口可能启动后台 watch。即使 `json` 或 `list` 本身只展示，也可能间接产生副作用。MLIPFlow 采用命令/查询分离：

- `list/status/json/inspect/logs/route/doctor` 只读；
- 查询以 SQLite `mode=ro` 和 `PRAGMA query_only=ON` 打开状态库；
- 查询不得建库、迁移、写事件、拉回文件、提交、取消、重试或启动后台进程；
- 实时调度器信息只能形成内存 overlay，不能隐式持久化。

### 状态由目录即时猜测

Taskflow 主要由作业和文件判据即时推导状态。MLIPFlow 会显式持久化运行、attempt、job ID、远端目录、依赖、时间、重试次数、产物引用和事件；调度器 `COMPLETED` 仍需插件完成判据确认后才是 `OK`。

### 单文件函数集合

Taskflow 的单文件交付适合超算端零安装，但不利于跨训练框架、数据谱系、模型注册和严格测试。MLIPFlow 使用正常 Python 包，按状态、服务、插件、执行后端、manifest、registry、routing 和 provenance 分层。

### 提示词承担过多安全责任

Taskflow 的破坏性门控主要由 `AGENTS.md` 约束，有 shell 权限的 Agent 仍可绕开。MLIPFlow 将安全边界实现到代码：

- 昂贵操作先用 dry-run 展示计划，再以显式 `--approve` 确认；
- `stop` 始终显式确认；
- 外部命令使用 argv 与 `shell=False`；
- SSH 只引用用户配置 profile，仓库不保存密码、私钥路径或 token；
- retry 创建新 attempt，绝不覆盖旧结果。

## 已发现的文档漂移

这些问题是新项目测试和文档同步机制的反例：

- 现行类型是 `band`，旧 `AGENTS.md` 仍示例 `bd`；
- 当前 `retry` 只重新生成、不提交，旧文档描述为“重交”；
- 裸 `tf` 当前只显示版本/提示，旧文档称其为状态总表；
- 开发文档称没有 `_common`，实际主程序和技能均支持或引用它；
- “无状态”并非完全无状态，仍有取消记录和拉取戳记。

MLIPFlow 因此要求：CLI 帮助、JSON Schema、插件清单和用户文档由测试互相校验；实现状态逐阶段记录在 `docs/IMPLEMENTATION_STATUS.md`。

## LLM 接口结论

Taskflow 本身没有 OpenAI、Anthropic、Gemini、LangChain 或 MCP 模型调用。其 LLM 接口实质是：

```text
自然语言监督规则 + 结构化 JSON 观察 + 原子 CLI + 退出码
```

因此适用对象不是某个品牌模型，而是任何能稳定调用 shell 工具、解析 JSON、保留状态差异并执行确认门控的 Agent。纯聊天模型没有终端、SSH 或定时运行能力，只能解释和提出建议。

MLIPFlow 继续采用模型无关接口，并把九个 Agent Skills 与确定性计算插件彻底分开：Skills 教 Agent 如何监督，插件定义机器可验证的输入、计划、执行和结果。
