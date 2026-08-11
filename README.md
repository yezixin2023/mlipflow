# MLIPFlow

MLIPFlow 是一个面向机器学习原子势（MLIP）研究的、确定性的工作流层。它把结构生成、势能面采样、DFT 标注、模型训练、基准评估、离子输运、成分筛选和电化学电压组织为显式 DAG，并把状态、审批、产物和来源写入可审计记录。

当前版本是 **0.1.0 alpha**。请勿把它当作已经验证过的生产级科学软件，也不要据此直接作高成本计算或发表结论。

## 当前边界

已实现的基础能力：

- `project.yaml`/JSON 项目读取和无环依赖检查；
- SQLite 持久状态及显式状态转换；
- 严格查询命令与变更命令分离；
- `--dry-run` 计划和精确 `plan_digest` 审批；
- 小型结果 manifest 的离线 replay 与产物指纹；
- 根据项目内基准证据和任务策略进行模型路由；
- local、SLURM、SSH+SLURM 的受控后端边界，以及身份绑定的调度完成清单。

当前科学能力按证据级别明确分开：

- **可执行后处理**：从显式 reference/prediction pairs 重算 MAE、RMSE、Pearson r；从既有 MSD/轨迹计算扩散、电导率与 Arrhenius；从总能序列计算相邻平均嵌锂电压；对既有候选结果作稳定排序；
- **真实证据重放**：六个论文模型的 PES/输运/电压指标、三组历史 speedup、247 个大超胞候选及其 top-k 均可被校验、归一化和来源绑定；
- **LASP/SSW 历史后处理重放**：`lasp-ssw-normalize-replay` 可解析只读 `allstr.arc`，按能量过滤后再按 accepted order stride 选择结构，并可采集 `best.arc`/`md.arc`；代表性初始目录的 6 个 SSW frame 被核对为 5 个能量接受、5 个选中、2 个 `best.arc` 候选，源文件 SHA-256 已核对；这不表示重跑 LASP 或建立势能面数值 parity；
- **真实数值一致性**：历史 MACE/DeepMD 输运后处理以及 247 候选的 ID/数值规范化与排序已有实际对照；
- **本地集成 smoke**：seeded ASE/icet SQS wrapper 已在一个受限的 Li–M–P–S fixture 上真实运行并验证两次确定性输出；该 100-step 运行不是生产 SQS，也不是历史结构 parity；
- **真实 MLIP→输运交接 smoke**：一个外部本地 MACE 模型已实际加载，历史 ASE-MD 源完成 3×10 步 Li3YCl6 MD，trajectory 被原 `ionic_conductivity.py` 接收并生成 MSD/D/σ/Arrhenius，随后通过 Adapter `check/collect`；这些极短轨迹的数值没有科学意义；
- **LASP/SSW 本地执行契约**：`lasp-ssw-execute` 对用户自备 LASP、显式 `lasp_version`、`input.arc`、`lasp.in` 和辅助文件提供 `shell=False` 的本地薄封装，可直接运行；如使用 MPI，必须给出名为 `mpirun`/`mpiexec` 的普通可执行文件路径和 `-np` 数量，路径与内容会进入审批指纹。fake executable contract smoke 已通过，但真实授权 LASP 尚未执行；
- **仍需外部环境**：论文六模型的 fresh benchmark inference、DIRECT、真实 LASP/SSW 执行与科学数值 parity、VASP/DFT、训练、长 MD、RDF/局域结构、top-candidate AIMD/DFT 对照与 unseen quantitative validation；
- **仍未验证**：真实 SSH+SLURM stage/submit/monitor/fetch 闭环。内置科学 Adapter 当前只声明 `local`，没有生产集群可用性主张。

八个插件不会实现或替代 DeepMD、MACE、CHGNet、M3GNet、VASP、LASP、LAMMPS、MAML
或 icet。`replay` 只表示既有证据被严格导入和重新归约，不表示模型、SSW、MD、DFT
或训练被重新运行。完整边界见 `docs/IMPLEMENTATION_STATUS.md`、
`docs/SCIENTIFIC_VALIDATION.md` 与 `docs/MANUSCRIPT_REPRODUCTION.md`。

## 安装

需要 Python 3.9 或更高版本。开发安装：

```bash
cd mlipflow
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

只安装运行时：

```bash
python -m pip install -e .
```

可选 `science` extra 只安装仓库内直接调用的一组通用科学依赖：

```bash
python -m pip install -e '.[science]'
```

外部历史程序仍应在各自经过验证的环境运行。尤其 MAML DIRECT、LASP、pymatgen、各
MLIP 框架和 VASP 不会因安装 MLIPFlow 自动获得、下载或授权；所需版本以对应
`plugins/*/plugin.yaml` 的 dependency/limitation 为准。

`project.yaml` 和 `model_registry.yaml` 可以直接写成 JSON；JSON 是 YAML 的严格子集，因此示例在不依赖 YAML 特性的环境里也易于审阅。

## 快速体验：纯离线 replay

示例覆盖完整 DAG，但不包含模型权重、真实结构/训练数据、VASP 文件、POTCAR、凭据或远程路径。M 子晶格严格使用论文元素 `Mn/Fe/Ni/Cu/Zn`，其余元素为 `Li/P/S`，不含 Co。建议复制到临时目录，避免在版本库内生成 `.mlipflow/` 状态：

```bash
DEMO_ROOT="$(mktemp -d)"
cp -R examples/high_entropy_sulfide "$DEMO_ROOT/"
DEMO_PROJECT="$DEMO_ROOT/high_entropy_sulfide"
```

查询命令在初始化前也可查看静态预览，而且不会创建状态文件：

```bash
mlipflow --project "$DEMO_PROJECT" list
mlipflow --project "$DEMO_PROJECT" status
mlipflow --project "$DEMO_PROJECT" inspect structure-replay
mlipflow --project "$DEMO_PROJECT" doctor
```

初始化是一次显式写操作：

```bash
mlipflow --project "$DEMO_PROJECT" init
```

先生成不落盘的计划：

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay --dry-run
```

从输出复制完整的 `sha256:...`，仅批准这一份计划：

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay \
  --approve 'sha256:<粘贴上一条命令给出的摘要>'
```

该命令只读取一个小型 result JSON 和 CSV，记录指纹与运行 manifest，不会生成结构、启动训练/DFT/MD、访问网络或提交作业。完整示例按以下 DAG 逐层 replay：

```text
structure → PES sampling → DFT labeling → MLIP training → benchmark → transport
                                                               └→ composition screening
                                                                    → high-fidelity validation
                                                                    → voltage
```

每次节点成功后，仍需对 `advance --dry-run` 的摘要进行单独批准，依赖节点才会从 `WAIT` 变为 `READY`；查询不会自动推进。完整命令顺序见 `examples/high_entropy_sulfide/README.md`。

## 基于证据的模型路由

示例声明 DeepMD、MACE、CHGNet 三种 metadata-only 训练原型，以及一个 M3GNet unseen-transfer family；不分发任何权重。同一份 `model_registry.yaml` 和 `replay/model_benchmark.csv` 故意让 DeepMD fixture 的输运指标更好、CHGNet fixture 的电压指标更好：

```bash
mlipflow --project "$DEMO_PROJECT" route \
  --task ionic-transport \
  --elements Li Mn Fe Ni Cu Zn P S \
  --scenario synthetic-high-entropy-sulfide-v1

mlipflow --project "$DEMO_PROJECT" route \
  --task electrochemical-voltage \
  --elements Li Mn Fe Ni Cu Zn P S \
  --scenario synthetic-high-entropy-sulfide-v1
```

预期分别选择 `deepmd-demo` 和 `chgnet-demo`。这是示例基准数据与任务权重的结果，不是核心代码中的模型偏好，也不代表这些模型在真实体系中一定更好。

## 真实论文证据复现

`examples/high_entropy_sulfide_reproduction/` 是独立于上面 synthetic DAG 的紧凑
论文证据包。它不包含 Word 原文、模型权重、完整训练数据、轨迹或 VASP 大文件，
也不访问网络；只重放可分享的小型转录/派生证据并重新计算透明 reduction：

```bash
cd examples/high_entropy_sulfide_reproduction
PYTHONPATH=../../src python3 reproduce.py --check
```

验收会核对证据 SHA-256、六模型 registry、benchmark 标准四件套、路由决策、
247/247 screening、top-1 历史输运连接、unseen claim-level 边界和最终报告。
当前证据使路由得到：static PES → `deepmd-dpa2`、ionic transport →
`deepmd-se_atten_v2`、voltage → `chgnet`；registry 没有 `recommended_tasks`，核心也
没有模型品牌条件分支。该结果是 `REPLAY_VERIFIED`，不是 fresh prediction。

## CLI 安全语义

严格只读命令是：

- `list`
- `status`
- `json`
- `inspect`
- `logs`
- `route`
- `doctor`

它们不会创建数据库、写文件、导入或执行插件适配器、提交/取消作业、重试、启动后台进程或自动推进 DAG。若项目已初始化，查询以 SQLite `mode=ro` 打开状态库。

可能改变状态或调用后端的命令是 `init`、`run`、`advance`、`retry` 和 `stop`。除 `init` 外，先运行 `--dry-run`，再把输出中的精确 `plan_digest` 传给 `--approve`；项目、插件实现、命令文件或计划变化后旧摘要失效。初始化后若项目配置变化，变更命令会拒绝静默漂移并要求显式迁移或新状态库。`advance` 只同步依赖/已提交调度作业的状态，不会自动执行 READY 节点。

## 两类扩展，不要混淆

`.agents/skills/` 中有九个 Agent Skill：

1. `mlip-workflow`
2. `high-entropy-structure`
3. `pes-sampling`
4. `dft-labeling`
5. `mlip-training`
6. `mlip-benchmark`
7. `ionic-transport`
8. `composition-screening`
9. `electrochemical-voltage`

Skill 是给 LLM/IDE agent 的监督说明：帮助它规划、检查证据、解释状态、识别风险并请求授权。Skill 不是计算实现，也不应自行拼接远程命令来绕过 MLIPFlow。

`plugins/` 中有八个同名科学阶段插件（不含总控 `mlip-workflow`）。计算插件提供确定性的输入/输出、验证、计划、采集和 replay 契约；真正的数值程序必须通过声明过的适配器或外部后端运行。插件 Python 属于与构建脚本同等级的**受信任代码**：`run --dry-run` 会加载所选 Adapter，因此不要对不可信第三方插件运行计划命令。模型家族属于 `mlip-training` 的后端选择，而不是额外的 Agent Skill。

## 后端与凭据边界

- `local`：以 argv 列表运行本地外部程序，不使用 shell 字符串，只继承白名单环境变量；
- `slurm`：显式批准后调用本机 `sbatch`/`scancel`；
- `ssh-slurm`：只接受简单的 SSH config 别名，再在远端调用调度器。

调度器返回 `COMPLETED` 不会直接成为科学 `OK`：完成清单必须是项目内普通文件，明确写入 `project_id/node_id/run_id/attempt/plugin_id/plan_digest`，并在批准与落库之间保持同一指纹。路由也默认只让完整 SHA-256 验证过的 benchmark evidence 参与排名。

项目配置不得嵌入密码、私钥或令牌。SSH 主机、认证和跳板配置由用户的 `~/.ssh/config` 或站点安全设施管理；插件 manifest 只描述能力，不保存凭据。远端目录、资源和模型路径必须是项目/站点配置的一部分，不能硬编码进科学算法。

## 开发与许可

测试：

```bash
python -m pytest
```

贡献前请阅读 `CONTRIBUTING.md` 和 `SECURITY.md`。MLIPFlow 以 Apache License 2.0 发布；外部数值程序、模型、数据集和势函数各自受其许可证约束，安装 MLIPFlow 不会自动授予它们的使用权。
