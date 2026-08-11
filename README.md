# MLIPFlow

MLIPFlow 是一个面向机器学习原子势（MLIP）研究的确定性工作流层。它负责把结构生成、PES 采样、DFT 标注、模型训练、基准评估、离子输运、成分筛选和电化学电压组织成显式 DAG，并把计划、审批、状态、版本、输入、输出、模型、指标和来源写入可审计记录。

当前版本：**0.1.0 alpha**。

这份 README 是 **最终用户使用手册**，不是某一台工作站或某一个集群的私人配置记录。任何用户都应该能够按照这里的步骤完成：

```text
安装 MLIPFlow
-> 本地离线验证
-> 配置所需科学环境
-> 本地 CPU/GPU smoke
-> 配置 SSH
-> 识别集群 CPU/GPU/SLURM 参数
-> 在计算节点验证环境
-> 用 SLURM 包裹 MLIPFlow 执行当前 local Adapter
-> 配置 VASP / POTCAR / LAMMPS / LASP
-> 做 tiny scientific validation
-> 扩大到真实计算
```

---

## 0. 先读这一节：README、用户配置和代码边界

### 0.1 README 中的 `<...>` 是变量，不是仓库配置

本文会使用：

```text
<SSH_ALIAS>
<LOGIN_HOST>
<USERNAME>
<CPU_PARTITION>
<GPU_PARTITION>
<ACCOUNT>
<QOS>
<GPU_RESOURCE_DIRECTIVE>
<CONDA_ROOT>
<MPI_MODULE>
<CUDA_MODULE>
<VASP_MODULE>
<LAMMPS_MODULE>
<PROJECT_ROOT>
<SCRATCH_ROOT>
```

这些值 **永远不应该替换后再提交回 README**。

正确流程是：

```text
README 告诉用户需要什么参数
-> 用户从自己的集群文档/命令查询真实值
-> 用户把真实值写进自己的 SSH 配置、sbatch、wrapper 或 project.yaml
-> 用户执行验证命令
-> README 保持通用
```

### 0.2 只有代码真实读取的文件才称为“MLIPFlow 配置”

当前主要产品配置是：

```text
project.yaml
model_registry.yaml
plugins/*/plugin.yaml
.ml ipflow/state.sqlite3      # 运行状态，由程序管理；实际目录名见下文
```

实际状态目录是：

```text
.mlipflow/state.sqlite3
.mlipflow/runs/<node>/attempt-<N>/
```

`project.yaml` 支持的顶层字段由 `schemas/project.schema.json` 定义，包括：

```text
project
plugin_paths
locations
backend_profiles
model_registry
state
workflow
routing
fingerprints
safety
```

README 不会再把一个代码没有读取的 `site.local.yaml` 描述成正式产品配置。

### 0.3 不把科学参数猜成默认值

README 可以告诉你 **参数在哪里配置、怎样记录、怎样检查**，但不会替你猜：

```text
POTCAR label
ENCUT
KPOINTS 密度
VASP NCORE/KPAR
LAMMPS timestep
训练超参数
MSD 拟合窗口
生产 MD 时长
CPU/GPU 数量
walltime
```

这些值必须来自你的研究方案、历史计算、论文复现目标、收敛测试或集群 benchmark。

---

## 1. 当前实现状态：什么能直接跑，什么还不能

这是理解后面所有集群配置的关键。

| 能力 | 当前状态 |
|---|---|
| 本地 `local` backend | 可执行 |
| replay 工作流 | 可执行 |
| 内置科学 Adapter | 当前都只声明 `local` |
| core `slurm` backend | 已实现 `sbatch/squeue/sacct/scancel` 抽象 |
| core `ssh-slurm` backend | 已实现 SSH alias、远端 scheduler 查询/提交/fetch 基础能力 |
| 科学 Adapter 直接 `backend: slurm` | **当前被核心显式阻止** |
| 科学 Adapter 直接 `backend: ssh-slurm` | **当前被核心显式阻止** |
| 自动 remote stage -> submit -> monitor -> fetch -> scientific check | **尚未完成真实 HPC 闭环验证** |
| 在 Slurm 分配到的计算节点中运行 `backend: local` | 当前最实际的集群科学执行方式 |

当前仓库状态明确记录为：

```text
REAL_HPC_INTEGRATION = EXTERNAL_VALIDATION_PENDING
```

因此：

- **现在可以在集群 CPU/GPU 计算节点上真正运行科学 Adapter**；
- 但当前方法是让 SLURM 先分配资源，然后 MLIPFlow 在该计算节点内以 `backend: local` 执行；
- 不能把科学节点直接改成 `backend: slurm` 或 `backend: ssh-slurm` 来绕过核心保护；
- 未来 first-class scheduler reconciliation 完成后，才能把 scheduler 提交和科学 `check/collect` 完整合并进 MLIPFlow backend。

---

## 2. MLIPFlow 的科学工作流

```text
structure generation
        |
        v
PES sampling / representative selection
        |
        v
DFT labeling
        |
        v
MLIP training
        |
        v
benchmark / model evidence
        |
        +---------------------+--------------------+
        |                     |                    |
        v                     v                    v
ionic transport      composition screening   voltage analysis
                              |
                              v
                    high-fidelity validation
```

当前八个科学插件：

| 插件 | 操作 | 主要依赖 | 当前 backend |
|---|---|---|---|
| `high-entropy-structure` | `generate-sqs` | ASE + icet | `local` |
| `pes-sampling` | `direct-select` / `lasp-ssw-execute` / `lasp-ssw-normalize-replay` | MAML/ASE/pymatgen 或用户 LASP | `local` |
| `dft-labeling` | `label` | 用户 DFT wrapper + VASP | `local` |
| `mlip-training` | `train` / `finetune` | 用户训练 wrapper + DeepMD/M3GNet/CHGNet/MACE | `local` |
| `mlip-benchmark` | benchmark normalize/evaluate | 内置或用户 prediction wrapper | `local` |
| `ionic-transport` | `analyze-existing` / bounded `md-smoke-and-analyze` | trajectory/MSD + 可选 ASE calculator | `local` |
| `composition-screening` | deterministic top-k | 已有候选和指标 | `local` |
| `electrochemical-voltage` | energy -> voltage / replay | 已有总能量序列 | `local` |

更严格的实现状态见：

- `docs/IMPLEMENTATION_STATUS.md`
- `docs/SCIENTIFIC_VALIDATION.md`
- `docs/HPC_VALIDATION.md`
- `docs/CODEBASE_INVENTORY.md`

---

## 3. 本地和集群到底需要哪些环境

不要把所有软件塞进一个 Python 环境。

推荐思路：**一个控制环境 + 按科学框架拆分的执行环境 + 站点外部程序**。

### 3.1 最小控制环境

用于：

```text
mlipflow CLI
project.yaml 解析
状态数据库
plan/digest/approval
replay
轻量 post-processing
```

建议名称：

```text
mlipflow-control
```

要求：

```text
Python >= 3.9
PyYAML >= 6.0
```

推荐 Python 3.11。

### 3.2 科学环境

推荐分开：

```text
mlipflow-sqs
mlipflow-direct
mlipflow-dft
mlipflow-mace-cpu
mlipflow-mace-gpu
mlipflow-deepmd-cpu
mlipflow-deepmd-gpu
mlipflow-chgnet-cpu
mlipflow-chgnet-gpu
mlipflow-m3gnet-cpu
mlipflow-m3gnet-gpu
mlipflow-analysis
```

不是每个用户都需要全部创建。

### 3.3 本地电脑

最低只需要：

```text
mlipflow-control
```

要运行 SQS：

```text
mlipflow-control
mlipflow-sqs
```

要本地 CPU 测试 MACE：

```text
mlipflow-control
mlipflow-mace-cpu
```

本机有 NVIDIA GPU 才需要：

```text
mlipflow-mace-gpu
mlipflow-deepmd-gpu
...
```

### 3.4 集群 CPU

集群共享文件系统中通常需要：

```text
mlipflow-control
所需 CPU 科学环境
MPI（如果科学程序需要）
VASP/LASP/LAMMPS CPU build（如果任务需要）
SLURM client commands
```

### 3.5 集群 GPU

在 CPU 环境基础上增加：

```text
NVIDIA driver                 # 集群管理员提供
CUDA runtime/toolkit          # 由站点 module 或 Python wheel 方案决定
GPU-compatible PyTorch
GPU-compatible MLIP framework
GPU-compatible LAMMPS build   # 只有 LAMMPS GPU MD 时需要
```

**不要在 Conda 环境里安装或替换计算节点的 NVIDIA kernel driver。**

---

## 4. 安装 MLIPFlow 控制环境

在仓库根目录：

```bash
conda create -n mlipflow-control python=3.11 -y
conda activate mlipflow-control
python -m pip install --upgrade pip
python -m pip install -e .
```

开发者额外安装：

```bash
python -m pip install -e '.[dev]'
```

仓库通用科学 extra：

```bash
python -m pip install -e '.[science]'
```

当前 `science` extra 是：

```text
numpy >= 1.23
ase >= 3.22
icet >= 2.0
pandas >= 1.5
plotly >= 5
```

检查：

```bash
python --version
mlipflow --version
python -m pytest
```

---

## 5. 第一次运行：先完成纯离线闭环

仓库自带一个 synthetic replay 示例，不需要 VASP、LAMMPS、LASP、GPU 或网络。

### 5.1 复制示例

```bash
DEMO_ROOT="$(mktemp -d)"
cp -R examples/high_entropy_sulfide "$DEMO_ROOT/"
DEMO_PROJECT="$DEMO_ROOT/high_entropy_sulfide"
```

### 5.2 只读检查

```bash
mlipflow --project "$DEMO_PROJECT" list
mlipflow --project "$DEMO_PROJECT" status
mlipflow --project "$DEMO_PROJECT" doctor
```

### 5.3 初始化状态库

```bash
mlipflow --project "$DEMO_PROJECT" init
```

### 5.4 dry-run

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay --dry-run
```

输出会包含精确：

```text
plan_digest = sha256:...
```

### 5.5 批准同一计划

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay --approve 'sha256:<PLAN_DIGEST>'
```

### 5.6 推进依赖

```bash
mlipflow --project "$DEMO_PROJECT" advance --dry-run
```

复制新的 digest 后：

```bash
mlipflow --project "$DEMO_PROJECT" advance --approve 'sha256:<PLAN_DIGEST>'
```

MLIPFlow 不会因为一个节点 `OK` 就自动执行下一个节点。

---

## 6. CLI 安全语义

严格只读：

```text
list
status
json
inspect
logs
route
doctor
```

会写状态或执行动作：

```text
init
run
advance
retry
stop
```

除 `init` 外，写操作遵循：

```text
dry-run
-> 检查计划
-> 复制 plan_digest
-> approve 精确 digest
```

例如：

```bash
mlipflow --project <PROJECT> run <NODE> --dry-run
mlipflow --project <PROJECT> run <NODE> --approve 'sha256:<PLAN_DIGEST>'
```

如果项目状态、输入、脚本或计划发生变化，旧 digest 不应被继续使用。

---

## 7. `project.yaml`：用户真正需要配置什么

最小结构：

```yaml
schema_version: 1
project:
  id: my-project
  name: My MLIP project
plugin_paths:
  - plugins
locations: {}
backend_profiles: {}
model_registry: model_registry.yaml
state:
  database_path: .mlipflow/state.sqlite3
workflow:
  nodes: []
routing:
  policies: {}
fingerprints:
  full_hash_max_bytes: 67108864
safety:
  auto_submit: false
  auto_advance: false
  require_approval_for_expensive: true
  require_approval_for_destructive: true
```

### 7.1 workflow node

Schema 支持：

```yaml
- id: node-name
  uses: plugin-id@0
  needs: []
  mode: execute
  backend: local
  inputs: {}
  parameters: {}
  resources: {}
```

可用 backend 名称：

```text
local
slurm
ssh-slurm
```

但 **schema 允许某个 backend，不代表科学 Adapter 当前允许它**。

当前内置科学 Adapter 都应使用：

```yaml
backend: local
```

### 7.2 不要依赖隐式 project 搜索

CLI 不会自动向父目录寻找 `project.yaml`。

显式使用：

```bash
mlipflow --project /path/to/project status
```

或：

```bash
mlipflow --project /path/to/project/project.yaml status
```

---

## 8. 科学环境一：SQS / 高熵结构生成

插件：

```text
high-entropy-structure
```

需要：

```text
ASE
icet
可选 pymatgen
```

创建环境：

```bash
conda create -n mlipflow-sqs python=3.11 -y
conda activate mlipflow-sqs
python -m pip install --upgrade pip
python -m pip install ase icet pymatgen
```

检查：

```bash
python -c 'import ase, icet; print(ase.__version__); print(icet.__version__)'
```

仓库已有 bounded smoke 记录：

```text
Python 3.11.14
ASE 3.28.0
icet 3.2
CPU
seed = 23
```

这是已运行过的集成证据，不是版本强制 pin。

### 8.1 SQS execute node 示例

下面字段来自当前 Adapter contract：

```yaml
- id: structure
  uses: high-entropy-structure@0
  mode: execute
  backend: local
  inputs:
    prototype_structure: inputs/prototype.cif
    composition_manifest: inputs/composition.json
  parameters:
    interpreter_argv:
      - /absolute/path/to/mlipflow-sqs/bin/python
    seed: 17
    max_candidates: 2
  resources:
    cpus: 1
```

`composition_manifest` 必须显式描述 sublattice、species/count、cluster cutoff、supercell 等科学设置；不要从 README 猜这些值。

---

## 9. 科学环境二：DIRECT / LASP / SSW

插件：

```text
pes-sampling
```

操作：

```text
direct-select
lasp-ssw-execute
lasp-ssw-normalize-replay
```

### 9.1 DIRECT

按使用的 reviewed DIRECT 源，需要的 Python 包可能包括：

```text
pymatgen
maml
ase
plotly
```

建议单独环境：

```bash
conda create -n mlipflow-direct python=3.11 -y
conda activate mlipflow-direct
python -m pip install pymatgen maml ase plotly
```

DIRECT 历史源没有可消费的 seed 参数；Adapter 会把 seed 限制写进 provenance，而不会伪造历史随机性。

### 9.2 LASP / SSW

MLIPFlow **不分发 LASP，也不实现 SSW 数值内核**。

用户需要准备：

```text
LASP executable
input.arc
lasp.in
必要 auxiliary files
LASP version identity
可选 mpirun/mpiexec
```

当前 LASP execute 是 `local` wrapper contract，所以在集群上应先获得计算节点 allocation，再运行该节点。

示意配置：

```yaml
- id: sampling
  uses: pes-sampling@0
  mode: execute
  backend: local
  inputs:
    lasp_executable: /absolute/site/path/to/lasp
    input_structure: inputs/input.arc
    lasp_input: inputs/lasp.in
  parameters:
    operation: lasp-ssw-execute
    lasp_version: '<ACTUAL_LASP_VERSION>'
    seed_status: HISTORICAL_PARAMETER_UNKNOWN
    acknowledge_uncontrolled_seed: true
    preserve_historical_order: true
    mpi_processes: 8
  resources:
    python_executable: /absolute/path/to/python
    mpi_launcher: /absolute/path/to/mpirun
```

如果不用 MPI，不要填写虚构的 `mpi_launcher`。

---

## 10. 科学环境三：DFT labeling

插件：

```text
dft-labeling
```

当前插件不是一个 VASP driver。它是：

```text
MLIPFlow plan/approval
-> 用户自备 Python labeling wrapper
-> wrapper 生成 DFT 输入并运行/读取 DFT
-> wrapper 写标准 result manifest
-> MLIPFlow check
-> MLIPFlow collect
```

### 10.1 Python 环境

按 wrapper 需要安装：

```text
pymatgen
ASE
dpdata
其他解析依赖
```

示例：

```bash
conda create -n mlipflow-dft python=3.11 -y
conda activate mlipflow-dft
python -m pip install pymatgen ase dpdata
```

VASP 本体不通过这个 Conda 环境安装。

### 10.2 DFT node 示例

```yaml
- id: labeling
  uses: dft-labeling@0
  mode: execute
  backend: local
  inputs:
    structures_manifest: inputs/selected-structures.json
    labeling_config: configs/vasp-labeling.json
    pseudopotential_reference: configs/pseudopotential-metadata.json
  parameters:
    operation: label
    label_script: wrappers/run_vasp_label.py
    interpreter_argv:
      - /absolute/path/to/mlipflow-dft/bin/python
    engine: vasp
    completion_policy:
      require_ionic_convergence: false
    units:
      energy: eV
      length: angstrom
      force: eV/angstrom
      stress: GPa-voigt-xx-yy-zz-yz-xz-xy
```

如果任务是结构优化，把离子收敛要求按照你的 wrapper contract 明确打开。

### 10.3 `label_script` 不能做什么

当前 Adapter 明确要求：

- 不要在 `label_script` 中再次提交 `sbatch`；
- 不要把 scheduler `COMPLETED` 直接视为科学 `OK`；
- 不要只 grep OUTCAR 某一句文字就宣称收敛；
- 不要把 POTCAR 或私钥写入仓库。

在 Slurm 计算节点中，wrapper 可以调用已经分配资源内允许的 MPI/VASP 启动方式，但不能再嵌套提交新的 batch job。

---

## 11. 科学环境四：MLIP training

插件：

```text
mlip-training
```

当前声明框架：

```text
DeepMD
M3GNet
CHGNet
MACE
```

关键事实：**MLIPFlow 不直接 import 或重写这些训练框架。**

它调用用户自备 wrapper：

```text
Python executable
+ wrapper script
+ framework config
+ labeled data
+ requested output
+ seed/device/precision/fingerprints
```

wrapper 最终必须写标准 `training-result.json`。

### 11.1 training node 示例

```yaml
- id: train-mace
  uses: mlip-training@0
  mode: execute
  backend: local
  inputs:
    executable: /absolute/path/to/mlipflow-mace-gpu/bin/python
    script: wrappers/mace_train.py
    config: configs/mace.yaml
    data: data/labeled
    output: model.model
    result_manifest: training-result.json
  parameters:
    framework: mace
    operation: train
    seed: 20260810
    device: cuda
    precision: float64
    dataset_fingerprint: 'sha256:<DATASET_MANIFEST_HASH>'
    config_fingerprint: 'sha256:<CONFIG_HASH>'
  resources:
    cpus: 8
    gpus: 1
```

对于 CPU，把 wrapper 选择的 device 改成 CPU，并使用 CPU framework 环境。

DeepMD 当前只暴露 fresh `train`；当前仓库没有验证完整 DeepMD finetune/freeze/test 生产入口。

M3GNet、CHGNet、MACE 支持 wrapper contract 层面的 `train`/`finetune`，但 framework-specific 参数仍归你的 config/wrapper 管理。

---

## 12. MACE 环境：CPU 和 NVIDIA GPU

MACE 官方推荐先安装适合系统的 PyTorch，再安装 `mace-torch`。

### 12.1 CPU

```bash
conda create -n mlipflow-mace-cpu python=3.11 -y
conda activate mlipflow-mace-cpu
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install mace-torch ase
```

检查：

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

CPU 环境中 `torch.cuda.is_available()` 应为 `False`。

### 12.2 GPU

先在 **真正的 GPU 机器或计算节点**：

```bash
nvidia-smi
```

然后使用 PyTorch 官方安装选择器，根据：

```text
OS
package manager
Python
CUDA compute platform
```

生成适合该站点的安装命令。

不要把 README 中某个 CUDA 小版本当成永久要求。

```bash
conda create -n mlipflow-mace-gpu python=3.11 -y
conda activate mlipflow-mace-gpu
python -m pip install --upgrade pip
python -m pip install <PYTORCH_COMMAND_FROM_OFFICIAL_SELECTOR>
python -m pip install mace-torch ase
```

验证：

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA GPU")'
```

只有以下条件全部成立才算 GPU Python 环境接通：

```text
nvidia-smi 正常
PyTorch import 正常
torch.cuda.is_available() == True
GPU 名称正确
真实模型单点 inference 成功
```

---

## 13. DeepMD 环境：CPU 和 NVIDIA GPU

DeePMD 建议和 MACE 分开环境。

当前 DeePMD stable 文档的 Python interface 要求 Python 3.10+，因此推荐 Python 3.11。

### 13.1 CPU

```bash
conda create -n mlipflow-deepmd-cpu python=3.11 -y
conda activate mlipflow-deepmd-cpu
python -m pip install --upgrade pip
python -m pip install 'deepmd-kit[cpu]'
```

验证：

```bash
dp -h
python -c 'import deepmd; print(deepmd.__version__)'
```

### 13.2 GPU

当前 stable 文档提供 CUDA 12 预编译路线：

```bash
conda create -n mlipflow-deepmd-gpu python=3.11 -y
conda activate mlipflow-deepmd-gpu
python -m pip install --upgrade pip
python -m pip install 'deepmd-kit[gpu,cu12]'
```

这不是说所有集群都必须 CUDA 12。你的集群如果使用其他受支持组合，应按该 DeePMD release 的官方安装矩阵选择。

如果需要 DeePMD 自带/匹配的 LAMMPS 集成，可按照对应版本官方文档选择 `lmp` extra；不要安装一个 DeePMD 后再随意混用另一个来源的 LAMMPS executable。

---

## 14. CHGNet / M3GNet 环境

当前 MLIPFlow 对二者提供的是 **wrapper contract**，不固定它们的完整 Python dependency graph。

建议：

```text
每个框架单独环境
固定 Python/framework/backend 版本
把环境版本写进 result manifest
不要用“latest”作为科学 provenance
```

需要记录：

```text
Python version
framework version
backend version
CPU/GPU device
CUDA runtime（GPU 时）
precision
seed
dataset fingerprint
config fingerprint
model fingerprint
```

如果复现历史 M3GNet，不要在没有验证的情况下静默替换成 MatGL 并仍标成同一模型环境。

---

## 15. benchmark / transport / screening / voltage 分别需要什么

### 15.1 `mlip-benchmark`

当前 normalize/execute 路线可以从显式 reference/prediction pairs 或历史证据重新计算/归一化：

```text
MAE
RMSE
Pearson
```

它不自动加载 MLIP 做 prediction；如果要真实模型预测，使用用户 prediction wrapper。

### 15.2 `ionic-transport`

两类操作：

```text
analyze-existing
md-smoke-and-analyze
```

推荐生产路线是：

```text
外部可靠 MD（如经过验证的 LAMMPS）
-> trajectory/MSD
-> ionic-transport analyze-existing
```

`md-smoke-and-analyze` 有严格时长上限，只用于 integration smoke，不是生产输运计算。

该 smoke 的设备字段接受：

```text
cpu
cuda
mps
```

Linux NVIDIA GPU 使用 `cuda`。

### 15.3 `composition-screening`

对已有候选指标做确定性排序/top-k。它不应自己启动 DFT 或训练。

### 15.4 `electrochemical-voltage`

从显式 total-energy sequence 计算相邻组分电压。输入能量来源必须有明确的 completion status 和 provenance。

---

## 16. 连接集群：先收集站点参数

不要猜 partition、GPU 资源语法、account 或 module 名。

### 16.1 需要收集的字段

| 变量 | 从哪里找 | 写到哪里 |
|---|---|---|
| `<LOGIN_HOST>` | 集群用户文档 | `~/.ssh/config` |
| `<USERNAME>` | 集群账号 | `~/.ssh/config` |
| `<SSH_ALIAS>` | 用户自己定义 | `~/.ssh/config`；未来/实验 `backend_profiles.ssh_profile` |
| `<CPU_PARTITION>` | `sinfo` / 管理员文档 | 用户自己的 `sbatch` |
| `<GPU_PARTITION>` | `sinfo` / 管理员文档 | 用户自己的 `sbatch` |
| `<ACCOUNT>` | 管理员文档 / `sacctmgr`（若开放） | 用户自己的 `sbatch` |
| `<QOS>` | 管理员文档 / `sacctmgr`（若开放） | 用户自己的 `sbatch` |
| `<GPU_RESOURCE_DIRECTIVE>` | Slurm/site 文档 | 用户自己的 `sbatch` |
| `<CUDA_MODULE>` | `module avail` / `module spider` | site shell/wrapper；不是 README |
| `<MPI_MODULE>` | `module avail` / `module spider` | site shell/wrapper |
| `<VASP_MODULE>` | 机构 VASP 文档 | site shell/wrapper |
| `<LAMMPS_MODULE>` | 集群软件目录 | site shell/wrapper |
| `<PROJECT_ROOT>` | 集群 storage 文档 | 用户 project |
| `<SCRATCH_ROOT>` | 集群 storage 文档 | 用户运行目录/外部程序 |

### 16.2 登录节点查询

```bash
hostname
uname -a
which sbatch
which srun
which squeue
which sacct
sinfo
sinfo -o '%P %a %l %D %G'
```

可用时：

```bash
scontrol show partition <PARTITION>
module avail
module spider cuda
module spider mpi
module spider vasp
module spider lammps
```

如果 `sacctmgr` 对普通用户开放，可查询用户关联；否则直接看站点文档：

```bash
sacctmgr show assoc where user="$USER" format=User,Account,Partition,QOS
```

任何命令不可用时，以集群管理员文档为准。

---

## 17. SSH 配置

MLIPFlow 不保存密码、私钥文本或 token。

### 17.1 生成 HPC 登录 key

在你的本地电脑：

```bash
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519_hpc -C 'mlipflow-hpc'
chmod 600 ~/.ssh/id_ed25519_hpc
```

把 **公钥** 按集群机构规定安装到远端账号。

不要把私钥放进项目目录或 GitHub。

### 17.2 `~/.ssh/config`

```sshconfig
Host <SSH_ALIAS>
    HostName <LOGIN_HOST>
    User <USERNAME>
    IdentityFile ~/.ssh/id_ed25519_hpc
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

需要跳板机时：

```sshconfig
Host <SSH_ALIAS>
    HostName <INTERNAL_LOGIN_HOST>
    User <USERNAME>
    IdentityFile ~/.ssh/id_ed25519_hpc
    ProxyJump <BASTION_ALIAS>
```

第一次连接时应按机构文档核对 host key fingerprint，不要盲目关闭 host key verification。

测试：

```bash
ssh <SSH_ALIAS> 'hostname'
```

然后：

```bash
ssh <SSH_ALIAS> 'which sbatch && which srun && which squeue'
```

### 17.3 MLIPFlow 中只引用 alias

未来/实验性的 remote profile 形式：

```yaml
backend_profiles:
  cluster-a:
    ssh_profile: <SSH_ALIAS>
```

不要写：

```text
password
private key content
token
IdentityFile
```

当前科学 Adapter 不应直接切换到 `ssh-slurm`；这个 profile 主要用于核心 backend 能力和后续 HPC 集成。

---

## 18. 把仓库和环境放到集群哪里

代码和环境必须放在 **登录节点和计算节点都能访问** 的文件系统。

常见布局示意：

```text
$HOME/src/mlipflow                  # 代码
$PROJECT/mlipflow/envs/             # 大 Python 环境（如果站点推荐）
$PROJECT/mlipflow/models/           # 模型
$PROJECT/mlipflow/projects/         # project.yaml / configs
$SCRATCH/mlipflow-runs/             # 大型临时科学输出
```

实际 `$PROJECT/$SCRATCH` 规则必须以站点文档为准。

私有仓库如果允许集群直接访问 GitHub，按机构安全政策配置 GitHub 认证；不要复用或复制 HPC 登录私钥作为 GitHub 私钥。

如果集群不允许外网访问，用站点批准的 `rsync/scp` 方法同步代码和小配置。

---

## 19. 集群控制环境

在集群共享文件系统中建立控制环境：

```bash
conda create -n mlipflow-control python=3.11 -y
conda activate mlipflow-control
cd <MLIPFLOW_REPOSITORY>
python -m pip install --upgrade pip
python -m pip install -e .
```

检查：

```bash
mlipflow --version
mlipflow --project examples/high_entropy_sulfide doctor
```

登录节点只做：

```text
编辑配置
查看状态
生成 dry-run plan
提交 batch
读取小日志
```

不要在登录节点运行长训练、MD、VASP 或大 SQS。

---

## 20. 集群 CPU：先用交互 allocation 验证

先查询真实 CPU partition。

典型请求形式：

```bash
salloc --partition=<CPU_PARTITION> --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:30:00
```

如果站点要求 account/QoS，加入对应字段。

进入计算节点：

```bash
srun --pty bash
```

验证：

```bash
hostname
lscpu | head
python --version
mlipflow --version
```

激活对应科学环境做 tiny smoke，例如 SQS：

```bash
conda activate mlipflow-sqs
python -c 'import ase, icet; print(ase.__version__, icet.__version__)'
```

成功后再运行一个有界 scientific node。

---

## 21. 集群 GPU：先识别 Slurm 的 GPU 请求方式

Slurm 可以使用多种 GPU 请求方式，例如：

```text
--gres=gpu:1
--gres=gpu:<TYPE>:1
--gpus=1
--gpus-per-node=1
```

并不是每个站点都支持所有写法。特别是 `--gpus*` 依赖站点的 Slurm `select/cons_tres` 配置。

先看：

```bash
sinfo -o '%P %G'
```

再看集群文档确认正式 GPU directive。

### 21.1 申请 GPU 交互节点

常见示意：

```bash
salloc --partition=<GPU_PARTITION> --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=00:30:00
```

如果你的站点使用其他 GPU directive，替换 `--gres=gpu:1`。

然后：

```bash
srun --pty bash
```

验证：

```bash
hostname
nvidia-smi
printf '%s\n' "$CUDA_VISIBLE_DEVICES"
```

Slurm GPU GRES/TRES 配置通常会为 job step 设置 `CUDA_VISIBLE_DEVICES`；不要在自己的程序里无条件硬编码物理 GPU `0` 并绕开 scheduler 分配。

### 21.2 Python GPU 验证

激活实际 GPU 环境：

```bash
conda activate mlipflow-mace-gpu
```

检查：

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA GPU")'
```

通过后顺序应是：

```text
tensor/device test
-> 单结构 model inference
-> CPU/GPU energy-force 对照
-> tiny MD
-> 再做训练或长 MD
```

---

## 22. 一个容易忽略的限制：MLIPFlow 会清洗执行环境

`LocalBackend` 不会把父 shell 的所有环境变量原样传给 scientific wrapper。

当前默认允许的核心变量包括：

```text
HOME
LANG
PATH
PYTHONHOME
PYTHONPATH
TMP/TEMP/TMPDIR
VIRTUAL_ENV
CONDA_PREFIX
CONDA_DEFAULT_ENV
```

以及前缀：

```text
LC_
SLURM_
CUDA_
ROCR_
OMP_
MKL_
```

因此不要假设：

```bash
module load <SOME_SOFTWARE>
```

之后 module 设置的任意 `LD_LIBRARY_PATH`、自定义变量等一定会进入 MLIPFlow 启动的 wrapper。

更稳妥的做法：

1. 在 `project.yaml` 中为科学环境使用 **明确的 Python executable**；
2. 对 VASP/LAMMPS/LASP 使用明确的 executable path；
3. 尽量使用站点提供、RPATH 已正确设置的 build；
4. 如果软件必须依赖额外动态库/环境，由用户自备的受审 Python wrapper 显式建立该软件的运行环境；
5. 不要通过把密码/token 塞入环境变量来绕过安全边界。

这也是为什么“在 shell 中 `module load` 成功”不能单独证明 MLIPFlow scientific node 一定成功。

---

## 23. 当前版本怎样在 Slurm 上真正跑科学节点

当前最通用的方式是：

```text
SLURM 负责分配 CPU/GPU
-> batch script 启动一次 MLIPFlow CLI
-> workflow node 仍写 backend: local
-> MLIPFlow 在已经分配的计算节点里运行 Adapter argv
-> Adapter check/collect
-> MLIPFlow 写状态和 provenance
```

这不是绕过 MLIPFlow，因为科学命令仍然由 MLIPFlow plan/approval/state/check/collect 管理；`sbatch` 只是外层资源分配器。

### 23.1 先在登录节点生成计划

```bash
mlipflow --project <PROJECT_ROOT> run <NODE_ID> --dry-run
```

检查：

```text
输入
脚本
模型
backend=local
resources
输出目录
plan_digest
```

复制准确 digest。

### 23.2 CPU batch wrapper

保存为用户自己的 `run_mlipflow_cpu.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=mlipflow-cpu
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate mlipflow-control
cd <PROJECT_ROOT>
mlipflow --project <PROJECT_ROOT> run <NODE_ID> --approve 'sha256:<PLAN_DIGEST>'
```

如果集群要求 account/QoS，在用户自己的脚本中添加：

```text
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>
```

### 23.3 GPU batch wrapper

```bash
#!/bin/bash
#SBATCH --job-name=mlipflow-gpu
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate mlipflow-control
nvidia-smi
printf '%s\n' "$CUDA_VISIBLE_DEVICES"
cd <PROJECT_ROOT>
mlipflow --project <PROJECT_ROOT> run <NODE_ID> --approve 'sha256:<PLAN_DIGEST>'
```

把 `#SBATCH --gres=gpu:1` 替换成你的站点正式 GPU 请求方式。

### 23.4 提交与检查

```bash
sbatch run_mlipflow_cpu.sbatch
```

或：

```bash
sbatch run_mlipflow_gpu.sbatch
```

查看 scheduler：

```bash
squeue -u "$USER"
```

结束后：

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

然后读取 **MLIPFlow 自己的状态**：

```bash
mlipflow --project <PROJECT_ROOT> status <NODE_ID>
mlipflow --project <PROJECT_ROOT> logs <NODE_ID>
```

只有 MLIPFlow plugin check/collect 返回 `OK` 才算科学节点成功。`sacct` 的 `COMPLETED` 不是科学完成判据。

---

## 24. VASP 配置

MLIPFlow 不分发 VASP。

用户需要合法的：

```text
VASP license
VASP executable
MPI runtime
POTCAR library
```

### 24.1 找到 VASP

站点可能使用 module：

```bash
module spider vasp
module load <VASP_MODULE>
which vasp_std
```

也可能提供绝对路径。

记录真实：

```text
VASP version
executable path/module identity
MPI implementation
compiler/toolchain（能获取时）
```

### 24.2 VASP 资源不要从 README 抄生产值

CPU/MPI batch allocation 通常需要：

```text
nodes
ntasks
cpus-per-task
memory
walltime
partition
account/qos
```

`NCORE/KPAR` 等 VASP 参数必须做站点和体系 benchmark。

---

## 25. POTCAR / 赝势：怎样配置才可复现

MLIPFlow **不会下载、生成或提交 POTCAR**。

### 25.1 pymatgen 目录

如果 wrapper 使用 pymatgen，可把机构合法获得的 VASP pseudopotential 整理成 pymatgen 能读取的目录。

当前 pymatgen 官方方式：

```bash
pmg config -p /path/to/original/potcar_PBE /path/to/pmg_potcars
pmg config --add PMG_VASP_PSP_DIR /path/to/pmg_potcars
```

不要把 POTCAR 内容提交仓库。

### 25.2 元素映射必须显式

例如 Li-M-P-S 项目应维护类似元数据：

```yaml
functional: PBE
species:
  Li: <ACTUAL_LI_POTCAR_LABEL>
  Mn: <ACTUAL_MN_POTCAR_LABEL>
  Fe: <ACTUAL_FE_POTCAR_LABEL>
  Ni: <ACTUAL_NI_POTCAR_LABEL>
  Cu: <ACTUAL_CU_POTCAR_LABEL>
  Zn: <ACTUAL_ZN_POTCAR_LABEL>
  P: <ACTUAL_P_POTCAR_LABEL>
  S: <ACTUAL_S_POTCAR_LABEL>
```

不要因为某个例子使用 `Fe_pv` 就假定所有项目都应该用 `Fe_pv`。

### 25.3 记录 metadata，不提交势文件

至少保存：

```text
element
POTCAR label
functional
release/header identity
SHA-256
```

本地计算 hash：

```bash
sha256sum /licensed/path/to/POTCAR
```

如果是多个元素的独立源文件，分别记录每个源文件 identity/hash，再记录最终拼接 POTCAR 的 hash。

### 25.4 `pseudopotential_reference`

`dft-labeling` 可以接受一个 **用户管理的 reference**；这个 reference 应描述势函数身份，而不是把 POTCAR binary 放进 MLIPFlow 仓库。

---

## 26. DFT wrapper 必须输出什么

wrapper 至少应该完成：

```text
读取 structures manifest
生成 POSCAR/INCAR/KPOINTS
按显式映射组装 POTCAR
运行 VASP
检查电子收敛
需要时检查离子收敛
检查输出是否截断
解析 energy/force/stress
统一单位和 stress convention
输出 dataset
输出 dft-labeling-result.json
计算 artifact SHA-256
```

当前 Adapter 要求 result 中至少能表达：

```json
{
  "schema_version": 1,
  "plugin_id": "dft-labeling",
  "status": "OK",
  "engine": "vasp",
  "completion": {
    "scheduler_success": true,
    "electronic_converged": true,
    "ionic_convergence_required": false,
    "ionic_converged": null,
    "truncated": false
  },
  "units": {
    "energy": "eV",
    "length": "angstrom",
    "force": "eV/angstrom",
    "stress": "<EXPLICIT_CONVENTION>"
  },
  "source_structure_count": 1,
  "label_count": 1,
  "artifacts": []
}
```

真正结果还应包含批准输入对应的 fingerprints 和实际 dataset artifact。

---

## 27. LAMMPS：为什么 README 不固定一个通用版本

**MLIPFlow 本身不依赖某个固定 LAMMPS 版本。**

LAMMPS 版本由你使用的 MLIP interface 决定。

错误做法：

```text
“README 写 LAMMPS 202X，所以 DeepMD/MACE 都统一装这个版本”
```

正确做法：

```text
先决定 MLIP interface
-> 查该 framework release 的官方 LAMMPS 兼容方式
-> 建一套固定 build
-> 在目标 CPU/GPU 节点 smoke
-> 记录 exact build/version
```

### 27.1 任何 LAMMPS build 先做这些检查

```bash
which lmp
lmp -h | head -n 40
lmp -h | grep -Ei 'deepmd|mace|mliap|kokkos'
```

`lmp -h` 可以确认当前 executable 编译进了哪些 style/package。

生产 provenance 至少记录：

```text
LAMMPS version/tag/commit
executable path
compiler
MPI
CPU/GPU build
Kokkos/GPU package 状态
MLIP interface
framework version
model SHA-256
```

---

## 28. DeepMD + LAMMPS

DeePMD 官方目前支持两种主要集成：

```text
built-in mode
plugin mode
```

### 28.1 最简单的原则

如果安装 DeePMD 时同时安装了它匹配的 LAMMPS integration：

```text
尽量使用同一个 DeePMD 环境提供的 lmp
```

不要同时 `module load` 一个不相关的 LAMMPS 后仍假设 `pair_style deepmd` ABI 一定匹配。

### 28.2 plugin mode

官方文档支持加载：

```text
libdeepmd_lmp.so
```

较新的 LAMMPS plugin 系统也可以通过 `LAMMPS_PLUGIN_PATH` 配置。

先验证：

```bash
lmp -h | grep -i deepmd
```

然后做一个 0/1/2-step 极小模型 smoke，再做正式 MD。

### 28.3 units

DeePMD 文档说明常见 `metal` units 与其内部 Å/eV/eV/Å 自然一致；其他受支持 units 可由 LAMMPS 转换，但必须记录实际 unit style。

---

## 29. MACE + LAMMPS

MACE 当前有不止一种 LAMMPS interface，不能混成一套说明。

### 29.1 原始 MACE LAMMPS interface

官方文档提供专用 MACE/LAMMPS build，支持 CPU/GPU；GPU build 通常使用 Kokkos/CUDA。

模型需要先导出为对应 LAMMPS model。

在真正 MD 前必须对照：

```text
同一结构
MACE ASE calculator
vs
MACE LAMMPS
```

比较 energy/forces。

### 29.2 ML-IAP interface

MACE 的 ML-IAP 路线是另一套接口。当前官方文档要求特定 LAMMPS build 选项，例如：

```text
PKG_ML-IAP
MLIAP_ENABLE_PYTHON
PKG_PYTHON
Kokkos GPU 配置
```

当前文档还把该接口标为需要谨慎验证的较新路线。

所以 README 不会把它和原始 `pair_style mace` 视为同一个 build。

### 29.3 GPU 架构

MACE-LAMMPS GPU build 的 Kokkos architecture 必须匹配目标 GPU 架构。不要在未知 GPU 架构上复制另一台机器的 CMake flags。

---

## 30. LAMMPS scientific smoke 顺序

无论 DeepMD 还是 MACE，都按：

```text
1. lmp -h 正常
2. 目标 pair style/interface 可见
3. 模型文件可加载
4. 元素 type map 明确
5. unit style 明确
6. 单结构 0/1/2 step 成功
7. 原生 framework 与 LAMMPS energy/force 对照
8. CPU/GPU 对照（需要 GPU 时）
9. tiny NVE/NVT
10. 再做长 MD
```

不要从“LAMMPS 能启动”直接跳到生产长 MD。

---

## 31. LASP 集群运行

LASP 由用户所在机构提供。

集群运行前确认：

```bash
which <LASP_EXECUTABLE_NAME>
```

记录：

```text
LASP version
executable SHA/path identity
MPI implementation
input.arc hash
lasp.in hash
auxiliary file hashes
```

`pes-sampling` 的 LASP wrapper 可以在已经分配的 Slurm 计算节点中运行，并使用显式 `mpirun/mpiexec`；它自己不提交 scheduler job。

---

## 32. `doctor` 能检查什么，不能检查什么

```bash
mlipflow --project <PROJECT> doctor
```

它适合检查：

```text
project 是否能加载
plugin 是否可发现
声明的必需 Python 包
部分外部 executable
state database
选择 slurm/ssh-slurm 时基础 scheduler/ssh executable
```

但 `doctor` 不能替代：

```text
GPU driver/runtime 验证
VASP license/module 验证
POTCAR identity 验证
LAMMPS pair style ABI 验证
模型 inference 验证
MPI 并行 smoke
科学参数收敛测试
```

站点软件名称常常是逻辑名或 module 提供的命令，因此必须继续执行本文每个程序的显式 preflight。

---

## 33. 当前 first-class `slurm` / `ssh-slurm` 的边界

Schema 已允许：

```yaml
backend: slurm
```

和：

```yaml
backend: ssh-slurm
```

核心也已经有：

```text
sbatch
squeue
sacct
scancel
SSH alias
remote scheduler query
scp fetch primitive
job-id parsing
identity-bound completion context
```

但是当前 `run_node` 会对 Adapter-backed scientific node 的 scheduler execution 显式报错，原因是：

```text
scheduler completion path 尚未与 pinned scientific checker 完整结合
```

因此不要通过手工修改 plugin manifest 来绕过限制。

真正 first-class HPC 闭环需要：

```text
local project
-> controlled remote stage
-> exact approved remote work dir
-> sbatch
-> persisted job ID
-> squeue/sacct reconciliation
-> terminal state
-> allowlisted fetch
-> hash verification
-> pinned plugin scientific check
-> collect
-> final run manifest
```

这部分完成真实集群验收后，才应该把 README 的“当前 Slurm 外层包裹 local Adapter”升级成 MLIPFlow 原生 scheduler 使用方式。

---

## 34. CPU/GPU 科学一致性检查

GPU 跑得起来不代表科学结果正确。

至少选择小样本比较：

```text
same structure
same model
same units
same element map
same precision（尽可能）
CPU energy/forces
GPU energy/forces
predefined tolerance
```

对于 LAMMPS 还要比较：

```text
framework native inference
vs
LAMMPS inference
```

只有通过后再扩大到长 MD/大训练。

---

## 35. 生产运行必须记录的 provenance

至少保存：

```text
MLIPFlow git commit
project.yaml SHA-256
plugin id/version
adapter/script SHA-256
Python version
framework name/version
backend library version
CPU model 或 GPU model
NVIDIA driver（GPU）
CUDA runtime（GPU）
precision
seed
MPI implementation/version
SLURM job ID
partition/account/qos/resources
VASP version（DFT）
POTCAR identity + SHA-256（DFT）
LAMMPS version/build/interface（MD）
LASP version（LASP）
model SHA-256
dataset/input/config SHA-256
```

这样本地、集群 CPU 和集群 GPU 的结果才可以真正比较。

---

## 36. 第一次真实集群验证顺序

不要直接提交生产 DFT/训练/长 MD。

推荐：

```text
A. ssh <SSH_ALIAS> 成功
B. 登录节点能看到 sbatch/srun/squeue
C. 找到 CPU/GPU partition/account/qos
D. CPU salloc 成功
E. CPU 节点 MLIPFlow/control env 成功
F. CPU tiny science smoke 成功
G. GPU salloc 成功
H. GPU 节点 nvidia-smi 成功
I. torch.cuda.is_available() == True
J. GPU 单模型 inference 成功
K. CPU/GPU energy-force 对照
L. CPU batch wrapper 跑 MLIPFlow node 成功
M. GPU batch wrapper 跑 MLIPFlow node 成功
N. VASP tiny MPI job + convergence parser 成功
O. POTCAR identity/hash 固定
P. DeepMD 或 MACE 对应 LAMMPS build 成功
Q. LAMMPS 1-2 step parity 成功
R. LASP tiny execute（需要时）
S. 小规模真实工作负载
T. 生产工作负载
```

任何一步失败都先修这一层。

---

## 37. 常见问题

### `ModuleNotFoundError`

```bash
which python
python --version
python -m pip list
```

检查 `project.yaml` 中是否真的指向正确 scientific environment 的 Python executable。

### 登录节点没有 `nvidia-smi`

很多集群登录节点没有 GPU。申请 GPU allocation 后在计算节点检查。

### `torch.cuda.is_available()` 是 False

检查：

```text
是否真的在 GPU 计算节点
Slurm 是否分配 GPU
nvidia-smi 是否成功
CUDA_VISIBLE_DEVICES
是否安装 GPU 版 PyTorch
PyTorch/CUDA 与 NVIDIA driver 是否兼容
是否激活了错误环境
```

### `sbatch: command not found`

说明当前环境不是该 Slurm 集群的正常登录环境，或 scheduler module/PATH 未初始化。

### `sacct` 查不到作业

可能站点没有开放 accounting、存在延迟或使用不同 accounting 方法。按站点文档处理。

### `lmp` 能启动但没有 `deepmd/mace/mliap`

当前 LAMMPS build 没有目标 MLIP interface。不要继续跑 MD，先换/重编译正确 build。

### `lmp` 找到 pair style 但模型加载失败

检查：

```text
framework/interface 版本
动态库
模型导出格式
模型元素集合
pair_coeff/type map
CPU/GPU build
```

### VASP 找不到 POTCAR

检查机构合法 POTCAR 根目录、pymatgen `PMG_VASP_PSP_DIR`、元素 label 和 wrapper 的实际组装路径。

### shell 里 `module load` 后外部程序仍在 MLIPFlow 中失败

检查第 22 节的环境清洗规则。尤其不要假设 `LD_LIBRARY_PATH` 一定会继承到 wrapper。

### `backend: slurm` scientific node 被拒绝

这是当前设计行为，不是 YAML 拼写错误。内置 Adapter scheduler reconciliation 还没有开放；当前集群执行请使用 Slurm 外层分配 + node `backend: local`。

### scheduler `COMPLETED`，但 MLIPFlow 是 FAIL

这是正确行为。查看：

```bash
mlipflow --project <PROJECT> logs <NODE>
mlipflow --project <PROJECT> inspect <NODE>
```

检查 plugin completion manifest、artifact hash、科学收敛或单位条件。

---

## 38. 当前已有的真实本地 smoke 证据

### SQS

```text
Python 3.11.14
ASE 3.28.0
icet 3.2
CPU/local
seed 23
```

### MACE MD -> transport integration smoke

```text
Python 3.12.12
ASE 3.28.0
MACE 0.3.15
PyTorch 2.10.0
NumPy 2.4.3
Pandas 2.3.3
Plotly 6.6.0
CPU float64
```

后者每个温度只有极少 production steps，只是软件链路 smoke，不是输运收敛证据。

这些版本是 **已知曾跑通的环境证据**，不是要求用户统一安装这些精确版本。

---

## 39. 外部官方文档

依赖和 HPC 软件更新很快。安装 GPU/framework/LAMMPS interface 时应同时检查对应 release 的官方文档。

- PyTorch installation: <https://docs.pytorch.org/get-started/locally/>
- DeePMD stable installation: <https://docs.deepmodeling.com/projects/deepmd/en/stable/install/easy-install.html>
- DeePMD + LAMMPS: <https://docs.deepmodeling.com/projects/deepmd/en/stable/install/install-lammps.html>
- DeePMD LAMMPS commands: <https://docs.deepmodeling.com/projects/deepmd/en/stable/third-party/lammps-command.html>
- MACE installation: <https://mace-docs.readthedocs.io/en/latest/guide/installation.html>
- MACE LAMMPS: <https://mace-docs.readthedocs.io/en/latest/guide/lammps.html>
- MACE ML-IAP LAMMPS: <https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html>
- Slurm GPU/GRES: <https://slurm.schedmd.com/gres.html>
- LAMMPS command-line options: <https://docs.lammps.org/Run_options.html>
- pymatgen POTCAR setup: <https://pymatgen.org/installation.html>

README 给出的是 MLIPFlow 的稳定配置逻辑；framework/driver/build 的具体版本仍应以你实际安装版本的官方文档为准。

---

## 40. 开发与测试

```bash
python -m pytest
```

静态检查：

```bash
ruff check .
```

贡献前阅读：

- `CONTRIBUTING.md`
- `SECURITY.md`
- `AGENTS.md`
- `docs/PLUGIN_DEVELOPMENT.md`

---

## 41. 一句话使用路径

如果你是第一次使用：

```text
clone/install
-> offline replay
-> 为所需插件建立独立科学环境
-> 本地/交互计算节点 tiny smoke
-> 配 SSH + 查清 partition/account/GPU directive
-> 用 sbatch 只负责资源分配，在计算节点运行 backend: local 的 MLIPFlow node
-> 配 VASP/POTCAR/LAMMPS/LASP
-> 做 CPU/GPU/原生框架/LAMMPS 科学对照
-> 通过 check/collect
-> 再扩大到生产规模
```

当前版本不要把科学 Adapter 强行改成 `slurm`/`ssh-slurm`。等仓库完成真实 stage -> submit -> monitor -> fetch -> pinned scientific check 闭环后，再启用 MLIPFlow 原生远程调度路径。

---

## License

MLIPFlow 使用 Apache License 2.0。

VASP、POTCAR、LASP、LAMMPS、MLIP frameworks、模型和数据仍受各自许可证与机构政策约束；安装 MLIPFlow 不会自动获得这些外部资产的使用权。
