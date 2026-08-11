# MLIPFlow

MLIPFlow 是一个面向机器学习原子势（MLIP）研究的确定性工作流层。它负责把结构生成、PES 采样、DFT 标注、模型训练、基准评估、离子输运、成分筛选和电化学电压组织成显式 DAG，并把状态、审批、产物、版本和来源写入可审计记录。

当前版本为 **0.1.0 alpha**。

> **重要边界**
>
> - MLIPFlow 不是 VASP、LAMMPS、LASP、DeepMD、MACE、CHGNet、M3GNet 或其他数值软件的替代品。
> - 仓库不会分发 VASP、POTCAR、LASP、模型权重、私钥、token 或集群账号信息。
> - 核心代码已经实现 `local`、`slurm`、`ssh-slurm` 后端边界，但当前内置科学 Adapter 仍只声明 `local`；真实远程 stage → `sbatch` → monitor → fetch → scientific check 闭环尚未完成生产验证。
> - 当前真实设备验证以 **CPU** 为主。`ionic-transport` 的现有 MD 接口只接受 `cpu`、`cuda`、`mps`；仓库没有 Ascend/CANN/`torch_npu` 后端，因此 **NPU 当前不是受支持设备**。
> - 任何 DFT、长 MD、训练、大规模筛选或真实集群提交都应先 `--dry-run`，确认计划摘要后再显式批准。

---

## 1. MLIPFlow 能做什么

核心工作流：

```text
structure
  -> PES sampling
  -> DFT labeling
  -> MLIP training
  -> benchmark
       |-> ionic transport
       |-> composition screening -> high-fidelity validation
       `-> electrochemical voltage
```

当前八个科学插件：

| 插件 | 主要用途 | 当前执行边界 |
|---|---|---|
| `high-entropy-structure` | 高熵/SQS 结构生成 | 本地 seeded icet wrapper 已实现 |
| `pes-sampling` | DIRECT 代表结构选择、LASP/SSW execute/replay | 仅 local；LASP 由用户自行提供 |
| `dft-labeling` | DFT 标注与数据集装配 | 用户自备 VASP/脚本；仅 local Adapter |
| `mlip-training` | DeepMD/M3GNet/CHGNet/MACE 训练 wrapper contract | 用户自备训练环境；仅 local Adapter |
| `mlip-benchmark` | MAE/RMSE/Pearson、历史 benchmark 归一化 | 本地可执行 |
| `ionic-transport` | 轨迹/MSD 后处理、极小 ASE-MD smoke | 本地可执行；现有设备为 CPU/CUDA/MPS |
| `composition-screening` | 已有候选指标排序与 top-k | 本地可执行 |
| `electrochemical-voltage` | 总能序列电压后处理、S11 replay | 本地可执行 |

更严格的完成度见：

- `docs/IMPLEMENTATION_STATUS.md`
- `docs/SCIENTIFIC_VALIDATION.md`
- `docs/HPC_VALIDATION.md`
- `docs/CODEBASE_INVENTORY.md`

---

## 2. 推荐部署模型

不要试图把所有软件塞进一个 Python 环境。建议至少分成以下三层：

```text
control environment
  MLIPFlow / PyYAML / SQLite / lightweight science dependencies

scientific environments
  SQS: ASE + icet
  MACE: PyTorch + mace-torch
  DeepMD: deepmd-kit
  CHGNet: chgnet
  M3GNet/MatGL: framework-specific environment
  analysis: numpy/pandas/plotly/pymatgen as needed

site programs
  VASP
  LASP
  LAMMPS
  MPI
  SLURM
  SSH
```

MLIPFlow 的职责是记录“使用了哪个环境、哪个脚本、哪个模型、哪个版本、哪个输入、哪个资源配置”，而不是把所有外部程序安装到同一个环境中。

---

## 3. Python 环境

### 3.1 最低要求

`pyproject.toml` 当前要求：

```text
Python >= 3.9
```

核心运行时只有：

```text
PyYAML >= 6.0
```

推荐实际部署使用 Python 3.11 或 3.12，并为 MLIPFlow 单独建立环境。

### 3.2 推荐安装

使用 Conda/Mamba：

```bash
conda create -n mlipflow python=3.11 -y
conda activate mlipflow
python -m pip install --upgrade pip
python -m pip install -e .
```

开发环境：

```bash
python -m pip install -e '.[dev]'
```

包含仓库直接使用的通用科学依赖：

```bash
python -m pip install -e '.[science]'
```

当前 `science` extra 包含：

```text
numpy >= 1.23
ase >= 3.22
icet >= 2.0
pandas >= 1.5
plotly >= 5
```

### 3.3 安装后检查

```bash
python --version
python -c 'import mlipflow; print(mlipflow.__version__)'
mlipflow --version
mlipflow --project examples/high_entropy_sulfide doctor
```

如果只是运行离线 replay，不需要 VASP、LAMMPS、LASP 或模型框架。

---

## 4. 按功能安装科学环境

### 4.1 SQS / 高熵结构生成

推荐独立环境：

```bash
conda create -n mlipflow-sqs python=3.11 -y
conda activate mlipflow-sqs
python -m pip install ase icet pymatgen
```

仓库当前真实 smoke 使用过：

```text
ASE 3.28.0
icet 3.2
```

这两个版本是已执行过的本地集成记录，不表示它们是唯一允许版本。

### 4.2 MACE

建议单独环境：

```bash
conda create -n mlipflow-mace python=3.11 -y
conda activate mlipflow-mace
python -m pip install mace-torch
```

仓库当前真实 CPU smoke 记录过：

```text
MACE 0.3.15
ASE 3.28.0
```

生产训练应把 `mace-torch`、PyTorch、CUDA/ROCm/NPU runtime 的精确版本写入结果 manifest，不要只写“latest”。

### 4.3 DeepMD

建议独立环境：

```bash
conda create -n mlipflow-deepmd python=3.11 -y
conda activate mlipflow-deepmd
python -m pip install deepmd-kit
```

如果要在 LAMMPS 中运行 DeepMD，必须确保实际使用的 LAMMPS build 包含与当前 DeepMD 安装匹配的接口。不要假定系统里的任意 `lmp` 都支持 `pair_style deepmd`。

### 4.4 CHGNet / M3GNet / MatGL

这些框架的依赖变化较快，建议分别建环境，不要与 MACE/DeepMD 强行混装：

```text
mlipflow-chgnet
mlipflow-m3gnet
mlipflow-matgl
```

每个训练 wrapper 都应输出：

```text
framework name
framework version
python version
device
precision
seed
dataset SHA-256
config SHA-256
model SHA-256
```

---

## 5. CPU、GPU、NPU 支持状态

### CPU

CPU 是当前最稳妥的执行路径，也是仓库已有真实 integration smoke 的设备。

训练或 MD 的项目参数示例：

```yaml
parameters:
  device: cpu
  precision: float64
```

### CUDA GPU

核心环境变量白名单允许 `CUDA_*`，`ionic-transport` 当前接口接受 `device=cuda`，但具体是否可运行取决于你安装的 PyTorch/框架和驱动。

GPU 不应只记录“GPU”；至少记录：

```text
GPU model
CUDA driver
CUDA runtime
PyTorch/framework version
device index
precision
```

### NPU / Ascend

**当前仓库没有 NPU Adapter。**

以下内容目前都未在 MLIPFlow 中实现或验证：

```text
device=npu
device=ascend
torch_npu
CANN toolkit
Ascend runtime
NPU-aware DeepMD/MACE/CHGNet/M3GNet wrapper
NPU SLURM resource mapping
```

因此不要在 `project.yaml` 中把 NPU 当成已经支持的设备。

如果要扩展 NPU，建议新增独立 wrapper，并至少验证：

1. 框架是否原生支持 Ascend/NPU；
2. `torch_npu`/CANN 与 Python/PyTorch 的版本矩阵；
3. 单卡推理；
4. 单卡训练；
5. 多卡通信；
6. seed 与数值重复性；
7. result manifest 中的设备/驱动/runtime 版本；
8. SLURM 中 NPU 资源请求字段；
9. 与 CPU 或受信 GPU 结果进行科学对照。

完成这些验证前，NPU 应标记为 `EXTERNAL_VALIDATION_PENDING`。

---

## 6. VASP 与赝势配置

MLIPFlow **不会保存或生成 POTCAR**。VASP 许可证、可执行文件和赝势由用户所在机构管理。

### 6.1 推荐目录布局

站点层目录示例：

```text
/apps/vasp/
  6.x/
    vasp_std
    vasp_gam
    vasp_ncl

/data/pseudopotentials/
  potpaw_PBE/
    Li/
    Mn_pv/
    Fe_pv/
    Ni/
    Cu/
    Zn/
    P/
    S/
```

真实目录可以不同。不要把绝对站点路径提交到公开仓库。

### 6.2 pymatgen 赝势目录

如果你的输入生成脚本使用 pymatgen，可在用户环境中设置赝势根目录，例如：

```bash
export PMG_VASP_PSP_DIR=/data/pseudopotentials
```

然后用你自己的站点配置确认 pymatgen 能解析所需 POTCAR family。

不要把 `PMG_VASP_PSP_DIR`、POTCAR 内容或绝对私人路径硬编码进插件源码。

### 6.3 势函数选择必须显式记录

对于 Li-M-P-S 体系，至少记录每个元素实际使用的 POTCAR label，例如：

```yaml
pseudopotentials:
  functional: PBE
  species:
    Li: Li
    Mn: Mn_pv
    Fe: Fe_pv
    Ni: Ni
    Cu: Cu
    Zn: Zn
    P: P
    S: S
```

上面的 label 只是配置格式示例，不是强制科学选择。实际 POTCAR 必须与论文、历史计算或你的重新验证方案一致。

### 6.4 不要只记录文件名

建议把每个 POTCAR 的可公开元数据写入你的私有 site manifest：

```text
species
POTCAR label
functional
release/header identity
file SHA-256
```

POTCAR 本体仍留在受许可证保护的站点目录。

### 6.5 DFT wrapper 应负责的内容

当前 `dft-labeling` 插件要求用户自备 wrapper。wrapper 至少应：

1. 根据结构生成 INCAR/KPOINTS/POSCAR；
2. 从站点赝势目录组装 POTCAR；
3. 启动 VASP 或准备给调度器执行的命令；
4. 检查电子收敛；
5. 若为结构优化，检查离子收敛；
6. 检查输出是否截断；
7. 统一能量/力/应力单位与应力约定；
8. 输出标准 `dft-labeling-result.json`；
9. 对数据集产物计算 SHA-256。

不能只通过搜索 OUTCAR 中某一句文本就把任务判为成功。

---

## 7. LAMMPS 配置

### 7.1 MLIPFlow 当前没有固定 LAMMPS 版本

仓库没有声明一个通用的 `LAMMPS == x.y`。这是有意的，因为不同 MLIP 后端通常依赖不同的 LAMMPS build 或插件。

因此生产配置应记录“实际 build”，而不是 README 中虚构一个统一版本。

### 7.2 在集群上确认实际版本

```bash
which lmp
lmp -h | head -n 20
```

如果站点使用 module：

```bash
module avail lammps
module load lammps/<SITE_VERSION>
which lmp
lmp -h | head -n 20
```

建议把输出保存进每次运行的 provenance。

### 7.3 验证 MLIP 接口

DeepMD 示例：

```bash
lmp -h | grep -i deepmd
```

MACE-LAMMPS 的接口取决于实际导出方式和 LAMMPS build。不要仅凭“LAMMPS 能启动”就认为模型插件可用。

最小验证顺序：

```text
1. lmp 本身可启动
2. 目标 pair_style/插件出现在 build 中
3. 模型文件可加载
4. 1-2 step 单结构 smoke 成功
5. 能量/力单位与独立 ASE/框架推理一致
6. 再运行长 MD
```

### 7.4 推荐 site manifest

```yaml
lammps:
  executable: /apps/lammps/<build>/bin/lmp
  version_source: lmp-h
  build_id: <fill-after-validation>
  mpi: true
  packages:
    - <required-package>
  model_interface:
    family: deepmd
    pair_style: deepmd
```

不要把这个示例中的占位符原样用于生产。

---

## 8. LASP / SSW

MLIPFlow 不分发 LASP。

`pes-sampling` 当前支持：

```text
direct-select
lasp-ssw-execute
lasp-ssw-normalize-replay
```

`lasp-ssw-execute` 要求显式提供：

```text
LASP executable
lasp_version
input.arc
lasp.in
必要辅助文件
```

如使用 MPI，还必须提供真实、可执行且 basename 为 `mpirun` 或 `mpiexec` 的 launcher 路径以及 `mpi_processes`。

示例项目参数：

```yaml
inputs:
  lasp_executable: /apps/lasp/bin/lasp
  input_structure: inputs/input.arc
  lasp_input: inputs/lasp.in

parameters:
  operation: lasp-ssw-execute
  lasp_version: '<SITE_LASP_VERSION>'
  seed_status: HISTORICAL_PARAMETER_UNKNOWN
  acknowledge_uncontrolled_seed: true
  preserve_historical_order: true

resources:
  python_executable: /path/to/python
  mpi_launcher: /usr/bin/mpirun
```

当前历史 LASP 源没有可恢复 seed，因此不要人为补一个“历史随机种子”。

---

## 9. SLURM 配置

### 9.1 当前真实边界

核心 `SlurmBackend` 已实现：

```text
sbatch
scancel
squeue
sacct
```

但当前内置科学 Adapter 只声明 `local`，因此不要绕过 Adapter 限制直接把现有节点强制改成 `slurm` 后宣称科学闭环已支持。

下面的 `sbatch` 模板主要用于：

- 你自己的外部 wrapper；
- 站点环境验证；
- 后续为科学 Adapter 增加受测 scheduler support。

### 9.2 CPU 作业模板

```bash
#!/bin/bash
#SBATCH --job-name=mlipflow-cpu
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

source ~/.bashrc
conda activate mlipflow-mace

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python your_wrapper.py --config config.yaml
```

### 9.3 GPU 作业模板

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

source ~/.bashrc
conda activate mlipflow-mace

python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
python your_wrapper.py --config config.yaml
```

### 9.4 NPU 作业模板

仓库目前没有经过验证的 NPU backend，因此不提供一个假装可直接使用的生产 `sbatch` 模板。

如果集群管理员已经提供 Ascend/NPU module，应先在独立实验 wrapper 中验证 runtime，再把准确的资源字段和 module 命令加入站点文档。

### 9.5 VASP CPU/MPI 模板

```bash
#!/bin/bash
#SBATCH --job-name=vasp-label
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
module load <MPI_MODULE>
module load <VASP_MODULE>

srun vasp_std
```

实际核数、KPAR/NCORE、partition、account、QoS 和 walltime 必须按你的集群和体系测试，不要直接复制这里的数值作为生产参数。

---

## 10. SSH 密钥与远程集群

### 10.1 MLIPFlow 不保存密钥

`project.yaml` 中只允许引用一个 `~/.ssh/config` 别名，例如：

```yaml
backend_profiles:
  cluster-a:
    ssh_profile: cluster-a
```

不要在项目文件中写：

```text
password
private key text
IdentityFile path
token
API key
```

### 10.2 生成 SSH key

如果站点允许公钥认证，可在本机生成：

```bash
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519_hpc -C 'mlipflow-hpc'
```

私钥权限：

```bash
chmod 600 ~/.ssh/id_ed25519_hpc
```

将公钥按集群管理员要求安装到远端账号。不要把私钥复制进仓库。

### 10.3 `~/.ssh/config` 示例

```sshconfig
Host cluster-a
    HostName <LOGIN_HOST>
    User <USERNAME>
    IdentityFile ~/.ssh/id_ed25519_hpc
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

如果站点需要跳板机：

```sshconfig
Host cluster-a
    HostName <INTERNAL_LOGIN_HOST>
    User <USERNAME>
    IdentityFile ~/.ssh/id_ed25519_hpc
    ProxyJump <BASTION_ALIAS>
```

MLIPFlow 项目中仍然只写：

```yaml
ssh_profile: cluster-a
```

### 10.4 先做只读测试

```bash
ssh cluster-a 'hostname'
ssh cluster-a 'which sbatch && which squeue && which sacct'
```

真实 HPC 集成验证必须从秒级、无科学意义的 tiny job 开始，不要直接用 DFT/训练/长 MD 当连接测试。

---

## 11. 推荐的站点配置文件

不要把集群专用路径散落在插件源码里。建议自己维护一个**不提交到公开仓库**的站点清单，例如：

```yaml
site: my-hpc

python:
  mlipflow: /home/user/miniconda3/envs/mlipflow/bin/python
  sqs: /home/user/miniconda3/envs/mlipflow-sqs/bin/python
  mace: /home/user/miniconda3/envs/mlipflow-mace/bin/python

scheduler:
  type: slurm
  cpu_partition: <CPU_PARTITION>
  gpu_partition: <GPU_PARTITION>
  account: <ACCOUNT>
  qos: <QOS>

ssh:
  profile: cluster-a

vasp:
  executable: /apps/vasp/bin/vasp_std
  module: <VASP_MODULE>
  pseudopotential_root: /data/pseudopotentials

lammps:
  executable: /apps/lammps/bin/lmp
  build_id: <RECORD_AFTER_lmp_-h>

lasp:
  executable: /apps/lasp/bin/lasp
  version: <SITE_LASP_VERSION>

mpi:
  launcher: /usr/bin/mpirun
```

真正进入 MLIPFlow 项目的配置只引用安全的逻辑名称、相对输入和经过批准的必要路径。

---

## 12. `project.yaml` 基本结构

项目 schema 支持：

```yaml
schema_version: 1

project:
  id: demo
  name: Demo project

plugin_paths:
  - plugins

locations: {}

backend_profiles:
  cluster-a:
    ssh_profile: cluster-a

model_registry: model_registry.yaml

state:
  database_path: .mlipflow/state.sqlite3

workflow:
  nodes:
    - id: structure
      uses: high-entropy-structure@0
      mode: execute
      backend: local
      inputs: {}
      parameters: {}
      resources: {}

safety:
  auto_submit: false
  auto_advance: false
  require_approval_for_expensive: true
  require_approval_for_destructive: true
```

支持的 backend 名称：

```text
local
slurm
ssh-slurm
```

但“schema 支持某个 backend 名称”不等于“每个科学 Adapter 已允许该 backend”。以对应 `plugins/*/plugin.yaml` 中的 `execution.backends` 为准。

---

## 13. 推荐的首次使用流程

### 13.1 先跑纯离线 replay

```bash
DEMO_ROOT="$(mktemp -d)"
cp -R examples/high_entropy_sulfide "$DEMO_ROOT/"
DEMO_PROJECT="$DEMO_ROOT/high_entropy_sulfide"

mlipflow --project "$DEMO_PROJECT" list
mlipflow --project "$DEMO_PROJECT" status
mlipflow --project "$DEMO_PROJECT" doctor
mlipflow --project "$DEMO_PROJECT" init
```

### 13.2 看计划，不执行

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay --dry-run
```

输出会包含精确 `plan_digest`。

### 13.3 只批准这一份计划

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay --approve 'sha256:<PLAN_DIGEST>'
```

### 13.4 推进依赖状态

```bash
mlipflow --project "$DEMO_PROJECT" advance --dry-run
```

然后使用该计划的新摘要执行：

```bash
mlipflow --project "$DEMO_PROJECT" advance --approve 'sha256:<PLAN_DIGEST>'
```

MLIPFlow 不会因为一个节点成功就自动执行下一个节点。

---

## 14. 从本地 CPU 到集群的推荐验证顺序

不要直接从“安装完成”跳到“生产 60k DFT/长 MD/完整训练”。推荐顺序：

```text
1. offline replay
2. mlipflow doctor
3. local lightweight post-processing
4. local SQS tiny smoke
5. local framework import/version check
6. local model single-point inference
7. local tiny MD
8. SSH read-only connectivity
9. SLURM tiny echo/python job
10. remote stage/fetch tiny files
11. scheduler completion + local scientific checker
12. tiny scientific cluster job
13. small representative workload
14. production workload
```

任何一步失败都应停止扩容并先修复环境或接口。

---

## 15. `doctor` 与环境自检

先运行：

```bash
mlipflow --project <PROJECT> doctor
```

另外建议为科学环境建立站点检查脚本，例如：

```bash
python --version
python -c 'import ase; print(ase.__version__)'
python -c 'import numpy; print(numpy.__version__)'
which mpirun || true
which sbatch || true
which squeue || true
which sacct || true
which lmp || true
```

MACE 环境：

```bash
python -c 'import torch, mace; print(torch.__version__); print(torch.cuda.is_available())'
```

NPU 扩展环境在真正接入前应至少独立验证：

```text
PyTorch import
torch_npu import
NPU availability
single tensor operation
single model forward
```

但这些检查目前不属于仓库内置功能。

---

## 16. 安全与审批语义

严格只读命令：

```text
list
status
json
inspect
logs
route
doctor
```

可能改变状态或调用后端：

```text
init
run
advance
retry
stop
```

除 `init` 外，应先：

```bash
mlipflow ... --dry-run
```

再使用完全匹配的摘要：

```bash
mlipflow ... --approve 'sha256:...'
```

以下行为应单独审批：

```text
远端 staging
sbatch
scancel
DFT/AIMD
长 MD
训练/微调
大规模筛选
覆盖或删除数据
```

调度器显示 `COMPLETED` 仍不等于科学结果 `OK`。必须继续通过插件的结构化 `check`/`collect`。

---

## 17. 当前已经真实执行过的本地环境证据

仓库报告中有两类重要 smoke：

### SQS smoke

```text
Python 3.11.14
ASE 3.28.0
icet 3.2
CPU/local
seed 23
```

这是 100-step 有界 integration smoke，不是 production SQS 收敛证明。

### MACE MD → transport smoke

```text
Python 3.12.12
ASE 3.28.0
MACE 0.3.15
PyTorch 2.10.0
NumPy 2.4.3
Pandas 2.3.3
Plotly 6.6.0
device = cpu
default_dtype = float64
```

每个温度只有 10 个 production step，仅用于验证软件交接，不具有科学收敛意义。

这些版本可以作为“已知至少跑通过一次的环境参考”，不是全项目统一 pin。

---

## 18. 真实 HPC 状态

当前：

```text
REAL_HPC_INTEGRATION = EXTERNAL_VALIDATION_PENDING
```

已经有：

```text
local backend
SLURM backend abstraction
SSH+SLURM backend abstraction
job ID parsing
squeue/sacct status
scancel
mock/local fixture tests
```

尚未完成真实生产闭环：

```text
remote staging
real sbatch submission
real queue monitoring
remote result fetch
local scientific re-check
production scientific job
```

因此当前 README 不声称“集群已经接通”或“NPU 集群可以直接运行”。

---

## 19. 常见错误

### `ModuleNotFoundError`

先确认当前环境：

```bash
which python
python --version
python -m pip list
```

不要在一个环境中安装了 MLIPFlow，却从另一个环境调用 wrapper。

### `sbatch: command not found`

说明当前节点不是 SLURM login/compute 环境，或 module/PATH 未初始化。

### `sacct` 查不到已结束作业

可能是站点没有启用 accounting，或需要集群特定选项。真实支持前必须在该站点验证。

### LAMMPS 找不到 `pair_style`

说明当前 `lmp` build 没包含所需 MLIP 接口。先检查：

```bash
lmp -h
```

再核对 DeepMD/MACE/其他后端的实际 build 方式。

### VASP 找不到 POTCAR

检查你的站点赝势路径、pymatgen 配置和元素 label。MLIPFlow 不会下载 POTCAR。

### `device=npu` 失败

这是预期行为：当前 MLIPFlow 没有 NPU 支持声明。

### SSH 后端拒绝 key/path 参数

这是安全设计。把认证配置移到 `~/.ssh/config`，项目中只引用 alias。

---

## 20. 开发与测试

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

## 21. 许可

MLIPFlow 使用 Apache License 2.0。

外部软件、数据、模型和势函数仍受各自许可证约束。安装 MLIPFlow 不会自动获得 VASP、LASP、POTCAR、第三方模型或数据集的使用权。

---

## 22. 给新用户的一句话版本

如果你第一次使用 MLIPFlow：

```text
先建独立 Python 环境 -> 安装 MLIPFlow -> 跑 offline replay -> 配置你自己的科学环境 ->
在 CPU 上做 tiny smoke -> 再验证 SSH/SLURM -> 最后才做真实 DFT/训练/长 MD。
```

**不要把 NPU、LAMMPS 插件、VASP 赝势或真实集群当作“安装 MLIPFlow 后自动就有”的能力。它们必须由站点单独配置、记录版本并完成验证。**
