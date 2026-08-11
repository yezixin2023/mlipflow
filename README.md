# MLIPFlow

MLIPFlow 是一个面向机器学习原子势（MLIP）研究的确定性工作流层。它把结构生成、PES 采样、DFT 标注、模型训练、基准评估、离子输运、成分筛选和电化学电压组织成显式 DAG，并记录状态、审批、版本、输入、输出、模型与来源。

当前版本为 **0.1.0 alpha**。

这份 README 的目标不是只介绍软件，而是帮助你按下面的顺序真正把环境跑通：

```text
本地离线流程
-> 本地 CPU 科学 smoke
-> 本地 GPU smoke（如果本机有 NVIDIA GPU）
-> SSH 登录集群
-> 集群 CPU 计算节点 smoke
-> 集群 GPU 计算节点 smoke
-> VASP / POTCAR / LAMMPS / LASP 等站点程序验证
-> SLURM tiny job
-> 小规模真实科学任务
-> 生产计算
```

> **重要边界**
>
> - MLIPFlow 不是 VASP、LAMMPS、LASP、DeepMD、MACE、CHGNet 或 M3GNet 的替代品。
> - 仓库不会分发 VASP、POTCAR、LASP、模型权重、私钥、token 或集群账号信息。
> - 核心代码已经实现 `local`、`slurm`、`ssh-slurm` 后端边界，但当前内置科学 Adapter 仍只声明 `local`。
> - 当前真实 HPC 的 stage -> `sbatch` -> monitor -> fetch -> scientific check 全闭环仍是 `EXTERNAL_VALIDATION_PENDING`。
> - **这不妨碍你现在在集群计算节点上运行 MLIPFlow。** 目前最稳妥的方式是先用 `salloc`/`srun` 获得 CPU 或 GPU 计算节点，再在计算节点中使用 MLIPFlow 的 `local` Adapter。
> - `ionic-transport` 当前设备接口接受 `cpu`、`cuda`、`mps`；Linux NVIDIA GPU 集群使用 `cuda`。
> - DFT、长 MD、训练、大规模筛选和真实集群作业都应从 tiny smoke 开始，不要直接跳到生产规模。

---

## 1. 工作流与当前能力

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
| `pes-sampling` | DIRECT、LASP/SSW execute/replay | Adapter 仅 `local`；LASP 由用户提供 |
| `dft-labeling` | DFT 标注与数据集装配 | 用户自备 VASP/脚本；Adapter 仅 `local` |
| `mlip-training` | DeepMD/M3GNet/CHGNet/MACE wrapper contract | 用户自备框架环境；Adapter 仅 `local` |
| `mlip-benchmark` | MAE/RMSE/Pearson、历史 benchmark 归一化 | 本地可执行 |
| `ionic-transport` | 轨迹/MSD 后处理、极小 ASE-MD smoke | 本地可执行；设备 `cpu/cuda/mps` |
| `composition-screening` | 已有候选指标排序与 top-k | 本地可执行 |
| `electrochemical-voltage` | 总能序列电压后处理、论文证据 replay | 本地可执行 |

详细状态见：

- `docs/IMPLEMENTATION_STATUS.md`
- `docs/SCIENTIFIC_VALIDATION.md`
- `docs/HPC_VALIDATION.md`
- `docs/CODEBASE_INVENTORY.md`

---

## 2. 最重要的环境划分

不要把所有软件塞进一个 Conda 环境。建议把 **控制环境** 和 **科学环境** 分开。

### 2.1 本地电脑

```text
mlipflow-control
  MLIPFlow
  PyYAML
  pytest/ruff（开发时）

mlipflow-sqs
  ASE
  icet
  pymatgen

mlipflow-mace-cpu
  PyTorch CPU
  MACE
  ASE

mlipflow-deepmd-cpu
  DeePMD-kit CPU

可选本地 GPU 环境
  mlipflow-mace-gpu
  mlipflow-deepmd-gpu
```

如果本地只想先验证 MLIPFlow 本身，第一阶段只需要 `mlipflow-control`。

### 2.2 集群

建议在共享 HOME/PROJECT 文件系统中建立：

```text
mlipflow-control
mlipflow-sqs
mlipflow-mace-cpu
mlipflow-mace-gpu
mlipflow-deepmd-cpu
mlipflow-deepmd-gpu
mlipflow-chgnet-gpu        # 只有需要 CHGNet 时安装
mlipflow-m3gnet-gpu        # 只有需要 M3GNet/MatGL 时安装
```

站点程序由集群提供或单独安装：

```text
SLURM
MPI
NVIDIA driver
CUDA module/runtime（按站点要求）
VASP
POTCAR/pseudopotential library
LAMMPS CPU build
LAMMPS GPU build
LASP
```

**Conda 环境不要负责安装 NVIDIA kernel driver。** GPU 驱动属于计算节点操作系统/集群管理员管理的范围。

---

## 3. 本地：先把 MLIPFlow 控制层跑通

### 3.1 Python 要求

`pyproject.toml` 当前要求：

```text
Python >= 3.9
PyYAML >= 6.0
```

建议使用 Python 3.11。

### 3.2 创建控制环境

```bash
conda create -n mlipflow-control python=3.11 -y
conda activate mlipflow-control
python -m pip install --upgrade pip
python -m pip install -e .
```

开发时：

```bash
python -m pip install -e '.[dev]'
```

需要仓库通用科学依赖时：

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

### 3.3 基本检查

```bash
python --version
python -c 'import mlipflow; print(mlipflow.__version__)'
mlipflow --version
mlipflow --project examples/high_entropy_sulfide doctor
python -m pytest
```

### 3.4 纯离线 replay

```bash
DEMO_ROOT="$(mktemp -d)"
cp -R examples/high_entropy_sulfide "$DEMO_ROOT/"
DEMO_PROJECT="$DEMO_ROOT/high_entropy_sulfide"
mlipflow --project "$DEMO_PROJECT" list
mlipflow --project "$DEMO_PROJECT" status
mlipflow --project "$DEMO_PROJECT" doctor
mlipflow --project "$DEMO_PROJECT" init
mlipflow --project "$DEMO_PROJECT" run structure-replay --dry-run
```

把 `--dry-run` 输出中的完整 `sha256:...` 复制出来：

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay --approve 'sha256:<PLAN_DIGEST>'
```

推进依赖仍然需要单独审批：

```bash
mlipflow --project "$DEMO_PROJECT" advance --dry-run
mlipflow --project "$DEMO_PROJECT" advance --approve 'sha256:<PLAN_DIGEST>'
```

---

## 4. 本地：SQS 环境

推荐独立环境：

```bash
conda create -n mlipflow-sqs python=3.11 -y
conda activate mlipflow-sqs
python -m pip install --upgrade pip
python -m pip install ase icet pymatgen
```

检查：

```bash
python -c 'import ase, icet, pymatgen; print(ase.__version__); print(icet.__version__)'
```

仓库已有真实 bounded smoke 使用过：

```text
Python 3.11.14
ASE 3.28.0
icet 3.2
CPU
seed = 23
```

这些版本证明至少有一套环境跑通过，不是全项目强制 pin。

---

## 5. 本地：MACE CPU/GPU 环境

MACE 官方安装方式是先安装与机器匹配的 PyTorch，再安装 `mace-torch`。不要先随便装一套 CUDA PyTorch 再猜是否兼容。

### 5.1 CPU 环境

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
python -c 'import mace; print(mace.__version__ if hasattr(mace,"__version__") else "mace imported")'
```

CPU 环境预期 `torch.cuda.is_available()` 为 `False`。

### 5.2 NVIDIA GPU 环境

只有本机确实有 NVIDIA GPU 时才建立。

先检查主机驱动：

```bash
nvidia-smi
```

然后根据 PyTorch 官方安装选择器，安装与当前 NVIDIA 驱动兼容的 CUDA wheel。不要把 README 中某个 CUDA 小版本永久当作要求。

环境框架：

```bash
conda create -n mlipflow-mace-gpu python=3.11 -y
conda activate mlipflow-mace-gpu
python -m pip install --upgrade pip
python -m pip install <PYTORCH_CUDA_COMMAND_FROM_OFFICIAL_SELECTOR>
python -m pip install mace-torch ase
```

GPU 检查：

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA GPU")'
```

只有输出 `True` 并显示正确 GPU 型号，才算 Python GPU 环境真正接通。

仓库已经记录过一次 MACE CPU integration smoke：

```text
Python 3.12.12
ASE 3.28.0
MACE 0.3.15
PyTorch 2.10.0
CPU float64
```

这仍只是已知工作环境，不表示 GPU 必须使用相同组合。

---

## 6. 本地：DeepMD CPU/GPU 环境

DeepMD 建议与 MACE 分开环境。

### 6.1 CPU

```bash
conda create -n mlipflow-deepmd-cpu python=3.11 -y
conda activate mlipflow-deepmd-cpu
python -m pip install --upgrade pip
python -m pip install 'deepmd-kit[cpu]'
```

检查：

```bash
dp -h
python -c 'import deepmd; print(deepmd.__version__)'
```

### 6.2 NVIDIA GPU

当前 DeePMD-kit 官方文档提供 CUDA 预编译安装方式，例如 CUDA 12 路线可使用 GPU extra。实际选择仍必须与集群驱动/站点模块匹配。

```bash
conda create -n mlipflow-deepmd-gpu python=3.11 -y
conda activate mlipflow-deepmd-gpu
python -m pip install --upgrade pip
python -m pip install 'deepmd-kit[gpu,cu12]'
```

如果还希望该环境同时提供 DeePMD 的 LAMMPS 集成，可按 DeePMD 官方对应版本文档使用 `lmp` extra；不要和站点另一个不匹配的 LAMMPS build 混用。

检查：

```bash
dp -h
python -c 'import deepmd; print(deepmd.__version__)'
```

GPU 是否真正工作必须在有 GPU 的机器/计算节点上验证，而不是只看安装成功。

---

## 7. CHGNet、M3GNet/MatGL

这几套框架的版本/API 漂移比 MLIPFlow 核心更快。建议各建独立环境，而且只在真正要跑对应模型时安装。

```text
mlipflow-chgnet-cpu
mlipflow-chgnet-gpu
mlipflow-m3gnet-cpu
mlipflow-m3gnet-gpu
```

每个 wrapper 必须记录：

```text
Python version
framework name/version
PyTorch/TensorFlow backend version
CUDA runtime（GPU 时）
GPU model（GPU 时）
device
precision
seed
dataset SHA-256
config SHA-256
model SHA-256
```

不要依赖“latest”作为科学 provenance。

---

## 8. 连接集群前先收集站点信息

在真正配置之前，你需要从管理员文档或登录节点明确以下信息：

```text
登录域名
用户名
是否需要 VPN / 跳板机
CPU partition 名
GPU partition 名
是否必须填写 account
是否必须填写 qos
GPU 型号
GPU 请求语法（--gres=gpu:1 / --gpus=1 / 其他站点约定）
CPU 每节点数量
内存规则
HOME quota
scratch/project 路径
module 名称
CUDA module
MPI module
VASP module/path
LAMMPS module/path
LASP path
作业最长 walltime
```

在登录节点运行：

```bash
hostname
uname -a
which sbatch
which srun
which squeue
which sacct
sinfo
scontrol show config | head
module avail 2>&1 | head -n 80
```

如果站点限制 `scontrol` 或 `module avail`，按管理员文档为准。

---

## 9. SSH：从本地连接集群

### 9.1 本地生成集群登录 key

```bash
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519_hpc -C 'mlipflow-hpc'
chmod 600 ~/.ssh/id_ed25519_hpc
```

把 `.pub` 公钥按集群要求加入远端账号。私钥不要上传 GitHub、不要放项目目录。

### 9.2 本地 `~/.ssh/config`

```sshconfig
Host mlip-cluster
    HostName <LOGIN_HOST>
    User <USERNAME>
    IdentityFile ~/.ssh/id_ed25519_hpc
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

需要跳板机时：

```sshconfig
Host mlip-cluster
    HostName <INTERNAL_LOGIN_HOST>
    User <USERNAME>
    IdentityFile ~/.ssh/id_ed25519_hpc
    ProxyJump <BASTION_ALIAS>
```

验证：

```bash
ssh mlip-cluster
```

登录后只读检查：

```bash
hostname
which sbatch
which srun
which squeue
which sacct
```

MLIPFlow 的 `ssh-slurm` 配置只引用 `mlip-cluster` 这个 alias，不保存密码或私钥。

---

## 10. 把私有仓库放到集群

推荐把代码放在共享 HOME 或 PROJECT 路径，让登录节点和计算节点都能看到。

目录示例：

```text
$HOME/src/mlipflow                 # 代码
$HOME/.conda/envs/...              # 小型 Conda 环境，若 HOME quota 允许
$PROJECT/mlipflow/envs/...         # 大环境/模型，若站点推荐 project FS
$PROJECT/mlipflow/models/...       # 模型
$PROJECT/mlipflow/projects/...     # 项目配置
$SCRATCH/mlipflow-runs/...         # 大量临时运行结果
```

如果集群允许访问 GitHub，可在集群单独配置 GitHub SSH key，然后：

```bash
git clone git@github.com:yezixin2023/mlipflow.git "$HOME/src/mlipflow"
cd "$HOME/src/mlipflow"
git checkout docs/readme-setup-hpc-guide
```

如果计算中心不允许直接访问 GitHub，则从本地安全地 `rsync` 代码到集群。不要把本地私钥一起同步。

---

## 11. 集群：先建立控制环境

控制环境不需要 GPU，也不需要在 GPU 节点安装。

如果集群已有 Conda/Mamba module：

```bash
module load <CONDA_OR_MAMBA_MODULE>
```

否则使用站点允许的 Miniforge/Miniconda。

创建环境：

```bash
conda create -n mlipflow-control python=3.11 -y
conda activate mlipflow-control
cd "$HOME/src/mlipflow"
python -m pip install --upgrade pip
python -m pip install -e .
```

检查：

```bash
mlipflow --version
mlipflow --project examples/high_entropy_sulfide doctor
```

**不要在登录节点运行训练、MD、VASP、LASP 或大 SQS。** 登录节点只用于配置、编辑、提交和轻量检查。

---

## 12. 集群：CPU 科学环境

CPU 环境和本地逻辑相同，但必须安装在计算节点可见的共享路径。

### SQS

```bash
conda create -n mlipflow-sqs python=3.11 -y
conda activate mlipflow-sqs
python -m pip install ase icet pymatgen
```

### MACE CPU

```bash
conda create -n mlipflow-mace-cpu python=3.11 -y
conda activate mlipflow-mace-cpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install mace-torch ase
```

### DeepMD CPU

```bash
conda create -n mlipflow-deepmd-cpu python=3.11 -y
conda activate mlipflow-deepmd-cpu
python -m pip install 'deepmd-kit[cpu]'
```

CPU 框架 smoke 应放在 CPU 计算节点执行，而不是登录节点长期跑。

---

## 13. 集群：GPU 科学环境

### 13.1 先申请 GPU 计算节点

不同站点参数不同。典型形式：

```bash
salloc --partition=<GPU_PARTITION> --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=01:00:00
```

获得 allocation 后进入计算节点 shell：

```bash
srun --pty bash
```

如果站点使用 `--gpus=1`，按管理员规定替换 `--gres=gpu:1`。Slurm 的 GPU GRES/TRES 配置由站点决定。

### 13.2 在计算节点检查 GPU

```bash
hostname
nvidia-smi
printf '%s\n' "$CUDA_VISIBLE_DEVICES"
```

Slurm 通常通过 `CUDA_VISIBLE_DEVICES` 限定当前 job step 可见 GPU。

### 13.3 加载站点 CUDA（如果需要）

```bash
module purge
module load <CUDA_MODULE>
```

是否需要显式 `module load cuda/...` 取决于站点。PyTorch wheel 自带用户态 CUDA runtime 的场景下仍然必须有兼容的 NVIDIA 驱动。

### 13.4 MACE GPU 环境

可以在登录节点安装环境，也可以在交互 GPU allocation 中安装；环境最终必须位于所有计算节点可见的文件系统。

```bash
conda create -n mlipflow-mace-gpu python=3.11 -y
conda activate mlipflow-mace-gpu
python -m pip install --upgrade pip
python -m pip install <PYTORCH_CUDA_COMMAND_FROM_OFFICIAL_SELECTOR>
python -m pip install mace-torch ase
```

在 GPU 计算节点验证：

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA GPU")'
```

再做一个真实模型单点 inference，然后才做 tiny MD。

### 13.5 DeepMD GPU 环境

```bash
conda create -n mlipflow-deepmd-gpu python=3.11 -y
conda activate mlipflow-deepmd-gpu
python -m pip install --upgrade pip
python -m pip install 'deepmd-kit[gpu,cu12]'
```

这条命令是当前官方 CUDA 12 预编译路线之一；若集群环境需要不同后端/版本，按对应 DeePMD 官方版本文档调整。

GPU 计算节点验证：

```bash
dp -h
python -c 'import deepmd; print(deepmd.__version__)'
```

随后用一个极小模型/输入做 inference，再进行 LAMMPS smoke。

---

## 14. 当前最推荐的“集群运行 MLIPFlow”方式

当前科学 Adapter 只声明 `local`。因此现在要真正使用集群 CPU/GPU，推荐：

```text
本地电脑
  -> ssh 登录节点
  -> salloc 请求 CPU/GPU 资源
  -> srun --pty bash 进入计算节点
  -> 激活对应科学环境
  -> 在这个计算节点上运行 MLIPFlow
  -> MLIPFlow 仍使用 backend: local
```

这里的 `local` 指的是“相对于当前计算节点本地执行”，不是“必须在你的笔记本上执行”。

这是当前最容易验证、也最符合现有 Adapter 能力的路线。

### CPU allocation 示例

```bash
salloc --partition=<CPU_PARTITION> --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G --time=01:00:00
srun --pty bash
```

进入节点后：

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate mlipflow-sqs
cd <PROJECT_DIRECTORY>
python -c 'import ase, icet; print(ase.__version__, icet.__version__)'
```

### GPU allocation 示例

```bash
salloc --partition=<GPU_PARTITION> --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=01:00:00
srun --pty bash
```

进入节点后：

```bash
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate mlipflow-mace-gpu
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

完成这些再调用真正 wrapper/MLIPFlow 节点。

---

## 15. VASP 与赝势/POTCAR

MLIPFlow **不会保存、下载或生成 POTCAR**。VASP 可执行文件、许可证和赝势库由用户/机构管理。

### 15.1 站点目录示例

```text
/apps/vasp/6.x/vasp_std
/apps/vasp/6.x/vasp_gam
/project/pseudopotentials/potpaw_PBE/...
```

真实路径按你的集群修改。

### 15.2 pymatgen POTCAR 配置

当前 pymatgen 推荐用 `pmg config` 管理 `PMG_VASP_PSP_DIR`。例如先把机构合法获得的 VASP POTCAR 目录整理成 pymatgen 格式：

```bash
pmg config -p /path/to/original/potpaw_PBE /path/to/pmg_potcars
pmg config --add PMG_VASP_PSP_DIR /path/to/pmg_potcars
```

检查配置：

```bash
pmg config --help
```

不要提交 `.pmgrc.yaml` 中的私人绝对路径到公共仓库。

### 15.3 Li-M-P-S 体系必须记录实际 label

配置格式示例：

```yaml
pseudopotentials:
  functional: PBE
  species:
    Li: <ACTUAL_LI_LABEL>
    Mn: <ACTUAL_MN_LABEL>
    Fe: <ACTUAL_FE_LABEL>
    Ni: <ACTUAL_NI_LABEL>
    Cu: <ACTUAL_CU_LABEL>
    Zn: <ACTUAL_ZN_LABEL>
    P: <ACTUAL_P_LABEL>
    S: <ACTUAL_S_LABEL>
```

不要因为 README 示例就擅自决定 `Mn`/`Mn_pv` 或 `Fe`/`Fe_pv`。如果目标是复现实验/论文历史计算，应以历史 INCAR/POTCAR provenance 或论文方法为准。

每种赝势建议记录：

```text
element
POTCAR label
functional
release/header identity
SHA-256
```

POTCAR 本体不进入仓库。

### 15.4 VASP CPU/MPI 先做 tiny job

先验证：

```bash
which vasp_std
```

如果使用 module：

```bash
module load <MPI_MODULE>
module load <VASP_MODULE>
which vasp_std
```

不要直接用正式结构验证环境。先用允许的极小测试体系确认 MPI、POTCAR、输出与收敛检查链路。

---

## 16. DFT wrapper 的职责

当前 `dft-labeling` 需要用户自备 wrapper。wrapper 至少应完成：

1. 读取 structures manifest；
2. 生成 POSCAR/INCAR/KPOINTS；
3. 按显式映射从站点赝势库组装 POTCAR；
4. 调用/准备 VASP；
5. 检查电子收敛；
6. 对 relax 检查离子收敛；
7. 检查输出是否截断；
8. 明确能量、力、应力、长度单位与应力符号约定；
9. 生成标准结果 manifest；
10. 对 dataset/artifact 写 SHA-256。

不能仅通过 OUTCAR 中出现某个字符串就认定成功。

---

## 17. LAMMPS：不要把不同 MLIP build 混为一谈

MLIPFlow 当前没有固定一个通用 LAMMPS 版本，因为 DeepMD 和 MACE 的 LAMMPS 集成方式不同。

### 17.1 任何节点先记录实际 build

```bash
which lmp
lmp -h | head -n 30
```

生产 provenance 至少记录：

```text
lmp absolute path
LAMMPS version/build
compiler
MPI
CPU/GPU build
required package/pair_style
MLIP framework version
model SHA-256
```

### 17.2 DeepMD + LAMMPS

DeePMD 官方支持 built-in mode 和 plugin mode。当前官方文档说明 plugin mode 可加载 `libdeepmd_lmp.so`，较新的 LAMMPS 也可以通过 `LAMMPS_PLUGIN_PATH` 找插件。

先检查：

```bash
lmp -h | grep -i deepmd
```

如果是 plugin mode，还要确认对应 DeePMD plugin 路径和动态库依赖。

DeepMD 常用 LAMMPS `units metal`，其内部距离/能量/力单位与 `metal` 对应；正式计算仍应在 manifest 中记录 units。

### 17.3 MACE + LAMMPS

MACE 官方文档提供专门的 LAMMPS build/`ML-MACE` 路线。CPU 与 GPU build 不是同一套编译参数；GPU 通常结合 Kokkos/CUDA。

正式使用前顺序：

```text
MACE ASE 单点成功
-> 导出 LAMMPS model
-> LAMMPS 能识别 MACE pair style/package
-> 单结构 0/1 step smoke
-> ASE 与 LAMMPS 能量/力对照
-> tiny MD
-> 长 MD
```

MACE 官方当前文档仍特别提醒先对比 LAMMPS 模型与等价 ASE calculator。

### 17.4 不要只做 `lmp` 启动测试

真正合格的 MLIP-LAMMPS smoke 至少包括：

```text
lmp 可启动
目标 pair_style/package 可见
模型可以加载
元素 type map 正确
units 正确
1-2 step 可执行
能量/力与受信独立计算器一致
GPU build 时确实使用分配到的 GPU
```

---

## 18. LASP / SSW

MLIPFlow 不分发 LASP。

`pes-sampling` 当前操作：

```text
direct-select
lasp-ssw-execute
lasp-ssw-normalize-replay
```

`lasp-ssw-execute` 需要显式提供：

```text
LASP executable
lasp_version
input.arc
lasp.in
必要辅助文件
```

MPI 时还需提供：

```text
mpirun 或 mpiexec 的真实可执行文件
mpi_processes
```

配置示例：

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

历史 LASP 源没有可恢复 seed，不要伪造历史随机种子。

---

## 19. SLURM：CPU tiny job

先测试调度器，不跑科学程序。

`slurm_cpu_smoke.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=mlip-cpu-smoke
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate mlipflow-control
hostname
python --version
mlipflow --version
```

提交：

```bash
sbatch slurm_cpu_smoke.sbatch
squeue -u "$USER"
```

结束后：

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat slurm-<JOB_ID>.out
cat slurm-<JOB_ID>.err
```

如果站点未启用 `sacct`，使用管理员提供的 accounting 方法。

---

## 20. SLURM：GPU tiny job

`slurm_gpu_smoke.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=mlip-gpu-smoke
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate mlipflow-mace-gpu
nvidia-smi
printf '%s\n' "$CUDA_VISIBLE_DEVICES"
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA GPU")'
```

提交：

```bash
sbatch slurm_gpu_smoke.sbatch
```

只有 `torch.cuda.is_available()` 为 `True` 才继续做模型 inference。

Slurm 支持 `--gres=gpu:...`、`--gpus`、`--gpus-per-node` 等多种 GPU 请求参数，但实际可用形式取决于站点 `select/cons_tres` 和 GRES 配置；以你的集群文档为准。

---

## 21. SLURM：VASP CPU/MPI 模板

示例：

```bash
#!/bin/bash
#SBATCH --job-name=vasp-label
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=<MPI_RANKS>
#SBATCH --cpus-per-task=1
#SBATCH --mem=<MEMORY>
#SBATCH --time=<WALLTIME>
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
module purge
module load <MPI_MODULE>
module load <VASP_MODULE>
srun vasp_std
```

`MPI_RANKS`、`KPAR`、`NCORE`、节点数、内存和 walltime 必须按体系和站点 benchmark，不要直接复制别人的值。

---

## 22. SLURM：GPU Python/MLIP 模板

用于 MACE/CHGNet/M3GNet 训练或 GPU wrapper 的基本形式：

```bash
#!/bin/bash
#SBATCH --job-name=mlip-gpu
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
conda activate mlipflow-mace-gpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python your_wrapper.py --config config.yaml
```

不要在脚本里硬编码 GPU `0` 后忽略 `CUDA_VISIBLE_DEVICES`；让调度器控制当前作业可见设备。

---

## 23. LAMMPS CPU/GPU 的 sbatch 思路

LAMMPS 的具体命令取决于你的 build。

CPU/MPI 常见模式：

```bash
srun <CPU_LAMMPS_EXECUTABLE> -in in.lammps
```

GPU Kokkos 常见模式会包含 GPU/Kokkos 参数，但不要从 README 盲目复制到未知 build。应先根据你的 LAMMPS/MACE/DeepMD 官方 build 文档确认，然后把**最终已验证命令**保存到站点 profile。

特别是 MACE LAMMPS GPU，官方文档给出的实现与 Kokkos、libtorch、GPU 架构和 MACE-LAMMPS 接口绑定，必须在目标计算节点编译/验证。

---

## 24. 推荐的私有站点配置

不要把集群路径散落在插件代码里。建议自己维护一个 **不提交公共仓库** 的配置，例如 `site.local.yaml`：

```yaml
site: my-hpc
ssh:
  profile: mlip-cluster
filesystem:
  repo: /home/<USER>/src/mlipflow
  project_root: /project/<ACCOUNT>/mlipflow/projects
  model_root: /project/<ACCOUNT>/mlipflow/models
  scratch_root: /scratch/<USER>/mlipflow
scheduler:
  type: slurm
  cpu_partition: <CPU_PARTITION>
  gpu_partition: <GPU_PARTITION>
  account: <ACCOUNT>
  qos: <QOS_OR_NULL>
  gpu_request: '--gres=gpu:1'
python:
  control: /path/to/envs/mlipflow-control/bin/python
  sqs: /path/to/envs/mlipflow-sqs/bin/python
  mace_cpu: /path/to/envs/mlipflow-mace-cpu/bin/python
  mace_gpu: /path/to/envs/mlipflow-mace-gpu/bin/python
  deepmd_cpu: /path/to/envs/mlipflow-deepmd-cpu/bin/python
  deepmd_gpu: /path/to/envs/mlipflow-deepmd-gpu/bin/python
vasp:
  executable: /apps/vasp/bin/vasp_std
  module: <VASP_MODULE>
  pseudopotential_root: /project/<ACCOUNT>/pseudopotentials
lammps:
  cpu_executable: /apps/lammps/cpu/bin/lmp
  gpu_executable: /apps/lammps/gpu/bin/lmp
  cpu_build_id: <RECORD_AFTER_VALIDATION>
  gpu_build_id: <RECORD_AFTER_VALIDATION>
lasp:
  executable: /apps/lasp/bin/lasp
  version: <SITE_LASP_VERSION>
mpi:
  launcher: /usr/bin/mpirun
```

把 `site.local.yaml` 加进 `.gitignore` 或放在仓库外。

---

## 25. `project.yaml` 与 backend

Schema 支持：

```text
local
slurm
ssh-slurm
```

但是当前每个科学 Adapter 是否允许某后端，要看对应 `plugins/*/plugin.yaml` 的 `execution.backends`。

现阶段真实科学节点推荐仍写：

```yaml
workflow:
  nodes:
    - id: structure
      uses: high-entropy-structure@0
      mode: execute
      backend: local
      inputs: {}
      parameters: {}
      resources: {}
```

然后在 **Slurm 已经分配给你的计算节点里** 执行 MLIPFlow。

远端 profile 的安全格式：

```yaml
backend_profiles:
  cluster-a:
    ssh_profile: mlip-cluster
```

项目中不写密码、私钥、token。

---

## 26. 真正的 `ssh-slurm` 自动闭环目前还差什么

核心已经有：

```text
SSH alias
sbatch
scancel
squeue
sacct
scp fetch
job ID parsing
completion identity manifest logic
```

但还没有在生产集群完成一次受控的：

```text
local project
-> remote staging
-> remote sbatch
-> queue monitor
-> terminal state
-> allowlisted artifact fetch
-> SHA-256
-> local scientific check/collect
```

所以现阶段你应先把前面的手动/半手动集群流程跑通并记录真实站点字段。等 tiny HPC validation 完成后，再把这些字段固化成 `ssh-slurm` 的正式站点配置，而不是现在猜路径和资源。

---

## 27. 第一次集群验证的推荐顺序

严格按层推进：

```text
A. ssh 登录成功
B. 登录节点能看到 sbatch/srun/squeue
C. CPU salloc 成功
D. CPU 节点 python/MLIPFlow 成功
E. CPU tiny science smoke 成功
F. GPU salloc 成功
G. GPU 节点 nvidia-smi 成功
H. GPU Python torch.cuda.is_available() == True
I. GPU 单模型 inference 成功
J. CPU sbatch tiny job 成功
K. GPU sbatch tiny job 成功
L. VASP tiny MPI job 成功
M. POTCAR mapping/hash 固定
N. LAMMPS CPU pair_style smoke 成功
O. LAMMPS GPU pair_style smoke 成功
P. LASP tiny/local contract（如需要）
Q. MLIPFlow scientific wrapper tiny run
R. remote artifact fetch + hash + scientific check
S. 小规模真实体系
T. 生产计算
```

任何一步失败都先修这一层，不要继续扩大规模。

---

## 28. CPU/GPU 环境验收清单

### 本地 CPU

```bash
python --version
mlipflow --version
python -m pytest
python -c 'import ase; print(ase.__version__)'
```

### 本地 GPU（如果有）

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA GPU")'
```

### 集群 CPU 计算节点

```bash
hostname
lscpu | head
python --version
mlipflow --version
```

### 集群 GPU 计算节点

```bash
hostname
nvidia-smi
printf '%s\n' "$CUDA_VISIBLE_DEVICES"
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA GPU")'
```

### 集群外部程序

```bash
which mpirun || true
which sbatch || true
which srun || true
which lmp || true
which vasp_std || true
```

---

## 29. 科学任务开始前必须记录的版本

每次正式运行至少保存：

```text
MLIPFlow git commit
project.yaml SHA-256
plugin implementation SHA-256
Python version
framework version
PyTorch/TensorFlow version
CPU model or GPU model
NVIDIA driver
CUDA runtime（GPU 时）
MPI implementation/version
LAMMPS build/version（涉及 LAMMPS 时）
VASP version（涉及 DFT 时）
POTCAR identities + SHA-256（涉及 VASP 时）
LASP version（涉及 LASP 时）
model SHA-256
dataset/input SHA-256
seed
precision
scheduler job ID
SLURM resources
```

这样本地与集群结果才可比较。

---

## 30. CPU 与 GPU 结果要先做科学一致性检查

“GPU 跑得起来”不等于“GPU 的科学结果正确”。至少比较一个小样本：

```text
同一个结构
同一个模型文件
同一种 dtype/precision
CPU energy/forces
GPU energy/forces
允许误差阈值
```

对 MACE-LAMMPS/DeepMD-LAMMPS 还应比较：

```text
框架原生 inference
vs
LAMMPS inference
```

确认单位、元素映射和邻居设置无误后才进入长 MD。

---

## 31. MLIPFlow 审批语义

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

可能写状态或调用后端：

```text
init
run
advance
retry
stop
```

除 `init` 外先：

```bash
mlipflow ... --dry-run
```

然后：

```bash
mlipflow ... --approve 'sha256:...'
```

调度器显示 `COMPLETED` 不等于科学结果 `OK`；还必须通过插件的 `check`/`collect`。

---

## 32. 常见问题

### 登录节点 `nvidia-smi` 不存在

很多集群登录节点没有 GPU，这是正常的。先申请 GPU allocation，再在计算节点运行。

### `torch.cuda.is_available()` 为 False

依次检查：

```text
当前是否真的在 GPU 计算节点
Slurm 是否给了 GPU
nvidia-smi 是否正常
CUDA_VISIBLE_DEVICES 是否存在
是否安装了 CUDA 版 PyTorch
PyTorch wheel 与驱动是否兼容
是否激活了错误 Conda 环境
```

### `sbatch: command not found`

你可能不在 Slurm 集群环境，或 scheduler module 未加载。

### `sacct` 查不到作业

可能站点未启用 accounting 或有延迟；按站点文档处理。

### `lmp` 可以启动但 `pair_style` 不存在

当前 LAMMPS build 没编译/加载目标 MLIP 接口。重新核对 DeepMD/MACE 对应 build。

### DeepMD LAMMPS 找不到插件

检查 DeepMD/LAMMPS 是否是同一个兼容安装，plugin mode 时检查动态库和 `LAMMPS_PLUGIN_PATH`。

### VASP 找不到 POTCAR

检查机构赝势目录、pymatgen `PMG_VASP_PSP_DIR` 和元素 label 映射。MLIPFlow 不会下载 POTCAR。

### 集群 Python 环境登录节点可用、计算节点不可用

环境安装在了节点本地磁盘而不是共享文件系统，或者计算节点缺少依赖动态库/module。把环境迁到站点推荐的共享路径并在 allocation 中重新验证。

### SSH 后端拒绝 key/path 参数

这是设计行为。认证放 `~/.ssh/config`，MLIPFlow 只引用 alias。

---

## 33. 当前已有真实 smoke 证据

### SQS

```text
Python 3.11.14
ASE 3.28.0
icet 3.2
CPU/local
seed 23
```

### MACE MD -> transport

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

后者每个温度只有 10 个 production step，只证明软件交接，不具有科学收敛意义。

---

## 34. 当前 HPC 状态

```text
REAL_HPC_INTEGRATION = EXTERNAL_VALIDATION_PENDING
```

这意味着：

```text
核心 scheduler/SSH 代码存在
但生产 stage/submit/monitor/fetch/scientific-recheck 尚未完成真实站点验收
```

你的下一阶段目标应当就是按本 README 的顺序完成这次真实集群验收。

---

## 35. 最短可执行路线

如果你现在已经把本地流程跑通，下一步直接按这个顺序：

```text
1. 配好 ~/.ssh/config，确认 ssh mlip-cluster 成功
2. 在集群共享目录 clone/同步 mlipflow
3. 建 mlipflow-control
4. CPU salloc + srun 进入计算节点
5. 在 CPU 节点跑 doctor + tiny SQS/后处理
6. 建 mlipflow-mace-gpu 或 deepmd-gpu
7. GPU salloc + srun 进入计算节点
8. nvidia-smi + torch.cuda.is_available()
9. GPU 单模型 inference
10. GPU tiny MD
11. 提交 CPU/GPU sbatch smoke
12. 配 VASP + POTCAR 并跑 tiny DFT
13. 配 DeepMD/MACE 对应 LAMMPS build 并跑 1-2 step smoke
14. 把验证过的路径、module、partition、版本写进私有 site.local.yaml
15. 做 remote fetch + SHA-256 + MLIPFlow check/collect
16. 再扩大到真实训练、DFT、长 MD 和筛选
```

---

## 36. 开发与许可

测试：

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

MLIPFlow 使用 Apache License 2.0。外部数值程序、模型、数据、POTCAR 和势函数受各自许可证约束；安装 MLIPFlow 不会自动获得它们的使用权。
