# Cluster MLIP execution environments

MLIPipe uses one site-owned Python environment per MLIP framework. These are
execution environments for scheduled training, not the local MLIPipe
control/science environment.

```text
mlip-deepmd -> DeepMD-kit environment
mlip-mace   -> MACE environment
mlip-chgnet -> CHGNet environment
mlip-m3gnet -> MatGL/M3GNet environment
```

Create only the environments used by a site. Do not combine the four frameworks
into one environment. The staged cluster runner needs `PyYAML` and the selected
framework stack; it does not need MLIPipe itself or the MLIPipe `dev` extra.

Framework, Python, accelerator, driver, CUDA/ROCm, and PyTorch compatibility is a
site decision. The commands below intentionally do not prescribe one CUDA or
PyTorch version. For GPU environments, choose the install command that matches the
target compute nodes and site modules, then verify it inside an allocation on those
nodes. Useful upstream references are the [PyTorch installation selector](https://pytorch.org/get-started/locally/),
[DeepMD-kit installation guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/install.html),
[MACE installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html),
[CHGNet repository](https://github.com/CederGroupHub/chgnet#installation), and
[MatGL documentation](https://matgl.ai/).

## Choose the site runtime first

Before creating an environment, decide and record:

- the framework and MLIPipe framework id (`deepmd`, `mace`, `chgnet`, or
  `m3gnet`);
- CPU-only, NVIDIA CUDA, AMD ROCm, or another supported compute target;
- the site Python interpreter or module used to create the environment;
- the framework-supported Python and accelerator-runtime combination;
- for DeepMD, the backend selected by the training config, such as `tf` or `pt`;
- whether the environment is user-owned or centrally managed and who may update it.

Use a site-owned absolute environment root such as
`/ABS/PATH/TO/MLIP_ENVS`. A virtual environment is shown because its interpreter
has a stable `<environment>/bin/python` path. Conda/mamba environments work too;
bind their absolute `bin/python` path in exactly the same way.

If the cluster requires compiler, CUDA, or Python modules, load the same modules
when creating and validating the environment and in its framework-specific
`run.sh`. Do not assume that a successful import on a login node proves that a GPU
compute node has a working runtime.

## External scientific executables

Some MLIPipe capabilities require scientific programs that are installed and
maintained by the user or cluster site rather than by MLIPipe itself.

### LAMMPS for MLIP molecular dynamics

`lammps-md` requires a site-installed LAMMPS build that provides the interface
used by the selected MLIP backend. A generic `lmp` executable is not necessarily
sufficient.

DeepMD, MACE, and MatGL/M3GNet use different LAMMPS interfaces, and those
interfaces may require different LAMMPS versions, compiled packages, external
libraries, accelerator options, or build configurations. Follow the
[LAMMPS build documentation](https://docs.lammps.org/Build.html) together with
the installation instructions for the selected MLIP interface, and validate the
resulting executable on the target compute nodes.

A site may therefore maintain multiple validated LAMMPS builds. Bind the
appropriate executable, launcher, modules, and interface paths only in the
corresponding site-owned `lammps-<framework>-<target>/run.sh` template. Do not
assume that one LAMMPS binary supports every MLIP backend.

If you do not want to compile and maintain MLIP-specific LAMMPS builds, use
`ase-md` instead. `ase-md` runs dynamics through the corresponding ASE
calculator and supports DeepMD, MACE, MatGL/M3GNet, and CHGNet without requiring
an external LAMMPS executable.

### VASP for DFT labeling

`dft-labeling` does not provide or install VASP. VASP is licensed software and
must be obtained and compiled independently by a licensed user or site
administrator following the
[official VASP installation documentation](https://vasp.at/wiki/Installing_VASP.6.X.X).

The site is responsible for providing a validated VASP executable together with
the required compiler/runtime modules, MPI launcher, and numerical libraries.
Bind these only in the site-owned `vasp/run.sh` or `vasp-batch/run.sh` template;
do not place absolute VASP executable paths or cluster-specific launcher settings
in project files or capability parameters.

MLIPipe also does not distribute POTCAR data. `dft-labeling` assembles
runtime-only POTCAR inputs from the licensed site/user pseudopotential
installation and excludes POTCAR files from results, reports, and
repository content.

Before production DFT or AIMD runs, validate the VASP build on the target compute
nodes. Successful process or scheduler termination is not sufficient: the
`dft-labeling` scientific completion checks must still confirm the required
electronic and, where applicable, ionic convergence.

## Create the DeepMD environment

Create an environment only for the backend used by the approved DeepMD config.
The current bundled runner imports backend-specific entry points such as
`deepmd.tf.entrypoints.main` or `deepmd.pt.entrypoints.main`.

A CPU/TensorFlow-oriented starting point is:

```bash
/ABS/PATH/TO/SITE_PYTHON -m venv /ABS/PATH/TO/MLIP_ENVS/mlip-deepmd
DEEPMD_PYTHON=/ABS/PATH/TO/MLIP_ENVS/mlip-deepmd/bin/python
"$DEEPMD_PYTHON" -m pip install --upgrade pip
"$DEEPMD_PYTHON" -m pip install PyYAML 'deepmd-kit[cpu]'
```

For a PyTorch-backed DeepMD environment, first install the site-compatible
PyTorch build selected for the compute nodes, then install the upstream DeepMD
PyTorch extra:

```bash
DEEPMD_PYTHON=/ABS/PATH/TO/MLIP_ENVS/mlip-deepmd/bin/python
# Run the site-selected PyTorch installation command with "$DEEPMD_PYTHON" -m pip.
"$DEEPMD_PYTHON" -m pip install PyYAML 'deepmd-kit[torch]'
```

For TensorFlow GPU, Paddle, JAX, ROCm, source builds, or offline packages, follow
the matching upstream DeepMD instructions instead of copying a CUDA-specific
example from another cluster. The installed backend must agree with
the DeepMD config field `_mlipipe.backend`.

Verify the exact backend entry point. Replace `deepmd.pt.entrypoints.main` with
the module selected by the config when it is not PyTorch:

```bash
DEEPMD_PYTHON=/ABS/PATH/TO/MLIP_ENVS/mlip-deepmd/bin/python
"$DEEPMD_PYTHON" -m pip check
"$DEEPMD_PYTHON" -c 'import importlib, importlib.metadata as m, yaml; importlib.import_module("deepmd.pt.entrypoints.main"); print(m.version("deepmd-kit"))'
```

## Create the MACE environment

The distribution name is `mace-torch`; the unrelated PyPI package named `mace`
must not be used. For GPU execution, install a site-compatible PyTorch build first
using the upstream selector or the site's module/package policy.

```bash
/ABS/PATH/TO/SITE_PYTHON -m venv /ABS/PATH/TO/MLIP_ENVS/mlip-mace
MACE_PYTHON=/ABS/PATH/TO/MLIP_ENVS/mlip-mace/bin/python
"$MACE_PYTHON" -m pip install --upgrade pip
# Install the site-selected PyTorch build here when the default is not suitable.
"$MACE_PYTHON" -m pip install PyYAML mace-torch
"$MACE_PYTHON" -m pip check
"$MACE_PYTHON" -c 'import importlib.metadata as m, torch, yaml; from mace import tools; from mace.cli import run_train; print(m.version("mace-torch"), torch.__version__)'
```

## Create the CHGNet environment

For GPU execution, install the site-compatible PyTorch build before CHGNet.
MLIPipe's CHGNet training contract requires `precision: float32`; that scientific
constraint is separate from the site's Torch/CUDA choice.

```bash
/ABS/PATH/TO/SITE_PYTHON -m venv /ABS/PATH/TO/MLIP_ENVS/mlip-chgnet
CHGNET_PYTHON=/ABS/PATH/TO/MLIP_ENVS/mlip-chgnet/bin/python
"$CHGNET_PYTHON" -m pip install --upgrade pip
# Install the site-selected PyTorch build here when the default is not suitable.
"$CHGNET_PYTHON" -m pip install PyYAML chgnet
"$CHGNET_PYTHON" -m pip check
"$CHGNET_PYTHON" -c 'import importlib.metadata as m, torch, yaml; from chgnet.data.dataset import StructureData, collate_graphs; from chgnet.model.model import CHGNet; from chgnet.trainer import Trainer; print(m.version("chgnet"), torch.__version__)'
```

## Create the MatGL/M3GNet environment

MLIPipe names this framework `m3gnet`, while its Python distribution and import
package are `matgl`. For GPU execution, install the site-compatible PyTorch build
before MatGL.

```bash
/ABS/PATH/TO/SITE_PYTHON -m venv /ABS/PATH/TO/MLIP_ENVS/mlip-m3gnet
MATGL_PYTHON=/ABS/PATH/TO/MLIP_ENVS/mlip-m3gnet/bin/python
"$MATGL_PYTHON" -m pip install --upgrade pip
# Install the site-selected PyTorch build here when the default is not suitable.
"$MATGL_PYTHON" -m pip install PyYAML matgl
"$MATGL_PYTHON" -m pip check
```

MatGL has changed graph backends and training APIs across releases. The selected
version must pass the import path used by the MLIPipe config. With `api: auto`,
the runner uses the high-level API when both high-level classes are present and
otherwise tries the legacy API:

```bash
MATGL_PYTHON=/ABS/PATH/TO/MLIP_ENVS/mlip-m3gnet/bin/python
"$MATGL_PYTHON" - <<'PY'
import importlib.metadata as metadata
import lightning
import matgl
import torch
import yaml
from matgl.models import M3GNet

high_level = callable(getattr(matgl, "MGLDatasetLoader", None)) and callable(
    getattr(matgl, "MGLPotentialTrainer", None)
)
if not high_level:
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
    from matgl.utils.training import PotentialLightningModule

print(metadata.version("matgl"), metadata.version("lightning"), torch.__version__)
print("mlipipe_m3gnet_api=", "high_level" if high_level else "legacy")
PY
```

If neither import path works, do not bind that environment. Select and pin a
site-validated MatGL dependency set that passes the import smoke and a bounded
scheduled test. Do not silently change a project's `api` setting to accommodate an
unreviewed environment.

## Verify accelerator visibility and record the environment

MACE, CHGNet, and MatGL use PyTorch directly; a PyTorch-backed DeepMD environment
does too. For an environment intended for GPU work, run this on an allocated GPU
compute node under the same modules used by `run.sh`:

```bash
FRAMEWORK_PYTHON=/ABS/PATH/TO/ONE/FRAMEWORK_ENV/bin/python
"$FRAMEWORK_PYTHON" -c 'import torch; print("torch=", torch.__version__); print("runtime=", torch.version.cuda); print("gpu_available=", torch.cuda.is_available()); assert torch.cuda.is_available()'
```

For a CPU-only environment, `torch.cuda.is_available()` may correctly be false.
For any environment, retain a site-local, reviewable record after validation:

```bash
FRAMEWORK_PYTHON=/ABS/PATH/TO/ONE/FRAMEWORK_ENV/bin/python
"$FRAMEWORK_PYTHON" -m pip check
"$FRAMEWORK_PYTHON" -m pip freeze > /ABS/PATH/TO/SITE/ENVIRONMENT_RECORDS/mlip-framework.requirements.txt
"$FRAMEWORK_PYTHON" -c 'import platform, sys; print(sys.executable); print(platform.python_version())'
```

Use a distinct record name for each framework. Also record the site modules,
compute-node type, driver/runtime evidence, validation date, and a bounded smoke
result. A package import is necessary but is not scientific validation.

Prefer creating a new versioned environment, validating it, and then reviewing a
`PYTHON_BIN` change over mutating a working production environment in place.

## Bind each environment through `PYTHON_BIN`

Under the cluster profile's one canonical `remote_template_root`, install one
template family for each enabled framework:

| MLIPipe framework | Template | `PYTHON_BIN` target |
|---|---|---|
| `deepmd` | `mlip-deepmd/run.sh` | `/ABS/PATH/TO/MLIP_ENVS/mlip-deepmd/bin/python` |
| `mace` | `mlip-mace/run.sh` | `/ABS/PATH/TO/MLIP_ENVS/mlip-mace/bin/python` |
| `chgnet` | `mlip-chgnet/run.sh` | `/ABS/PATH/TO/MLIP_ENVS/mlip-chgnet/bin/python` |
| `m3gnet` | `mlip-m3gnet/run.sh` | `/ABS/PATH/TO/MLIP_ENVS/mlip-m3gnet/bin/python` |

Start each file from
[`examples/training_all_models/cluster/run.sh.example`](../examples/training_all_models/cluster/run.sh.example).
Set an absolute `PYTHON_BIN`, plus the site-owned `DATA_ROOT` and `MODEL_ROOT`:

```bash
PYTHON_BIN="/ABS/PATH/TO/MLIP_ENVS/mlip-mace/bin/python"
DATA_ROOT="/ABS/PATH/TO/CLUSTER/DATASETS"
MODEL_ROOT="/ABS/PATH/TO/CLUSTER/FOUNDATION_MODELS"
```

The MACE path above belongs only in `mlip-mace/run.sh`; the other three templates
must point to their own interpreters. Add site module initialization before the
Python invocation when required. Keep the template placeholders and completion
reporting from the example intact. The Slurm submit template already calls the
rendered script with `bash`, so the library copy of `run.sh` does not need to be an
executable file.

Do not run the template-library `run.sh` directly: its placeholders are rendered
and the bundled runner is staged only when MLIPipe creates a fresh attempt.

## How `site.yaml`, templates, and environments fit together

The three layers have separate ownership:

```text
local ~/.mlipipe/site.yaml
  backend_profile -> SSH config alias + remote_template_root + work_root
                                      |
                                      v
remote_template_root/                 persistent site-owned knowledge
  slurm/single-python/{cpu,gpu}.sbatch
  mlip-<framework>/run.sh             -> PYTHON_BIN + modules + DATA_ROOT/MODEL_ROOT
                                                        |
                                                        v
framework execution environment       PyYAML + exactly one framework stack

work_root/<project>/<node>/attempt-XXXX/
  rendered submit.sbatch/run.sh + staged runners + declared outputs
```

- `site.yaml` selects a named cluster profile. It does not contain framework
  package paths, module commands, credentials, datasets, or model weights.
- `remote_template_root` is the cluster's canonical template library. Do not create
  a sibling template root per framework; add `mlip-<framework>/run.sh` families
  below the existing site root.
- `work_root` is separate from the template root and holds fresh per-attempt
  workspaces. It is not an environment installation directory.
- The project node declares `backend_profile`, `framework`, scientific inputs, and
  abstract resources. The `mlip-training` adapter maps the framework to its template
  family; the project never supplies `PYTHON_BIN`.
- The Slurm template maps CPU/GPU/memory/walltime according to site policy. The
  framework `run.sh` selects the interpreter and cluster artifact roots. The
  interpreter then runs MLIPipe's staged `training_cluster.py`.

For example, a node with `backend_profile: cluster-a` and `framework: mace` resolves
the profile in local `site.yaml`, reads
`<remote_template_root>/slurm/single-python/{cpu,gpu}.sbatch` and
`<remote_template_root>/mlip-mace/run.sh`, and finally invokes the MACE environment
through that rendered script's `PYTHON_BIN`.

## Final preflight

Before approving a real training job:

1. Run the framework-specific import smoke with the exact `PYTHON_BIN`.
2. For GPU work, verify accelerator visibility inside a matching compute-node
   allocation, not only on the login node.
3. Confirm `DATA_ROOT` and `MODEL_ROOT` exist and that reference manifests resolve
   below them with the expected file/directory kind.
4. Confirm the canonical template library contains the matching
   `mlip-<framework>/run.sh` and `slurm/single-python/{cpu,gpu}.sbatch`.
5. Run `mlipipe run NODE --dry-run` locally and review the selected backend profile,
   template family, execution model, resources, staged inputs, and output allowlist.
6. Use a short scheduled environment smoke before production-scale training.

Scheduler `COMPLETED` is not enough: MLIPipe still requires the remote runner,
output schema, declared paths/parameters, and scientific checker to pass before the
attempt becomes `OK`.
