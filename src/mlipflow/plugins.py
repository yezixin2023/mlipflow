"""Built-in scientific capability lookup and adapter loading."""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import CapabilityError


BUILTIN_CAPABILITIES: dict[str, dict[str, Any]] = {
    "active-learning": {
        "adapter": "adapter.py",
        "backends": ("local", "ssh-slurm"),
        "operations": ("committee-evaluate", "select-candidates", "assess-round"),
        "approval_operations": (),
        "description": "Finite offline active-learning selection and convergence assessment.",
    },
    "ase-md": {
        "adapter": "adapter_restart.py",
        "backends": ("ssh-slurm",),
        "operations": ("run",),
        "approval_operations": ("run",),
        "description": "ASE NVT/NPT molecular dynamics with restart support.",
    },
    "candidate-ranking": {
        "adapter": "adapter.py",
        "backends": ("local",),
        "operations": ("rank-candidates",),
        "approval_operations": (),
        "description": "Deterministic single-metric candidate ranking.",
    },
    "dft-labeling": {
        "adapter": "adapter.py",
        "backends": ("local", "ssh-slurm"),
        "operations": ("vasp-prepare", "label", "dataset-assemble"),
        "approval_operations": ("label",),
        "description": "VASP preparation, labeling, relaxation/AIMD, and dataset assembly.",
    },
    "electrochemical-voltage": {
        "adapter": "adapter.py",
        "backends": ("local",),
        "operations": ("compute-from-energies", "replay-si-table-s11"),
        "approval_operations": (),
        "description": "Electrochemical voltage analysis and manuscript replay.",
    },
    "high-entropy-structure": {
        "adapter": "adapter.py",
        "backends": ("local",),
        "operations": ("generate-sqs",),
        "approval_operations": (),
        "description": "Seeded high-entropy/SQS structure generation.",
    },
    "ionic-transport": {
        "adapter": "adapter.py",
        "backends": ("local",),
        "operations": ("analyze-existing", "md-smoke-and-analyze"),
        "approval_operations": (),
        "description": "MSD, diffusion, conductivity, RDF, and Arrhenius analysis.",
    },
    "lammps-md": {
        "adapter": "adapter_restart.py",
        "backends": ("local", "ssh-slurm"),
        "operations": ("lammps-prepare", "execute"),
        "approval_operations": ("execute",),
        "description": "LAMMPS MLIP preparation/execution with restart support.",
    },
    "mlip-benchmark": {
        "adapter": "adapter.py",
        "backends": ("local", "ssh-slurm"),
        "operations": (
            "evaluate-fresh",
            "evaluate-static",
            "normalize-replay",
            "normalize-execute",
        ),
        "approval_operations": ("evaluate-fresh",),
        "description": "Fresh and replayed MLIP energy/force/stress benchmarks.",
    },
    "mlip-training": {
        "adapter": "adapter_cluster.py",
        "backends": ("local", "ssh-slurm"),
        "operations": ("train", "finetune"),
        "approval_operations": ("train", "finetune"),
        "description": "DeepMD, MACE, CHGNet, and MatGL/M3GNet training.",
    },
    "pes-sampling": {
        "adapter": "adapter_cluster.py",
        "backends": ("local", "ssh-slurm"),
        "operations": (
            "direct-select",
            "lasp-input-prepare",
            "merge-structures",
            "lasp-ssw-execute",
            "lasp-ssw-normalize-replay",
        ),
        "approval_operations": ("lasp-ssw-execute",),
        "description": "DIRECT selection and LASP/SSW preparation, execution, and replay.",
    },
}


def capability(capability_id: str) -> dict[str, Any]:
    try:
        return BUILTIN_CAPABILITIES[capability_id]
    except KeyError as exc:
        raise CapabilityError(f"unknown built-in capability {capability_id!r}") from exc


def capability_directory(capability_id: str) -> Path:
    spec = capability(capability_id)
    source_root = Path(__file__).resolve().parents[2] / "plugins"
    installed_root = Path(sysconfig.get_path("data")) / "share" / "mlipflow" / "plugins"
    for root in (source_root, installed_root):
        directory = root / capability_id
        if (directory / str(spec["adapter"])).is_file():
            return directory
    raise CapabilityError(f"built-in capability implementation is missing: {capability_id}")


def load_adapter(capability_id: str) -> Any:
    spec = capability(capability_id)
    path = capability_directory(capability_id) / str(spec["adapter"])
    module = _load_module(path, f"mlipflow_builtin_{capability_id.replace('-', '_')}")
    try:
        adapter = getattr(module, "Adapter")
    except AttributeError as exc:
        raise CapabilityError(f"built-in adapter class is missing: {path}") from exc
    return adapter()


def resolve_operation(adapter: Any, capability_id: str, context: Any) -> str:
    resolver = getattr(adapter, "operation", None)
    if not callable(resolver):
        raise CapabilityError(
            f"built-in capability {capability_id} does not expose operation resolution"
        )
    operation = resolver(context)
    if not isinstance(operation, str):
        raise CapabilityError(
            f"built-in capability {capability_id} resolved a non-string operation"
        )
    return operation


def _load_module(path: Path, name: str) -> ModuleType:
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise CapabilityError(f"cannot load built-in adapter: {path}")
    module = importlib.util.module_from_spec(module_spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise CapabilityError(f"built-in adapter import failed for {path}: {exc}") from exc
    finally:
        sys.dont_write_bytecode = previous
    return module
