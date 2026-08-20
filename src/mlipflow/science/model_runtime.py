"""Shared read-only MLIP family and inference runtime utilities.

Training records the same framework version while benchmark uses the lazy
inference loader. Importing this module never imports a heavy
scientific framework and never loads a model.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any


MODEL_FAMILIES = (
    "deepmd-se_atten_v2",
    "deepmd-se_e2_a",
    "deepmd-se_e2_r",
    "deepmd-dpa2",
    "m3gnet",
    "chgnet",
)

MODEL_FAMILY_FRAMEWORKS = {
    "deepmd-se_e2_a": "deepmd",
    "deepmd-se_e2_r": "deepmd",
    "deepmd-se_atten_v2": "deepmd",
    "deepmd-dpa2": "deepmd",
    "m3gnet": "m3gnet",
    "chgnet": "chgnet",
}

MODEL_FAMILY_ALIASES = {
    alias: family
    for family, aliases in {
        "deepmd-se_atten_v2": (
            "deepmd-se_atten_v2",
            "deepmd-se-atten-v2",
            "deepmd_se_atten_v2",
            "deepmd se_atten_v2",
            "se_atten_v2",
            "se-atten-v2",
        ),
        "deepmd-se_e2_a": (
            "deepmd-se_e2_a",
            "deepmd-se-e2-a",
            "deepmd_se_e2_a",
            "deepmd se_e2_a",
            "se_e2_a",
            "se-e2-a",
        ),
        "deepmd-se_e2_r": (
            "deepmd-se_e2_r",
            "deepmd-se-e2-r",
            "deepmd_se_e2_r",
            "deepmd se_e2_r",
            "se_e2_r",
            "se-e2-r",
        ),
        "deepmd-dpa2": ("deepmd-dpa2", "deepmd-dpa-2", "deepmd_dpa2", "dpa2", "dpa-2"),
        "m3gnet": ("m3gnet", "m3g-net"),
        "chgnet": ("chgnet", "chg-net"),
    }.items()
    for alias in aliases
}


class RuntimeCompatibilityError(RuntimeError):
    """The selected framework cannot load or evaluate the model artifact."""


def canonical_model_family(value: Any) -> str:
    """Resolve a historical spelling to one exact supported model family."""

    if not isinstance(value, str) or not value.strip():
        raise RuntimeCompatibilityError("model family must be a non-empty string")
    token = " ".join(value.strip().lower().split())
    try:
        return MODEL_FAMILY_ALIASES[token]
    except KeyError as exc:
        supported = ", ".join(MODEL_FAMILIES)
        raise RuntimeCompatibilityError(
            f"unsupported or under-specified model {value!r}; supported exact names: {supported}"
        ) from exc


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def framework_version(framework: str) -> str:
    distributions = {
        "deepmd": ("deepmd-kit",),
        "m3gnet": ("matgl",),
        "chgnet": ("chgnet",),
        "mace": ("mace-torch", "mace"),
    }
    if framework not in distributions:
        raise RuntimeCompatibilityError(f"unsupported MLIP framework: {framework}")
    return _distribution_version(*distributions[framework])


class DeepMDPredictor:
    """One shared DeepPot implementation for all four exact DeepMD families."""

    prediction_units = {
        "energy": "eV",
        "force": "eV/angstrom",
        "stress": "eV/angstrom^3",
    }

    def __init__(self, model: Path, family: str, device: str):
        del device
        try:
            from deepmd.infer import DeepPot
        except ImportError as exc:
            raise RuntimeCompatibilityError("DeepMD runtime is not installed") from exc
        try:
            self.model = DeepPot(str(model))
            self.type_map = list(self.model.get_type_map())
        except Exception as exc:
            raise RuntimeCompatibilityError(f"failed to load DeepMD model: {exc}") from exc
        self.runtime_details = {
            "framework": "deepmd",
            "framework_version": framework_version("deepmd"),
            "backend": "deepmd.infer.DeepPot",
            "device": "runtime-selected",
            "exact_model_family": family,
            "energy_convention": "total",
            "stress_convention": "ase-voigt-xx-yy-zz-yz-xz-xy",
        }

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]:
        try:
            import numpy as np

            types = [self.type_map.index(symbol) for symbol in sample["species"]]
            result = self.model.eval(
                np.asarray([sample["positions"]], dtype=float),
                np.asarray([sample["cell"]], dtype=float),
                np.asarray(types, dtype=int),
            )
            energy, forces = result[0], result[1]
            prediction: dict[str, Any] = {
                "energy": float(np.asarray(energy).reshape(-1)[0]),
                "force": np.asarray(forces).reshape(sample["natoms"], 3).tolist(),
            }
            if len(result) > 2 and result[2] is not None:
                virial = np.asarray(result[2]).reshape(3, 3)
                volume = abs(float(np.linalg.det(np.asarray(sample["cell"]))))
                if volume <= 0:
                    raise RuntimeCompatibilityError(
                        "DeepMD stress requires a positive cell volume"
                    )
                stress = -virial / volume
                prediction["stress"] = [
                    float(stress[0, 0]),
                    float(stress[1, 1]),
                    float(stress[2, 2]),
                    float(stress[1, 2]),
                    float(stress[0, 2]),
                    float(stress[0, 1]),
                ]
            return prediction
        except RuntimeCompatibilityError:
            raise
        except Exception as exc:
            raise RuntimeCompatibilityError(
                f"DeepMD inference failed for {sample['id']}: {exc}"
            ) from exc


class ASECalculatorPredictor:
    prediction_units = {
        "energy": "eV",
        "force": "eV/angstrom",
        "stress": "eV/angstrom^3",
    }

    def __init__(self, calculator: Any, runtime_details: dict[str, Any]):
        self.calculator = calculator
        self.runtime_details = runtime_details

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]:
        try:
            from ase import Atoms

            atoms = Atoms(
                symbols=sample["species"],
                positions=sample["positions"],
                cell=sample["cell"],
                pbc=sample["pbc"],
                calculator=self.calculator,
            )
            result = {
                "energy": float(atoms.get_potential_energy()),
                "force": atoms.get_forces().tolist(),
            }
            try:
                result["stress"] = atoms.get_stress(voigt=True).tolist()
            except Exception:
                pass
            return result
        except Exception as exc:
            raise RuntimeCompatibilityError(
                f"model inference failed for {sample['id']}: {exc}"
            ) from exc


def load_inference_predictor(model: Path, family: str, device: str):
    """Load an exact family using its training-compatible framework format."""

    framework = MODEL_FAMILY_FRAMEWORKS.get(family)
    if framework is None:
        raise RuntimeCompatibilityError(f"unsupported exact model family: {family}")
    if framework == "deepmd":
        return DeepMDPredictor(model, family, device)
    if framework == "m3gnet":
        try:
            import matgl
            from matgl.ext.ase import PESCalculator

            potential = matgl.load_model(str(model))
            calculator = PESCalculator(potential=potential)
        except ImportError as exc:
            raise RuntimeCompatibilityError("MatGL/M3GNet runtime is not installed") from exc
        except Exception as exc:
            raise RuntimeCompatibilityError(f"failed to load M3GNet model: {exc}") from exc
        return ASECalculatorPredictor(
            calculator,
            {
                "framework": framework,
                "framework_version": framework_version(framework),
                "backend": "matgl.ext.ase.PESCalculator",
                "device": device,
                "exact_model_family": family,
                "energy_convention": "total",
                "stress_convention": "ase-voigt-xx-yy-zz-yz-xz-xy",
            },
        )
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        loaded = CHGNet.from_file(str(model))
        calculator = CHGNetCalculator(model=loaded, use_device=device)
    except ImportError as exc:
        raise RuntimeCompatibilityError("CHGNet runtime is not installed") from exc
    except Exception as exc:
        raise RuntimeCompatibilityError(f"failed to load CHGNet model: {exc}") from exc
    return ASECalculatorPredictor(
        calculator,
        {
            "framework": framework,
            "framework_version": framework_version(framework),
            "backend": "chgnet.model.dynamics.CHGNetCalculator",
            "device": device,
            "exact_model_family": family,
            "energy_convention": "total",
            "stress_convention": "ase-voigt-xx-yy-zz-yz-xz-xy",
        },
    )
