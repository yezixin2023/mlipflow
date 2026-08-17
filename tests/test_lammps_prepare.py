from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

import pytest
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "lammps-md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _model(framework: str) -> dict:
    formats = {
        "deepmd": "deepmd-lammps-model",
        "mace": "mace-lammps-torchscript",
        "m3gnet": "matgl-lammps-torchscript",
        "chgnet": "chgnet-model",
    }
    return {
        "schema_version": 1,
        "model_id": f"{framework}-v1",
        "framework": framework,
        "relative_path": f"{framework}/model-v1.pt",
        "kind": "file",
        "fingerprint": "sha256:" + "2" * 64,
        "artifact_format": formats[framework],
        "elements": ["Li", "P", "S"],
    }


def _config(ensemble: str = "nvt") -> dict:
    value = {
        "schema_version": 1,
        "engine": "lammps",
        "ensemble": ensemble,
        "targets": ["cpu", "gpu"],
        "type_map": ["Li", "P", "S"],
        "temperature_k": 900.0,
        "timestep_fs": 1.0,
        "steps": 1_000_000,
        "thermo_interval": 100,
        "dump_interval": 100,
        "seed": 20260814,
        "thermostat_damping_fs": 100.0,
    }
    if ensemble == "npt-isotropic":
        value.update({"pressure_gpa": 0.5, "barostat_damping_fs": 1000.0})
    return value


def _context(tmp_path: Path, framework: str, ensemble: str = "nvt") -> dict:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    structure = inputs / "start.extxyz"
    structure.write_text(
        '3\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Li 0 0 0\nP 1 1 1\nS 2 2 2\n",
        encoding="utf-8",
    )
    _write_json(inputs / "model.json", _model(framework))
    _write_json(inputs / "lammps.json", _config(ensemble))
    return {
        "project_root": str(tmp_path),
        "attempt_dir": str(tmp_path / ".mlipflow" / "runs" / "lammps" / "attempt-1"),
        "backend": "local",
        "inputs": {
            "structure": "inputs/start.extxyz",
            "model_reference": "inputs/model.json",
            "lammps_config": "inputs/lammps.json",
        },
        "parameters": {"operation": "lammps-prepare"},
        "resources": {},
    }


@pytest.mark.parametrize("framework", ["deepmd", "mace", "m3gnet"])
def test_prepare_plan_is_ready_for_supported_frameworks(tmp_path: Path, framework: str) -> None:
    adapter = _load(f"lammps_adapter_{framework}", PLUGIN / "adapter.py")
    plan = adapter.Adapter().plan(_context(tmp_path, framework))
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["approval_summary"]["framework"] == framework
    assert plan["approval_summary"]["targets"] == ["cpu", "gpu"]
    assert plan["approval_summary"]["executes_lammps"] is False
    assert str(PLUGIN / "lammps_prepare.py") in plan["argv"]


def test_chgnet_is_explicitly_blocked(tmp_path: Path) -> None:
    adapter = _load("lammps_adapter_chgnet", PLUGIN / "adapter.py")
    plan = adapter.Adapter().plan(_context(tmp_path, "chgnet"))
    assert plan["status"] == "BLOCKED"
    assert any("CHGNet" in item["message"] for item in plan["diagnostics"])


def test_wrong_export_format_is_blocked(tmp_path: Path) -> None:
    adapter = _load("lammps_adapter_format", PLUGIN / "adapter.py")
    context = _context(tmp_path, "mace")
    model_path = tmp_path / "inputs" / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["artifact_format"] = "raw-training-checkpoint"
    _write_json(model_path, model)
    plan = adapter.Adapter().plan(context)
    assert plan["status"] == "BLOCKED"
    assert any("artifact_format" in item["message"] for item in plan["diagnostics"])


def test_deepmd_cpu_gpu_pair_style_stays_deepmd() -> None:
    module = _load("lammps_prepare_deepmd", PLUGIN / "lammps_prepare.py")
    config = _config("nvt")
    cpu = module._deck("deepmd", "cpu", config)
    gpu = module._deck("deepmd", "gpu", config)
    assert "pair_style      deepmd ${MODEL_FILE}" in cpu
    assert "pair_style      deepmd ${MODEL_FILE}" in gpu
    assert "-sf kk" not in cpu and "-sf kk" not in gpu


def test_mace_cpu_gpu_pair_styles() -> None:
    module = _load("lammps_prepare_mace", PLUGIN / "lammps_prepare.py")
    config = _config("nvt")
    cpu = module._deck("mace", "cpu", config)
    gpu = module._deck("mace", "gpu", config)
    assert "pair_style      mace\n" in cpu
    assert "pair_style      mace no_domain_decomposition" in gpu
    launcher = module._launcher("mace", "gpu", "in.gpu.lammps")
    assert launcher["argv_after_executable"][:6] == ["-k", "on", "g", "1", "-sf", "kk"]


def test_matgl_cpu_gpu_pair_styles() -> None:
    module = _load("lammps_prepare_matgl", PLUGIN / "lammps_prepare.py")
    config = _config("nvt")
    cpu = module._deck("m3gnet", "cpu", config)
    gpu = module._deck("m3gnet", "gpu", config)
    assert "pair_style      matgl\n" in cpu
    assert "pair_style      matgl/kk" in gpu
    assert "pair_coeff      * * ${MODEL_FILE} Li P S" in gpu


def test_gnnp_directory_interface_is_explicit_and_cpu_only(tmp_path: Path) -> None:
    adapter = _load("lammps_adapter_gnnp", PLUGIN / "adapter.py")
    context = _context(tmp_path, "m3gnet")
    model_path = tmp_path / "inputs" / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model.update(
        {
            "relative_path": "m3gnet/finetuned_model",
            "kind": "directory",
            "artifact_format": "matgl-model-directory",
            "lammps_interface": "gnnp",
        }
    )
    _write_json(model_path, model)
    config_path = tmp_path / "inputs" / "lammps.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["targets"] = ["cpu"]
    _write_json(config_path, config)

    plan = adapter.Adapter().plan(context)

    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["approval_summary"]["model_kind"] == "directory"
    assert plan["approval_summary"]["lammps_interface"] == "gnnp"
    module = _load("lammps_prepare_gnnp", PLUGIN / "lammps_prepare.py")
    deck = module._deck("m3gnet", "cpu", _config("nvt"), "gnnp")
    assert "pair_style      gnnp ${INTERFACE_PATH}" in deck
    assert "pair_coeff      * * matgl ${MODEL_FILE} Li P S" in deck
    launcher = module._launcher("m3gnet", "cpu", "in.cpu.lammps", "gnnp")
    assert launcher["argv_after_executable"].count("<site-resolved-interface-path>") == 1


def test_gnnp_interface_rejects_torchscript_file_and_gpu(tmp_path: Path) -> None:
    adapter = _load("lammps_adapter_gnnp_invalid", PLUGIN / "adapter.py")
    context = _context(tmp_path, "m3gnet")
    model_path = tmp_path / "inputs" / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["lammps_interface"] = "gnnp"
    _write_json(model_path, model)
    blocked_kind = adapter.Adapter().plan(context)
    assert blocked_kind["status"] == "BLOCKED"
    assert any("kind must be directory" in item["message"] for item in blocked_kind["diagnostics"])

    model.update(
        {
            "kind": "directory",
            "artifact_format": "matgl-model-directory",
        }
    )
    _write_json(model_path, model)
    blocked_gpu = adapter.Adapter().plan(context)
    assert blocked_gpu["status"] == "BLOCKED"
    assert any("does not support target(s): gpu" in item["message"] for item in blocked_gpu["diagnostics"])


def test_nvt_converts_femtoseconds_to_metal_picoseconds() -> None:
    module = _load("lammps_prepare_nvt_units", PLUGIN / "lammps_prepare.py")
    deck = module._deck("mace", "cpu", _config("nvt"))
    assert "timestep        0.001" in deck
    assert "fix             mlipflow all nvt temp 900 900 0.1" in deck


def test_npt_converts_gpa_to_metal_bar_and_damping_to_ps() -> None:
    module = _load("lammps_prepare_npt_units", PLUGIN / "lammps_prepare.py")
    deck = module._deck("m3gnet", "gpu", _config("npt-isotropic"))
    assert "fix             mlipflow all npt temp 900 900 0.1 iso 5000 5000 1" in deck


def test_model_path_is_runtime_variable_not_registry_path() -> None:
    module = _load("lammps_prepare_portable", PLUGIN / "lammps_prepare.py")
    for framework in ("deepmd", "mace", "m3gnet"):
        deck = module._deck(framework, "cpu", _config("nvt"))
        assert "${MODEL_FILE}" in deck
        assert "/cluster/" not in deck


def test_prepare_plan_binds_explicit_structure_format(tmp_path: Path) -> None:
    adapter = _load("lammps_adapter_explicit_format", PLUGIN / "adapter.py")
    context = _context(tmp_path, "deepmd")
    context["parameters"]["structure_format"] = "lammps-data"
    plan = adapter.Adapter().plan(context)
    assert plan["status"] == "READY", plan.get("diagnostics")
    assert plan["approval_summary"]["structure_format"] == "lammps-data"
    assert plan["argv"][-2:] == ["--structure-format", "lammps-data"]


def test_prepare_explicit_lammps_data_format_without_filename_inference(tmp_path: Path) -> None:
    module = _load("lammps_prepare_explicit_lammps_data", PLUGIN / "lammps_prepare.py")
    structure = tmp_path / "structure.data"
    structure.write_text(
        "(written by test)\n\n"
        "1 atoms\n1 atom types\n\n"
        "0.0 5.0 xlo xhi\n0.0 5.0 ylo yhi\n0.0 5.0 zlo zhi\n\n"
        "Masses\n\n1 6.94 # Li\n\n"
        "Atoms # atomic\n\n1 1 1.0 1.0 1.0\n",
        encoding="utf-8",
    )
    model = tmp_path / "model.json"
    model_value = _model("deepmd")
    model_value["elements"] = ["Li"]
    _write_json(model, model_value)
    config = tmp_path / "config.json"
    config_value = _config("nvt")
    config_value.update({"targets": ["cpu"], "type_map": ["Li"], "steps": 1})
    config_value.update({"thermo_interval": 1, "dump_interval": 1})
    _write_json(config, config_value)
    output = tmp_path / "prepared"

    manifest = module.prepare(
        structure,
        model,
        config,
        output,
        structure_format="lammps-data",
    )

    assert manifest["source_structure_format"] == "lammps-data"
    structure_data = output / "structure.data"
    assert structure_data.is_file()
    generated = read(structure_data, format="lammps-data", style="atomic")
    assert len(generated) == 1
    assert generated.get_chemical_symbols() == ["Li"]


def test_lammps_prepare_missing_ase_fails_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("lammps_prepare_missing_ase", PLUGIN / "lammps_prepare.py")
    context = _context(tmp_path, "deepmd")
    original_import = builtins.__import__

    def import_without_ase(name, *args, **kwargs):
        if name == "ase" or name.startswith("ase."):
            raise ModuleNotFoundError("simulated missing ASE")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ase)
    output = tmp_path / "prepared"
    with pytest.raises(module.ContractError, match="ASE is required by lammps-prepare"):
        module.prepare(
            tmp_path / context["inputs"]["structure"],
            tmp_path / context["inputs"]["model_reference"],
            tmp_path / context["inputs"]["lammps_config"],
            output,
        )
    assert not output.exists()
