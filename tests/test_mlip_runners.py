import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "mlip-training"
sys.path.insert(0, str(PLUGIN))
import mlip_chgnet  # noqa: E402
import mlip_deepmd  # noqa: E402
import mlip_m3gnet  # noqa: E402
import mlip_mace  # noqa: E402
from mlip_common import TrainingError, write_result  # noqa: E402


def ns(framework, operation="train", precision="float64"):
    return argparse.Namespace(
        framework=framework,
        operation=operation,
        output="model.bin",
        result_manifest="result.json",
        seed=23,
        device="cpu",
        precision=precision,
        foundation_model="foundation.model",
        data="dataset.json",
        config="config.json",
    )


def test_deepmd_plan():
    cfg = {
        "_mlipflow": {"backend": "pt"},
        "model": {"descriptor": {"type": "dpa2"}},
        "training": {"seed": 23},
    }
    assert "--finetune" in mlip_deepmd.plan(ns("deepmd", "finetune"), cfg)["train_args"]


def test_deepmd_run_expands_portable_system_patterns(tmp_path, monkeypatch):
    data = tmp_path / "dataset"
    for split in ("train", "valid"):
        system = data / split / "system-000001"
        system.mkdir(parents=True)
        (system / "type.raw").write_text("0\n", encoding="utf-8")
    result = tmp_path / "attempt" / "training-result.json"
    output = result.parent / "model.pb"
    args = ns("deepmd")
    args.data = str(data)
    args.output = str(output)
    args.result_manifest = str(result)
    config = {
        "_mlipflow": {"backend": "tf", "link_data_as": "data"},
        "training": {
            "seed": 23,
            "training_data": {"systems": ["data/train/system-*"]},
            "validation_data": {"systems": ["data/valid/system-*"]},
        },
    }
    observed = {}

    class FakeEntrypoint:
        @staticmethod
        def main(argv):
            observed.update(json.loads(Path(argv[1]).read_text(encoding="utf-8")))

    def freeze(argv, *, cwd, check):
        assert argv[:4] == [sys.executable, "-m", "deepmd", "freeze"]
        assert cwd == result.parent / "deepmd-work"
        assert check is True
        Path(argv[argv.index("-o") + 1]).write_bytes(b"model")

    monkeypatch.setattr(mlip_deepmd.importlib, "import_module", lambda _: FakeEntrypoint)
    monkeypatch.setattr(mlip_deepmd.subprocess, "run", freeze)
    mlip_deepmd.run(args, config, tmp_path / "config.json", data)

    training = observed["training"]
    assert training["training_data"]["systems"] == [
        str((result.parent / "deepmd-work/data/train/system-000001").absolute())
    ]
    assert training["validation_data"]["systems"] == [
        str((result.parent / "deepmd-work/data/valid/system-000001").absolute())
    ]
    assert output.read_bytes() == b"model"


def test_chgnet_guard(tmp_path):
    with pytest.raises(Exception):
        mlip_chgnet.plan(
            ns("chgnet"), {"framework": "chgnet", "chgnet": {}}, tmp_path / "data.jsonl"
        )


def test_chgnet_columnar_historical_schema_is_normalized(tmp_path):
    data = tmp_path / "historical.json"
    data.write_text(
        json.dumps(
            {
                "structure": [{"id": 1}, {"id": 2}],
                "energy_per_atom": [-1.0, -2.0],
                "force": [[[0, 0, 0]], [[1, 1, 1]]],
                "stress": [[[0, 0, 0]], [[1, 1, 1]]],
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "dataset": {
            "layout": "columnar",
            "energy_key": "energy_per_atom",
            "forces_key": "force",
            "energy_is_per_atom": True,
        }
    }

    normalized = list(
        mlip_chgnet._normalized_records(data, mlip_chgnet._dataset_contract(cfg), "efs")
    )

    assert normalized == [
        {
            "structure": {"id": 1},
            "energy": -1.0,
            "forces": [[0, 0, 0]],
            "stress": [[0, 0, 0]],
        },
        {
            "structure": {"id": 2},
            "energy": -2.0,
            "forces": [[1, 1, 1]],
            "stress": [[1, 1, 1]],
        },
    ]


def test_chgnet_columnar_max_records_selects_deterministic_prefix(tmp_path):
    data = tmp_path / "historical.json"
    data.write_text(
        json.dumps(
            {
                "structure": [{"id": 1}, {"id": 2}, {"id": 3}],
                "energy_per_atom": [-1.0, -2.0, -3.0],
                "force": [[[0, 0, 0]], [[1, 1, 1]], [[2, 2, 2]]],
            }
        ),
        encoding="utf-8",
    )
    contract = mlip_chgnet._dataset_contract(
        {
            "dataset": {
                "layout": "columnar",
                "energy_key": "energy_per_atom",
                "forces_key": "force",
                "energy_is_per_atom": True,
                "max_records": 2,
            }
        }
    )

    normalized = list(mlip_chgnet._normalized_records(data, contract, "ef"))

    assert [item["structure"]["id"] for item in normalized] == [1, 2]


def test_chgnet_columnar_schema_rejects_different_column_lengths(tmp_path):
    data = tmp_path / "historical.json"
    data.write_text(
        json.dumps(
            {
                "structure": [{"id": 1}, {"id": 2}],
                "energy_per_atom": [-1.0],
                "force": [[[0, 0, 0]], [[1, 1, 1]]],
            }
        ),
        encoding="utf-8",
    )
    contract = mlip_chgnet._dataset_contract(
        {
            "dataset": {
                "layout": "columnar",
                "energy_key": "energy_per_atom",
                "forces_key": "force",
                "energy_is_per_atom": True,
            }
        }
    )

    with pytest.raises(TrainingError, match="different lengths"):
        list(mlip_chgnet._normalized_records(data, contract, "ef"))


def test_chgnet_split_is_seeded_and_records_indices():
    cfg = {"split": {"train_ratio": 0.8, "val_ratio": 0.05, "include_test": True}}

    first = mlip_chgnet._split_indices(100, cfg, 23)
    second = mlip_chgnet._split_indices(100, cfg, 23)

    assert first == second
    train, val, test, evidence = first
    assert (len(train), len(val), len(test)) == (80, 5, 15)
    assert len(set(train + val + test)) == 100
    assert evidence["seed"] == 23
    assert evidence["train_indices"] == train
    assert evidence["validation_indices"] == val
    assert evidence["test_indices"] == test


def test_chgnet_historical_freeze_groups_are_explicit():
    class Parameter:
        requires_grad = True

        @staticmethod
        def numel():
            return 2

    class Module:
        def __init__(self):
            self.parameter = Parameter()

        def parameters(self):
            return [self.parameter]

    model = SimpleNamespace(
        atom_embedding=Module(),
        bond_embedding=Module(),
        angle_embedding=Module(),
        bond_basis_expansion=Module(),
        angle_basis_expansion=Module(),
        atom_conv_layers=[Module(), Module()],
        bond_conv_layers=[Module(), Module()],
        angle_layers=[Module(), Module()],
    )
    requested = [
        "atom_embedding",
        "bond_embedding",
        "angle_embedding",
        "bond_basis_expansion",
        "angle_basis_expansion",
        "atom_conv_layers_except_last",
        "bond_conv_layers",
        "angle_layers",
    ]

    frozen, count = mlip_chgnet._freeze_modules(model, requested, "finetune")

    assert frozen == requested
    assert count == 20
    assert model.atom_conv_layers[-1].parameter.requires_grad is True
    assert model.atom_conv_layers[0].parameter.requires_grad is False


def test_m3gnet_plan(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    assert (
        mlip_m3gnet.plan(
            ns("m3gnet"),
            {
                "framework": "m3gnet",
                "m3gnet": {"trainer": {"max_epochs": 1}},
            },
            data,
        )["framework"]
        == "m3gnet"
    )


def test_m3gnet_columnar_historical_schema_selects_prefix(tmp_path):
    data = tmp_path / "historical.json"
    data.write_text(
        json.dumps(
            {
                "structure": [{"id": 1}, {"id": 2}, {"id": 3}],
                "uncorrected_total_energy": [-10.0, -20.0, -30.0],
                "force": [[[0, 0, 0]], [[1, 1, 1]], [[2, 2, 2]]],
                "stress": [[[0, 0, 0]], [[1, 1, 1]], [[2, 2, 2]]],
            }
        ),
        encoding="utf-8",
    )
    contract = mlip_m3gnet._dataset_contract(
        {
            "dataset": {
                "layout": "columnar",
                "energy_key": "uncorrected_total_energy",
                "forces_key": "force",
                "stress_key": "stress",
                "energy_is_per_atom": False,
                "max_records": 3,
            }
        }
    )

    normalized = list(mlip_m3gnet._normalized_records(data, contract, True))

    assert normalized[0] == {
        "structure": {"id": 1},
        "energy": -10.0,
        "forces": [[0, 0, 0]],
        "stress": [[0, 0, 0]],
    }
    assert [item["structure"]["id"] for item in normalized] == [1, 2, 3]


def test_m3gnet_per_atom_energy_is_rejected():
    with pytest.raises(TrainingError, match="requires total energies"):
        mlip_m3gnet._dataset_contract({"dataset": {"energy_is_per_atom": True}})


def test_m3gnet_api_selection_preserves_high_level_and_legacy_paths():
    modern = SimpleNamespace(
        MGLDatasetLoader=type("MGLDatasetLoader", (), {}),
        MGLPotentialTrainer=type("MGLPotentialTrainer", (), {}),
    )
    historical = SimpleNamespace()

    assert mlip_m3gnet._select_api({"api": "auto"}, modern) == "high_level"
    assert mlip_m3gnet._select_api({"api": "auto"}, historical) == "legacy"
    assert mlip_m3gnet._select_api({"api": "legacy"}, modern) == "legacy"
    with pytest.raises(TrainingError, match="requires MGLDatasetLoader"):
        mlip_m3gnet._select_api({"api": "high_level"}, historical)


def test_m3gnet_split_is_seeded_and_records_indices():
    cfg = {"split": {"train_ratio": 0.8, "val_ratio": 0.1, "include_test": True}}

    first = mlip_m3gnet._split_indices(128, cfg, 23)
    second = mlip_m3gnet._split_indices(128, cfg, 23)

    assert first == second
    train, val, test, evidence = first
    assert (len(train), len(val), len(test)) == (102, 12, 14)
    assert len(set(train + val + test)) == 128
    assert evidence["seed"] == 23
    assert evidence["train_indices"] == train
    assert evidence["validation_indices"] == val
    assert evidence["test_indices"] == test


def test_m3gnet_history_requires_every_requested_finite_epoch(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "epoch,step,train_Energy_MAE,val_Energy_MAE\n0,1,,0.2\n0,1,0.1,\n",
        encoding="utf-8",
    )

    train, val = mlip_m3gnet._history(metrics, 1)

    assert train == [{"train_Energy_MAE": 0.1}]
    assert val == [{"val_Energy_MAE": 0.2}]
    with pytest.raises(TrainingError, match="requested epochs"):
        mlip_m3gnet._history(metrics, 2)


def test_mace_plan(tmp_path):
    data = tmp_path / "train.xyz"
    data.write_text("0\ncomment\n")
    cfg = {"framework": "mace", "mace": {"name": "demo", "options": {"max_num_epochs": 1}}}
    assert mlip_mace.plan(ns("mace"), cfg, tmp_path / "config.json", data)["framework"] == "mace"


def test_mace_plan_matches_cpu_site_store_true_parser(tmp_path):
    data = tmp_path / "train.xyz"
    data.write_text("0\ncomment\n")
    cfg = {
        "framework": "mace",
        "mace": {
            "name": "demo",
            "options": {"max_num_epochs": 1, "ema": True, "amsgrad": True},
        },
    }

    argv = mlip_mace.plan(ns("mace"), cfg, tmp_path / "config.json", data)["argv"]

    assert "--work_dir" not in argv
    assert argv.count("--ema") == 1
    assert argv.count("--amsgrad") == 1
    assert "True" not in argv


def test_mace_execute_falls_back_to_installed_main_and_restores_argv():
    parsed = object()
    calls = []

    class Parser:
        def parse_args(self, argv):
            calls.append(("parse", list(argv)))
            return parsed

    tools = SimpleNamespace(build_default_arg_parser=lambda: Parser())

    def main():
        calls.append(("main", list(sys.argv)))

    original = sys.argv
    mlip_mace._execute_mace(tools, SimpleNamespace(main=main), ["--name", "demo"])

    assert calls == [
        ("parse", ["--name", "demo"]),
        ("main", ["mace_run_train", "--name", "demo"]),
    ]
    assert sys.argv is original


def test_mace_prepares_all_explicit_output_directories(tmp_path):
    work = tmp_path / "mace-work"

    mlip_mace._prepare_output_directories(work)

    assert {path.name for path in work.iterdir()} == {
        "models",
        "logs",
        "checkpoints",
        "results",
    }
    assert all(path.is_dir() for path in work.iterdir())


def test_mace_runtime_environment_accepts_cpu_only_torch(monkeypatch):
    class CPUOnlyCUDA:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            raise AssertionError("CPU runtime must not query a CUDA device")

    torch = SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda=None),
        cuda=CPUOnlyCUDA(),
    )
    monkeypatch.setattr(mlip_mace.shutil, "which", lambda _name: None)
    monkeypatch.setattr(mlip_mace, "version", lambda *_names: "test-mace")

    environment = mlip_mace._runtime_environment(torch, "test-source")

    assert environment["gpu_model"] == "unavailable"
    assert environment["torch_cuda_version"] == "None"


def test_mace_select_model_prefers_native_over_compiled(tmp_path):
    native = tmp_path / "demo.model"
    compiled = tmp_path / "demo_compiled.model"
    native.write_bytes(b"native")
    compiled.write_bytes(b"compiled")

    assert mlip_mace._select_model(tmp_path, "demo") == native


def test_mace_plan_preserves_historical_gpu_options_and_pins_test_file(tmp_path):
    data = tmp_path / "train.xyz"
    test_data = tmp_path / "test.xyz"
    data.write_text("0\ntrain\n")
    test_data.write_text("0\ntest\n")
    cfg = {
        "framework": "mace",
        "mace": {
            "name": "MACE",
            "save_cpu": False,
            "auxiliary_data": {
                "test_file": {
                    "relative_path": "test.xyz",
                }
            },
            "options": {
                "E0s": "foundation",
                "amsgrad": True,
                "batch_size": 48,
                "ema": True,
                "ema_decay": 0.99,
                "energy_weight": 1.0,
                "forces_weight": 1.0,
                "lr": 0.01,
                "max_num_epochs": 1,
                "multiheads_finetuning": True,
                "r_max": 5.0,
                "scaling": "rms_forces_scaling",
                "valid_batch_size": 10,
                "valid_fraction": 0.05,
                "weight_decay": 5e-7,
            },
        },
    }

    argv = mlip_mace.plan(ns("mace", "finetune"), cfg, tmp_path / "config.json", data)["argv"]

    assert argv[argv.index("--test_file") + 1] == str(test_data)
    assert argv[argv.index("--multiheads_finetuning") + 1] == "True"
    assert argv[argv.index("--max_num_epochs") + 1] == "1"
    assert argv[argv.index("--batch_size") + 1] == "48"
    assert argv[argv.index("--lr") + 1] == "0.01"
    assert "--save_cpu" not in argv


def test_mace_fresh_plan_uses_dataset_energy_key_without_finetune_arguments(tmp_path):
    data = tmp_path / "train.xyz"
    test_data = tmp_path / "test.xyz"
    data.write_text("0\nenergy=-1.0\n")
    test_data.write_text("0\nenergy=-1.0\n")
    cfg = {
        "framework": "mace",
        "mace": {
            "name": "MACEFresh",
            "auxiliary_data": {
                "test_file": {
                    "relative_path": "test.xyz",
                }
            },
            "options": {
                "E0s": "average",
                "batch_size": 8,
                "energy_key": "energy",
                "max_num_epochs": 1,
                "r_max": 5.0,
            },
        },
    }

    argv = mlip_mace.plan(ns("mace", "train"), cfg, tmp_path / "config.json", data)["argv"]

    assert argv[argv.index("--E0s") + 1] == "average"
    assert argv[argv.index("--energy_key") + 1] == "energy"
    assert "--foundation_model" not in argv
    assert "--multiheads_finetuning" not in argv
    assert "--lora" not in argv
    assert argv[argv.index("--max_num_epochs") + 1] == "1"
    assert argv[argv.index("--batch_size") + 1] == "8"
    assert "--save_cpu" in argv


def test_mace_completion_accepts_one_finite_epoch(tmp_path):
    logs = tmp_path / "logs"
    results = tmp_path / "results"
    logs.mkdir()
    results.mkdir()
    (logs / "MACE_run-3.log").write_text(
        "Epoch 0: loss=0.25\nTraining complete\n", encoding="utf-8"
    )
    (results / "MACE_run-3_train.txt").write_text(
        json.dumps(
            {
                "mode": "eval",
                "epoch": 0,
                "loss": 0.25,
                "rmse_e_per_atom": 1.5,
                "rmse_f": 0.04,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics, evidence = mlip_mace._completion_evidence(tmp_path, 1)

    assert metrics["requested_epochs"] == 1.0
    assert metrics["completed_epochs"] == 1.0
    assert metrics["final_rmse_f"] == 0.04
    assert evidence["normal_completion"] is True
    assert evidence["all_recorded_metrics_finite"] is True


def test_mace_completion_rejects_early_exit(tmp_path):
    logs = tmp_path / "logs"
    results = tmp_path / "results"
    logs.mkdir()
    results.mkdir()
    (logs / "MACE.log").write_text("Training complete\n", encoding="utf-8")
    (results / "MACE_train.txt").write_text(
        '{"mode":"eval","epoch":0,"loss":0.2}\n', encoding="utf-8"
    )

    with pytest.raises(TrainingError, match="completed epochs"):
        mlip_mace._completion_evidence(tmp_path, 1)


def test_mace_completion_rejects_non_finite_history(tmp_path):
    logs = tmp_path / "logs"
    results = tmp_path / "results"
    logs.mkdir()
    results.mkdir()
    (logs / "MACE.log").write_text("Epoch 0: loss=nan\nTraining complete\n", encoding="utf-8")
    (results / "MACE_train.txt").write_text(
        '{"mode":"eval","epoch":0,"loss":NaN}\n', encoding="utf-8"
    )

    with pytest.raises(TrainingError, match="non-finite"):
        mlip_mace._completion_evidence(tmp_path, 1)


@pytest.mark.parametrize("metric", [True, "1.0", None, float("nan"), float("inf"), float("-inf")])
def test_write_result_rejects_invalid_metrics_without_manifest(tmp_path, metric):
    args = ns("chgnet", precision="float32")
    args.output = str(tmp_path / "model.pth.tar")
    args.result_manifest = str(tmp_path / "result.json")
    Path(args.output).write_bytes(b"model")

    with pytest.raises(TrainingError, match="metric 'loss'"):
        write_result(args, "OK", "test", {"loss": metric}, "application/octet-stream", {})

    assert not Path(args.result_manifest).exists()


def test_write_result_preserves_finite_numeric_metrics(tmp_path):
    args = ns("chgnet", precision="float32")
    args.output = str(tmp_path / "model.pth.tar")
    args.result_manifest = str(tmp_path / "result.json")
    Path(args.output).write_bytes(b"model")

    write_result(
        args,
        "OK",
        "test",
        {"Validation Loss": 0.125, "epochs": 2},
        "application/octet-stream",
        {},
    )

    result = json.loads(Path(args.result_manifest).read_text(encoding="utf-8"))
    assert result["status"] == "OK"
    assert result["metrics"] == {"epochs": 2.0, "validation_loss": 0.125}


def test_chgnet_training_completion_accepts_requested_finite_epochs():
    trainer = SimpleNamespace(
        epochs=2,
        starting_epoch=0,
        targets="ef",
        training_history={
            "e": {"train": [0.4, 0.2], "val": [0.5, 0.3], "test": []},
            "f": {"train": [0.8, 0.6], "val": [0.9, 0.7], "test": []},
        },
    )

    mlip_chgnet._validate_training_completion(trainer)


def test_chgnet_training_completion_rejects_early_exit():
    trainer = SimpleNamespace(
        epochs=2,
        starting_epoch=0,
        targets="e",
        training_history={"e": {"train": [0.4], "val": [0.5], "test": []}},
    )

    with pytest.raises(TrainingError, match="completed 1 of 2 requested epochs"):
        mlip_chgnet._validate_training_completion(trainer)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_chgnet_training_completion_rejects_non_finite_history(value):
    trainer = SimpleNamespace(
        epochs=1,
        starting_epoch=0,
        targets="e",
        training_history={"e": {"train": [0.4], "val": [value], "test": []}},
    )

    with pytest.raises(TrainingError, match="non-finite"):
        mlip_chgnet._validate_training_completion(trainer)
