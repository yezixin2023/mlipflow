"""The bundled runner must collect actual training evidence, not synthesize it."""

import json

import pytest

from mlipflow.plugins.mlip_training.deepmd_curve import export_training_evidence


def test_export_preserves_curve_checkpoint_and_log_bytes(tmp_path):
    work = tmp_path / "work"
    output = tmp_path / "output"
    work.mkdir()
    output.mkdir()
    sources = {
        "lcurve.out": b"# step rmse lr\n0 0.1 1e-3\n",
        "checkpoint": b'model_checkpoint_path: "model.ckpt-500"\n',
        "model.ckpt-500.index": b"fixture checkpoint index",
    }
    for name, value in sources.items():
        (work / name).write_bytes(value)
    log = b"fixture training stderr\nfinished training\n"
    (output / "training.stderr.log").write_bytes(log)
    result = {"framework_version": "test-version", "provenance": {
        "backend": "tf", "work_dir": str(work)
    }}
    export_training_evidence(output, result, "fixture-dataset")
    assert (output / "lcurve.out").read_bytes() == sources["lcurve.out"]
    assert (output / "checkpoint").read_bytes() == sources["checkpoint"]
    assert (output / "model.ckpt.index").read_bytes() == sources["model.ckpt-500.index"]
    assert (output / "train.stderr").read_bytes() == log
    report = json.loads((output / "training-report.json").read_text())
    assert report["dataset_id"] == "fixture-dataset"
    assert report["framework_version"] == "test-version"
    assert report["checkpoint_files"] == [{"name": "model.ckpt-500.index"}]


def test_missing_evidence_is_not_replaced_with_success_report(tmp_path):
    (tmp_path / "checkpoint").write_text('model_checkpoint_path: "model.ckpt-500"\n')
    result = {"framework_version": "test-version", "provenance": {
        "backend": "tf", "work_dir": str(tmp_path)
    }}
    with pytest.raises(FileNotFoundError):
        export_training_evidence(tmp_path, result, "fixture-dataset")
    assert not (tmp_path / "training-report.json").exists()
