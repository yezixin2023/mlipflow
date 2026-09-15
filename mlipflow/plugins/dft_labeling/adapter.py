"""dft labeling: adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import contracts, dataset, label, prepare


class Adapter:
    """Plan one reviewed operation and verify its standardized result."""

    def operation(self, context: Any) -> str:
        return contracts._operation(context)

    def validate(self, context: Any) -> list[dict[str, str]]:
        if not isinstance(context, Mapping):
            return contracts._base_diagnostics(context)
        if contracts._operation(context) == contracts.PREPARE_OPERATION:
            return prepare._validate_prepare(context)
        if contracts._operation(context) == contracts.LABEL_OPERATION:
            return label._validate_label(context)
        if contracts._operation(context) == contracts.DATASET_OPERATION:
            return dataset._validate_dataset_assemble(context)
        return contracts._base_diagnostics(context)

    def plan(self, context: Any) -> dict[str, Any]:
        if not isinstance(context, Mapping):
            return {
                "plugin_id": contracts.PLUGIN_ID,
                "status": "BLOCKED",
                "executable": False,
                "diagnostics": self.validate(context),
            }
        if contracts._operation(context) == contracts.PREPARE_OPERATION:
            return prepare._plan_prepare(context)
        if contracts._operation(context) == contracts.LABEL_OPERATION:
            return label._plan_label(context)
        if contracts._operation(context) == contracts.DATASET_OPERATION:
            return dataset._plan_dataset_assemble(context)
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "BLOCKED",
            "executable": False,
            "diagnostics": self.validate(context),
        }

    def _result_path(self, context: Mapping[str, Any]) -> Path:
        parameters = contracts._mapping(context["parameters"])
        default = {
            contracts.PREPARE_OPERATION: "dft-input-manifest.json",
            contracts.LABEL_OPERATION: "dft-labeling-result.json",
            contracts.DATASET_OPERATION: "dataset-assembly-result.json",
        }.get(contracts._operation(context), "result.json")
        return contracts._path(context["attempt_dir"], parameters.get("result_manifest", default))

    def _verify_result(
        self, context: Mapping[str, Any], manifest: Mapping[str, Any], verify_files: bool
    ) -> list[dict[str, str]]:
        if contracts._operation(context) == contracts.PREPARE_OPERATION:
            return prepare._verify_prepare_result(context, manifest, verify_files)
        if contracts._operation(context) == contracts.DATASET_OPERATION:
            return dataset._verify_dataset_assembly(context, manifest) if verify_files else []
        return label._verify_label_result(context, manifest, verify_files)

    def check(self, context: Any) -> dict[str, Any]:
        diagnostics = (
            prepare._validate_prepare(context, require_fresh_outputs=False)
            if isinstance(context, Mapping)
            and contracts._operation(context) == contracts.PREPARE_OPERATION
            else self.validate(context)
        )
        if contracts._errors(diagnostics):
            return {"plugin_id": contracts.PLUGIN_ID, "status": "FAIL", "diagnostics": diagnostics}
        assert isinstance(context, Mapping)
        if (
            context.get("backend") == "ssh-slurm"
            and contracts._operation(context) == contracts.LABEL_OPERATION
        ):
            raw_diagnostics, _ = label._scheduled_calculations_check(context)
            diagnostics.extend(raw_diagnostics)
            result_path = self._result_path(context)
            if result_path.exists() and not contracts._errors(diagnostics):
                manifest, read_diagnostic = contracts._read_json(result_path)
                if manifest is None:
                    assert read_diagnostic is not None
                    diagnostics.append(read_diagnostic)
                else:
                    diagnostics.extend(label._verify_scheduled_label_result(context, manifest))
            return {
                "plugin_id": contracts.PLUGIN_ID,
                "status": "FAIL" if contracts._errors(diagnostics) else "OK",
                "diagnostics": diagnostics,
                **({"result_manifest": str(result_path)} if result_path.exists() else {}),
            }
        result_path = self._result_path(context)
        manifest, read_diagnostic = contracts._read_json(result_path)
        if manifest is None:
            assert read_diagnostic is not None
            status = "WAIT" if read_diagnostic["level"] == "warning" else "FAIL"
            return {
                "plugin_id": contracts.PLUGIN_ID,
                "status": status,
                "diagnostics": [read_diagnostic],
            }
        diagnostics.extend(self._verify_result(context, manifest, verify_files=True))
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "FAIL" if contracts._errors(diagnostics) else "OK",
            "diagnostics": diagnostics,
            "result_manifest": str(result_path),
        }

    def collect(self, context: Any) -> dict[str, Any]:
        checked = self.check(context)
        if checked["status"] != "OK":
            return checked
        assert isinstance(context, Mapping)
        if (
            context.get("backend") == "ssh-slurm"
            and contracts._operation(context) == contracts.LABEL_OPERATION
        ):
            result_path = self._result_path(context)
            if result_path.exists():
                manifest, _ = contracts._read_json(result_path)
                assert manifest is not None
                return label._collected_scheduled_label(
                    context, manifest, checked.get("diagnostics", [])
                )
            return label._collect_scheduled_result(context)
        result_path = self._result_path(context)
        manifest, _ = contracts._read_json(result_path)
        assert manifest is not None
        parameters = contracts._mapping(context["parameters"])
        if contracts._operation(context) == contracts.DATASET_OPERATION:
            return dataset._collect_dataset_assembly(
                context, manifest, checked.get("diagnostics", [])
            )
        if contracts._operation(context) == contracts.PREPARE_OPERATION:
            artifacts: list[dict[str, Any]] = [
                {
                    "name": "dft-input-manifest",
                    "role": "dft-input-manifest",
                    "path": parameters.get("result_manifest", "dft-input-manifest.json"),
                    "media_type": "application/json",
                }
            ]
            for calculation in manifest["calculations"]:
                for name in ("POSCAR", "INCAR", "KPOINTS"):
                    record = dict(calculation["files"][name])
                    record["name"] = f"{calculation['structure_id']}-{name.lower()}"
                    record["role"] = record["name"]
                    artifacts.append(record)
            return {
                "plugin_id": contracts.PLUGIN_ID,
                "status": "OK",
                "artifacts": artifacts,
                "metrics": {
                    "structure_count": manifest["structure_count"],
                    "calculation_type": manifest["calculation_type"],
                    "potcar_ready": True,
                    "potcar_collected": False,
                },
            }
        artifacts = [
            {
                "name": "dft-labeling-result",
                "path": parameters.get("result_manifest", "dft-labeling-result.json"),
                "media_type": "application/json",
            },
            *[dict(item) for item in manifest["artifacts"]],
        ]
        return {
            "plugin_id": contracts.PLUGIN_ID,
            "status": "OK",
            "artifacts": artifacts,
            "metrics": {
                "source_structure_count": manifest["source_structure_count"],
                "label_count": manifest["label_count"],
                "electronic_converged": True,
                "ionic_converged": manifest["completion"].get("ionic_converged"),
            },
        }
