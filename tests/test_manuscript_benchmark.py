"""Local tests for manuscript benchmark normalization.

The XLSX fixtures reproduce the column layouts written by the inventoried
CHGNet and M3GNet scripts.  No model, network, or scheduler is used.
"""

from __future__ import annotations

import csv
from tests.helpers import load_module
import json
import math
import subprocess
import tempfile
import unittest
import zipfile
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "mlipflow" / "plugins" / "mlip_benchmark" / "benchmark_wrapper.py"
ADAPTER_PATH = ROOT / "mlipflow" / "plugins" / "mlip_benchmark" / "adapter.py"


def load_wrapper():
    module = load_module(WRAPPER_PATH, 'test_manuscript_benchmark_wrapper')
    return module


def load_adapter():
    module = load_module(ADAPTER_PATH, 'test_manuscript_benchmark_adapter')
    return module.Adapter()


def _column_name(index: int) -> str:
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _sheet_xml(rows: list[list[object]]) -> str:
    xml_rows = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for column, value in enumerate(row):
            reference = f"{_column_name(column)}{row_number}"
            if isinstance(value, str):
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
                )
            else:
                cells.append(f'<c r="{reference}" t="n"><v>{value}</v></c>')
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData></worksheet>"
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    workbook_sheets = []
    relationships = []
    content_overrides = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (name, rows) in enumerate(sheets, start=1):
            workbook_sheets.append(
                f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
            )
            relationships.append(
                "<Relationship "
                f'Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
                f'worksheet" Target="worksheets/sheet{index}.xml"/>'
            )
            content_overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.'
                'spreadsheetml.worksheet+xml"/>'
            )
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows))
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{''.join(workbook_sheets)}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{''.join(relationships)}</Relationships>",
        )
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
            'relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            'spreadsheetml.sheet.main+xml"/>'
            f"{''.join(content_overrides)}</Types>",
        )


class ManuscriptBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = load_wrapper()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load_outputs(self, output: Path):
        return (
            json.loads((output / "metrics.json").read_text(encoding="utf-8")),
            json.loads((output / "model_ranking.json").read_text(encoding="utf-8")),
            json.loads((output / "provenance.json").read_text(encoding="utf-8")),
        )

    def test_prepared_json_supports_all_six_exact_models_and_directional_ranking(self) -> None:
        evidence = self.root / "manuscript_table_s2.json"
        records = []
        for index, model in enumerate(self.wrapper.MODEL_NAMES, start=1):
            records.append(
                {
                    "model": model,
                    "task": "static-pes",
                    "scenario": "manuscript-si-s2",
                    "split": "test",
                    "metric": "energy_rmse",
                    "value": index / 100,
                    "unit": "eV/atom",
                    "direction": "minimize",
                    "sample_count": 9753,
                }
            )
        evidence.write_text(json.dumps({"records": records}), encoding="utf-8")
        before = evidence.read_bytes()

        output = self.root / "benchmark"
        paths = self.wrapper.normalize_benchmark(
            [
                {
                    "path": evidence,
                    "evidence_locator": "evidence/manuscript_table_s2.json",
                }
            ],
            output,
            mode="replay",
        )

        self.assertEqual(set(self.wrapper.OUTPUT_NAMES), set(paths))
        self.assertTrue(all(path.is_file() for path in paths.values()))
        metrics, ranking, provenance = self._load_outputs(output)
        self.assertEqual(list(self.wrapper.MODEL_NAMES), metrics["supported_models"])
        self.assertEqual(
            set(self.wrapper.MODEL_NAMES), {row["model"] for row in metrics["records"]}
        )
        self.assertEqual({"replay"}, {row["mode"] for row in metrics["records"]})
        self.assertTrue(
            all(
                row["source_path"] == "evidence/manuscript_table_s2.json"
                and row["unit"]
                and row["direction"]
                and row["sample_count"] > 0
                for row in metrics["records"]
            )
        )
        candidates = ranking["rankings"][0]["candidates"]
        self.assertEqual("deepmd-se_atten_v2", candidates[0]["model"])
        self.assertEqual("replay", provenance["mode"])
        self.assertFalse(provenance["model_execution"])
        self.assertEqual(
            "evidence/manuscript_table_s2.json", provenance["source_evidence"][0]["path"]
        )
        self.assertEqual(before, evidence.read_bytes())
        self.assertNotIn(str(self.root), "\n".join(path.read_text() for path in paths.values()))

        relocated = self.root / "different-host-layout" / "renamed-input.json"
        relocated.parent.mkdir()
        relocated.write_bytes(evidence.read_bytes())
        relocated_output = self.root / "relocated-benchmark"
        self.wrapper.normalize_benchmark(
            [
                {
                    "path": relocated,
                    "evidence_locator": "evidence/manuscript_table_s2.json",
                }
            ],
            relocated_output,
            mode="replay",
        )
        for name in self.wrapper.OUTPUT_NAMES:
            self.assertEqual((output / name).read_bytes(), (relocated_output / name).read_bytes())

        with (output / "benchmark_summary.csv").open(encoding="utf-8", newline="") as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual(6, len(summary))
        self.assertEqual("eV/atom", summary[0]["unit"])

    def test_existing_adapter_contract_accepts_each_exact_model_name(self) -> None:
        adapter = load_adapter()
        for model in self.wrapper.MODEL_NAMES:
            with self.subTest(model=model):
                context = {
                    "project_root": str(self.root),
                    "attempt_dir": str(self.root / "attempt"),
                    "inputs": {
                        "executable": "/usr/bin/python3",
                        "script": "benchmark.py",
                        "model": "model.bin",
                        "benchmark_dataset": "pairs.json",
                        "benchmark_config": "config.json",
                        "result_manifest": "result.json",
                    },
                    "parameters": {
                        "operation": "evaluate-static",
                        "model_family": model,
                        "task": "static-pes",
                        "scenario": "manuscript",
                        "energy_normalization": "source-declared",
                        "stress_convention": "source-declared",
                    },
                    "backend": "local",
                    "resources": {},
                }
                self.assertEqual("READY", adapter.plan(context)["status"])

    def test_adapter_normalize_replay_roundtrip_uses_bundled_wrapper(self) -> None:
        adapter = load_adapter()
        first = self.root / "reported-energy.json"
        second = self.root / "reported-force.json"
        first.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "metric": "energy_rmse",
                            "value": 0.031,
                            "unit": "eV/atom",
                            "direction": "minimize",
                            "sample_count": 24,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        second.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "metric": "force_rmse",
                            "value": 0.052,
                            "unit": "eV/angstrom",
                            "direction": "minimize",
                            "sample_count": 72,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        source_script = self.root / "historical_benchmark.py"
        source_script.write_text("# provenance-only historical script\n", encoding="utf-8")
        attempt = self.root / "replay-attempt"
        attempt.mkdir()
        context = {
            "project_root": str(self.root),
            "attempt_dir": str(attempt),
            "inputs": {
                "evidence_inputs": [first.name, second.name],
                "output_dir": "normalized",
                "source_script": {
                    "path": source_script.name,
                    "source_script_locator": "source/historical_benchmark.py",
                },
            },
            "parameters": {
                "operation": "normalize-replay",
                "model_family": "chgnet",
                "task": "static-pes",
                "scenario": "adapter-replay-v1",
                "split": "test",
            },
            "backend": "local",
            "resources": {"cpus": 1},
        }

        before = set(self.root.rglob("*"))
        plan = adapter.plan(context)
        self.assertEqual("READY", plan["status"])
        self.assertFalse(plan["shell"])
        self.assertEqual("replay", plan["argv"][2])
        self.assertEqual(2, plan["argv"].count("--input"))
        self.assertNotIn("--overwrite", plan["argv"])
        self.assertEqual(4, len(plan["expected_outputs"]))
        self.assertEqual(before, set(self.root.rglob("*")))
        completed = subprocess.run(
            plan["argv"],
            cwd=plan["cwd"],
            shell=plan["shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        checked = adapter.check(context)
        self.assertEqual("OK", checked["status"], checked["diagnostics"])
        self.assertEqual("replay", checked["mode"])
        self.assertEqual(2, len(checked["records"]))
        self.assertEqual(
            {"reported-energy.json", "reported-force.json"},
            {record["source_path"] for record in checked["records"]},
        )
        collected = adapter.collect(context)
        self.assertEqual("OK", collected["status"])
        self.assertEqual(4, len(collected["artifacts"]))
        self.assertTrue(all(path["path"].startswith(str(attempt)) for path in collected["artifacts"]))

        blocked = adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn(
            "benchmark.normalize_contract", {item["code"] for item in blocked["diagnostics"]}
        )

        metrics_path = attempt / "normalized" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["mode"] = "execute"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        failed = adapter.check(context)
        self.assertEqual("FAIL", failed["status"])
        self.assertIn(
            "benchmark.normalized_contract", {item["code"] for item in failed["diagnostics"]}
        )

    def test_adapter_normalize_execute_pairs_roundtrip_never_runs_model(self) -> None:
        adapter = load_adapter()
        pairs = self.root / "prediction-pairs.csv"
        with pairs.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["target", "reference", "prediction"]
            )
            writer.writeheader()
            for reference, prediction in ((0, 0), (1, 2), (2, 2)):
                writer.writerow(
                    {
                        "target": "energy",
                        "reference": reference,
                        "prediction": prediction,
                    }
                )
        attempt = self.root / "execute-attempt"
        attempt.mkdir()
        context = {
            "project_root": str(self.root),
            "attempt_dir": str(attempt),
            "inputs": {
                "evidence_inputs": [
                    {
                        "path": pairs.name,
                        "evidence_locator": "evidence/prediction-pairs.csv",
                    }
                ],
                "output_dir": "normalized",
            },
            "parameters": {
                "operation": "normalize-execute",
                "model_family": "deepmd-dpa2",
                "task": "static-pes",
                "scenario": "adapter-execute-v1",
                "split": "test",
                "units": {"energy": "eV/atom"},
            },
            "backend": "local",
            "resources": {"cpus": 1},
        }

        plan = adapter.plan(context)
        self.assertEqual("READY", plan["status"])
        self.assertFalse(plan["shell"])
        self.assertEqual("execute", plan["argv"][2])
        self.assertIn("--energy-unit", plan["argv"])
        self.assertIn("--evidence-locator", plan["argv"])
        self.assertIn("no model execution", plan["calculation_claim"])
        completed = subprocess.run(
            plan["argv"],
            cwd=plan["cwd"],
            shell=plan["shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

        checked = adapter.check(context)
        self.assertEqual("OK", checked["status"], checked["diagnostics"])
        self.assertEqual("execute", checked["mode"])
        records = {item["metric"]: item for item in checked["records"]}
        self.assertAlmostEqual(1 / 3, records["energy_mae"]["value"])
        self.assertAlmostEqual(math.sqrt(1 / 3), records["energy_rmse"]["value"])
        self.assertAlmostEqual(math.sqrt(3) / 2, records["energy_pearson_r"]["value"])
        self.assertEqual(
            {"evidence/prediction-pairs.csv"},
            {item["source_path"] for item in checked["records"]},
        )
        self.assertEqual({"deepmd-dpa2"}, {item["model"] for item in checked["records"]})
        collected = adapter.collect(context)
        self.assertEqual("OK", collected["status"])
        self.assertEqual("execute", collected["mode"])
        self.assertEqual(3, len(collected["metrics"]))

        context["parameters"]["scenario"] = "unapproved-scenario"
        self.assertEqual("FAIL", adapter.check(context)["status"])
        context["parameters"]["scenario"] = "adapter-execute-v1"
        pairs.write_text(pairs.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertEqual("OK", adapter.check(context)["status"])

    def test_chgnet_historical_workbook_schema_is_normalized_without_unit_guessing(self) -> None:
        workbook = self.root / "efs_metrics_by_split.xlsx"
        header = [
            "Split",
            "NumStructures",
            "Energy_PearsonR",
            "Energy_MAE",
            "Energy_RMSE",
            "Energy_N",
            "Force_PearsonR",
            "Force_MAE",
            "Force_RMSE",
            "Force_N",
            "Stress_PearsonR",
            "Stress_MAE",
            "Stress_RMSE",
            "Stress_N",
        ]
        overall = [header, ["test", 3, 0.99, 0.01, 0.02, 3, 0.9, 0.1, 0.2, 9, -0.8, 1.0, 2.0, 27]]
        by_natoms_header = ["Split", "Natoms", *header[1:]]
        by_natoms = [
            by_natoms_header,
            ["test", 56, 2, 0.98, 0.02, 0.03, 2, 0.8, 0.2, 0.3, 6, -0.7, 1.1, 2.1, 18],
        ]
        write_xlsx(workbook, [("Overall", overall), ("ByNatoms", by_natoms)])

        output = self.root / "chgnet-normalized"
        self.wrapper.normalize_benchmark(
            [
                {
                    "path": workbook,
                    "task": "static-pes",
                    "scenario": "chgnet-holdout",
                    "units": {"energy": "eV/atom"},
                }
            ],
            output,
        )
        metrics, _, provenance = self._load_outputs(output)
        self.assertEqual(18, metrics["record_count"])
        energy_rmse = next(
            row
            for row in metrics["records"]
            if row["metric"] == "energy_rmse" and row["dimensions"]["scope"] == "overall"
        )
        force_rmse = next(
            row
            for row in metrics["records"]
            if row["metric"] == "force_rmse" and row["dimensions"]["scope"] == "overall"
        )
        stress_r = next(
            row
            for row in metrics["records"]
            if row["metric"] == "stress_pearson_r" and row["dimensions"]["scope"] == "overall"
        )
        self.assertEqual("chgnet", energy_rmse["model"])
        self.assertEqual(3, energy_rmse["sample_count"])
        self.assertEqual("eV/atom", energy_rmse["unit"])
        self.assertEqual("source-unit-unspecified", force_rmse["unit"])
        self.assertEqual("not-declared-by-source", force_rmse["unit_provenance"])
        self.assertEqual("dimensionless", stress_r["unit"])
        self.assertEqual("chgnet-efs-metrics-xlsx", provenance["source_evidence"][0]["parser"])

    def test_deepmd_historical_workbook_schema_maps_counts_and_directions(self) -> None:
        workbook = self.root / "deepmd_metrics.xlsx"
        header = [
            "split",
            "n_dirs_input",
            "n_system_dirs",
            "n_structures(frames)",
            "n_atoms_total(frames*natoms)",
            "n_force_components",
            "energy_is_per_atom",
            "energy_pearson_r",
            "energy_RMSE",
            "force_pearson_r",
            "force_RMSE",
        ]
        write_xlsx(
            workbook,
            [
                (
                    "Sheet1",
                    [
                        header,
                        [
                            "test",
                            3,
                            747,
                            13024,
                            891242,
                            2673726,
                            True,
                            0.998,
                            0.007905963,
                            0.95,
                            0.185693608,
                        ],
                    ],
                )
            ],
        )
        output = self.root / "deepmd-normalized"
        self.wrapper.normalize_benchmark(
            [
                {
                    "path": workbook,
                    "evidence_locator": "evidence/deepmd_metrics.xlsx",
                    "model": "deepmd-se_atten_v2",
                    "task": "static-pes",
                    "scenario": "manuscript-holdout",
                    "units": {"energy": "eV/atom", "force": "eV/angstrom"},
                }
            ],
            output,
            mode="replay",
        )
        metrics, _, provenance = self._load_outputs(output)
        records = {record["metric"]: record for record in metrics["records"]}
        self.assertEqual("minimize", records["energy_rmse"]["direction"])
        self.assertEqual(13024, records["energy_rmse"]["sample_count"])
        self.assertEqual(2673726, records["force_rmse"]["sample_count"])
        self.assertEqual("maximize", records["force_pearson_r"]["direction"])
        self.assertEqual("deepmd-metrics-xlsx", provenance["source_evidence"][0]["parser"])

    def test_m3gnet_historical_workbook_schema_uses_scalar_sample_count(self) -> None:
        workbook = self.root / "metrics_summary.xlsx"
        header = ["split", "target", "r", "MAE", "RMSE", "N_struct", "N_scalar"]
        overall = [
            header,
            ["test", "energy", 0.97, 0.02, 0.03, 4, 4],
            ["test", "force", 0.88, 0.12, 0.22, 4, 36],
        ]
        by_natoms = [
            [*header, "natoms", "count_in_bin"],
            ["test", "energy", 0.96, 0.03, 0.04, 2, 2, 56, 2],
        ]
        write_xlsx(workbook, [("overall", overall), ("by_natoms", by_natoms)])

        output = self.root / "m3gnet-normalized"
        self.wrapper.normalize_benchmark(
            [
                {
                    "path": workbook,
                    "task": "static-pes",
                    "scenario": "m3gnet-holdout",
                    "units": {"energy": "eV/atom", "force": "eV/angstrom"},
                }
            ],
            output,
        )
        metrics, _, provenance = self._load_outputs(output)
        force_rmse = next(
            row
            for row in metrics["records"]
            if row["metric"] == "force_rmse" and row["dimensions"]["scope"] == "overall"
        )
        self.assertEqual("m3gnet", force_rmse["model"])
        self.assertEqual(36, force_rmse["sample_count"])
        self.assertEqual(4, force_rmse["dimensions"]["num_structures"])
        self.assertEqual("m3gnet-metrics-summary-xlsx", provenance["source_evidence"][0]["parser"])

    def test_prepared_csv_and_xlsx_are_both_strict_replay_inputs(self) -> None:
        csv_evidence = self.root / "manuscript_voltage.csv"
        with csv_evidence.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "model",
                    "task",
                    "scenario",
                    "split",
                    "metric",
                    "value",
                    "unit",
                    "direction",
                    "sample_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "model": "chgnet",
                    "task": "voltage",
                    "scenario": "manuscript-si",
                    "split": "test",
                    "metric": "voltage_mae",
                    "value": 0.12,
                    "unit": "V",
                    "direction": "minimize",
                    "sample_count": 8,
                }
            )
        self.wrapper.normalize_benchmark(
            [{"path": csv_evidence}], self.root / "csv-replay", mode="replay"
        )
        csv_metrics, _, csv_provenance = self._load_outputs(self.root / "csv-replay")
        self.assertEqual("prepared-csv", csv_provenance["source_evidence"][0]["parser"])
        self.assertEqual("V", csv_metrics["records"][0]["unit"])

        xlsx_evidence = self.root / "prepared_si_table.xlsx"
        write_xlsx(
            xlsx_evidence,
            [
                (
                    "Table S transport",
                    [
                        [
                            "model",
                            "task",
                            "scenario",
                            "split",
                            "metric",
                            "value",
                            "unit",
                            "direction",
                            "sample_count",
                        ],
                        [
                            "deepmd-se_atten_v2",
                            "ionic-transport",
                            "manuscript-si",
                            "test",
                            "diffusion_mae",
                            0.08,
                            "cm^2/s",
                            "minimize",
                            6,
                        ],
                    ],
                )
            ],
        )
        self.wrapper.normalize_benchmark(
            [{"path": xlsx_evidence}], self.root / "xlsx-replay", mode="replay"
        )
        xlsx_metrics, _, xlsx_provenance = self._load_outputs(self.root / "xlsx-replay")
        self.assertEqual("prepared-xlsx", xlsx_provenance["source_evidence"][0]["parser"])
        self.assertEqual("deepmd-se_atten_v2", xlsx_metrics["records"][0]["model"])

    def test_execute_recomputes_metrics_from_supplied_pairs_but_never_runs_model(self) -> None:
        pairs = self.root / "prediction_pairs.csv"
        with pairs.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "model",
                    "task",
                    "scenario",
                    "split",
                    "target",
                    "unit",
                    "reference",
                    "prediction",
                ],
            )
            writer.writeheader()
            for reference, prediction in ((0, 0), (1, 2), (2, 2)):
                writer.writerow(
                    {
                        "model": "CHGNet",
                        "task": "static-pes",
                        "scenario": "tiny-paired-evidence",
                        "split": "test",
                        "target": "energy",
                        "unit": "eV/atom",
                        "reference": reference,
                        "prediction": prediction,
                    }
                )

        output = self.root / "executed"
        self.wrapper.normalize_benchmark([{"path": pairs}], output, mode="execute")
        metrics, _, provenance = self._load_outputs(output)
        values = {record["metric"]: record for record in metrics["records"]}
        self.assertAlmostEqual(1 / 3, values["energy_mae"]["value"])
        self.assertAlmostEqual(math.sqrt(1 / 3), values["energy_rmse"]["value"])
        self.assertAlmostEqual(math.sqrt(3) / 2, values["energy_pearson_r"]["value"])
        self.assertEqual(3, values["energy_rmse"]["sample_count"])
        self.assertEqual("execute", metrics["mode"])
        self.assertFalse(provenance["model_execution"])
        self.assertIn("does not run an MLIP", provenance["limitations"][0])

    def test_execute_adapter_accepts_explicit_unavailable_single_pair_pearson(self) -> None:
        pairs = self.root / "single_prediction_pair.csv"
        with pairs.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "model",
                    "task",
                    "scenario",
                    "split",
                    "target",
                    "unit",
                    "reference",
                    "prediction",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "model": "chgnet",
                    "task": "static-pes",
                    "scenario": "single-held-out-structure",
                    "split": "test",
                    "target": "energy",
                    "unit": "eV/atom",
                    "reference": 1.0,
                    "prediction": 1.2,
                }
            )

        attempt = self.root / "single-pair-attempt"
        attempt.mkdir()
        context = {
            "project_root": str(self.root),
            "attempt_dir": str(attempt),
            "inputs": {
                "evidence_inputs": [
                    {
                        "path": pairs.name,
                        "evidence_locator": "evidence/single-prediction-pair.csv",
                    }
                ],
                "output_dir": "normalized",
            },
            "parameters": {
                "operation": "normalize-execute",
                "model_family": "chgnet",
                "task": "static-pes",
                "scenario": "single-held-out-structure",
                "split": "test",
                "units": {"energy": "eV/atom"},
            },
            "backend": "local",
            "resources": {"cpus": 1},
        }
        adapter = load_adapter()
        plan = adapter.plan(context)
        completed = subprocess.run(
            plan["argv"],
            cwd=plan["cwd"],
            shell=plan["shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        checked = adapter.check(context)
        self.assertEqual("OK", checked["status"], checked["diagnostics"])
        metrics = json.loads(
            (attempt / "normalized" / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            ["energy_mae", "energy_rmse"],
            [record["metric"] for record in metrics["records"]],
        )
        self.assertEqual("energy_pearson_r", metrics["unavailable_metrics"][0]["metric"])
        self.assertEqual(
            "insufficient-scalar-pairs", metrics["unavailable_metrics"][0]["reason"]
        )

    def test_replay_rejects_missing_units_unknown_models_and_silent_overwrite(self) -> None:
        evidence = self.root / "bad.json"
        base = {
            "model": "generic-deepmd",
            "task": "static-pes",
            "scenario": "bad",
            "split": "test",
            "metric": "energy_rmse",
            "value": 0.1,
            "unit": "eV/atom",
            "direction": "minimize",
            "sample_count": 2,
        }
        evidence.write_text(json.dumps([base]), encoding="utf-8")
        with self.assertRaisesRegex(self.wrapper.BenchmarkNormalizationError, "unsupported"):
            self.wrapper.normalize_benchmark([{"path": evidence}], self.root / "bad-output")

        base["model"] = "deepmd-se_e2_a"
        base.pop("unit")
        evidence.write_text(json.dumps([base]), encoding="utf-8")
        with self.assertRaisesRegex(self.wrapper.BenchmarkNormalizationError, "unit"):
            self.wrapper.normalize_benchmark([{"path": evidence}], self.root / "bad-output-2")

        base["unit"] = "eV/atom"
        evidence.write_text(json.dumps([base]), encoding="utf-8")
        output = self.root / "valid-output"
        self.wrapper.normalize_benchmark([{"path": evidence}], output)
        with self.assertRaisesRegex(self.wrapper.BenchmarkNormalizationError, "overwrite"):
            self.wrapper.normalize_benchmark([{"path": evidence}], output)

    def test_source_file_can_never_be_overwritten_as_a_normalized_output(self) -> None:
        output = self.root / "same-directory"
        output.mkdir()
        evidence = output / "metrics.json"
        evidence.write_text(
            json.dumps(
                [
                    {
                        "model": "chgnet",
                        "task": "static-pes",
                        "scenario": "collision",
                        "split": "test",
                        "metric": "energy_rmse",
                        "value": 0.1,
                        "unit": "eV/atom",
                        "direction": "minimize",
                        "sample_count": 2,
                    }
                ]
            ),
            encoding="utf-8",
        )
        before = evidence.read_bytes()
        with self.assertRaisesRegex(self.wrapper.BenchmarkNormalizationError, "read-only"):
            self.wrapper.normalize_benchmark(
                [{"path": evidence}], output, mode="replay", overwrite=True
            )
        self.assertEqual(before, evidence.read_bytes())


if __name__ == "__main__":
    unittest.main()
