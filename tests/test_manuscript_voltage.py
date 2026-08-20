"""Tests for deterministic SI Table S11 voltage evidence replay."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "plugins" / "electrochemical-voltage" / "manuscript_replay.py"
ADAPTER_PATH = ROOT / "plugins" / "electrochemical-voltage" / "adapter.py"
PLUGIN_PATH = ROOT / "plugins" / "electrochemical-voltage" / "plugin.yaml"
def load_replay():
    spec = importlib.util.spec_from_file_location("test_manuscript_voltage_replay", REPLAY_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {REPLAY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter():
    spec = importlib.util.spec_from_file_location("test_manuscript_voltage_adapter", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter()


# Nine distinct SI Table S11 systems.  The source table contains a duplicated
# Prototype II mixed-metal column; a strict long-form transcription includes it
# once, giving nine systems x two voltage stages = 18 evidence rows.
TABLE_S11_SYSTEMS = (
    (
        "Prototype I",
        "MnFeNiCu3Zn2",
        "Li8-Li4-Li0",
        (2.83, 2.66),
        (1.96, 2.17),
        (5.27, 5.44),
        (1.43, 1.49),
        (2.16, 2.22),
        (2.14, 2.31),
        (2.91, 2.75),
    ),
    (
        "Prototype I",
        "Mn",
        "Li2-Li1-Li0",
        (2.58, 2.43),
        (2.14, 2.36),
        (5.23, 5.42),
        (1.45, 1.42),
        (1.90, 2.04),
        (1.89, 1.94),
        (2.56, 2.52),
    ),
    (
        "Prototype I",
        "Fe",
        "Li2-Li1-Li0",
        (2.60, 2.41),
        (2.21, 2.35),
        (5.26, 5.28),
        (1.61, 1.60),
        (1.93, 2.02),
        (1.85, 2.13),
        (2.70, 2.63),
    ),
    (
        "Prototype II",
        "MnFeNiCu2Zn2",
        "Li10-Li5-Li0",
        (2.72, 2.87),
        (1.84, 2.17),
        (5.15, 5.32),
        (1.39, 1.40),
        (2.08, 2.28),
        (2.09, 2.40),
        (2.79, 2.97),
    ),
    (
        "Prototype II",
        "Mn",
        "Li10-Li5-Li0",
        (2.44, 2.69),
        (1.75, 1.82),
        (4.93, 5.19),
        (1.18, 1.31),
        (1.80, 2.12),
        (1.66, 1.93),
        (2.60, 2.56),
    ),
    (
        "Prototype II",
        "Fe",
        "Li10-Li5-Li0",
        (2.39, 2.67),
        (1.87, 1.67),
        (5.06, 4.98),
        (1.40, 1.41),
        (1.87, 2.05),
        (1.68, 2.10),
        (2.61, 2.60),
    ),
    (
        "Prototype III",
        "MnFeNiCuZn4",
        "Li16-Li8-Li0",
        (2.81, 3.13),
        (1.44, 2.41),
        (4.94, 5.83),
        (1.45, 1.18),
        (1.79, 2.64),
        (2.16, 2.65),
        (2.67, 3.45),
    ),
    (
        "Prototype III",
        "Mn",
        "Li2-Li1-Li0",
        (1.82, 2.46),
        (1.07, 2.07),
        (4.55, 5.54),
        (1.54, 1.02),
        (1.32, 2.26),
        (1.75, 2.21),
        (2.34, 3.02),
    ),
    (
        "Prototype III",
        "Fe",
        "Li2-Li1-Li0",
        (2.31, 2.89),
        (1.25, 2.16),
        (4.79, 5.68),
        (1.66, 1.18),
        (1.38, 2.23),
        (1.83, 2.34),
        (2.52, 3.08),
    ),
)


def manuscript_rows(replay) -> list[dict[str, object]]:
    rows = []
    for system in TABLE_S11_SYSTEMS:
        prototype, composition, sequence, reference, *predictions = system
        lithium = sequence.split("-")
        for stage_index in range(2):
            row: dict[str, object] = {
                "source_table": replay.SOURCE_TABLE,
                "unit_source": "SI Table S11 caption and voltage-value context",
                "prototype": prototype,
                "metal_composition": composition,
                "lithium_sequence": sequence,
                "stage": f"{lithium[stage_index]}->{lithium[stage_index + 1]}",
                "reference_dft_v": reference[stage_index],
                "unit": "V",
            }
            row.update(
                {
                    model: values[stage_index]
                    for model, values in zip(replay.MODEL_COLUMNS, predictions)
                }
            )
            rows.append(row)
    return rows


def write_csv(path: Path, replay, rows: list[dict[str, object]], fields=None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields or replay.CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


class ManuscriptVoltageReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.replay = load_replay()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_real_table_replay_ranks_chgnet_first_and_recomputes_every_error(self) -> None:
        evidence = self.root / "table_s11.csv"
        write_csv(evidence, self.replay, manuscript_rows(self.replay))
        evidence_before = evidence.read_bytes()

        output = self.root / "replay"
        paths = self.replay.replay_table(evidence, output)
        self.assertEqual(set(self.replay.OUTPUT_NAMES), set(paths))
        self.assertTrue(all(path.is_file() for path in paths.values()))

        metrics = json.loads(paths["metrics.json"].read_text(encoding="utf-8"))
        ranking = json.loads(paths["model_ranking.json"].read_text(encoding="utf-8"))
        provenance = json.loads(paths["provenance.json"].read_text(encoding="utf-8"))
        self.assertEqual(18, metrics["row_count"])
        self.assertEqual(18, len(metrics["row_errors"]))
        self.assertEqual(list(self.replay.MODEL_COLUMNS), [item["model"] for item in metrics["models"]])
        for row in metrics["row_errors"]:
            for model in self.replay.MODEL_COLUMNS:
                expected = row["predictions_v"][model] - row["reference_dft_v"]
                self.assertAlmostEqual(expected, row["signed_errors_v"][model], places=15)
                self.assertAlmostEqual(abs(expected), row["absolute_errors_v"][model], places=15)

        by_model = {item["model"]: item for item in metrics["models"]}
        self.assertEqual(18, by_model["chgnet"]["N"])
        self.assertTrue(
            math.isclose(
                by_model["chgnet"]["mae_v"],
                0.182777777777778,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertEqual("chgnet", ranking["winner"])
        self.assertEqual("chgnet", ranking["ranking"][0]["model"])
        self.assertEqual(list(range(1, 7)), [item["rank"] for item in ranking["ranking"]])

        self.assertFalse(provenance["model_execution"])
        self.assertFalse(provenance["dft_execution"])
        self.assertFalse(provenance["recomputed_from_total_energies"])
        self.assertIn("does not recompute voltages", provenance["disclaimer"])
        self.assertNotIn(str(self.root), "\n".join(path.read_text() for path in paths.values()))
        self.assertEqual(evidence_before, evidence.read_bytes())

        relocated = self.root / "different-layout" / "renamed.csv"
        relocated.parent.mkdir()
        relocated.write_bytes(evidence.read_bytes())
        relocated_output = self.root / "relocated-output"
        relocated_paths = self.replay.replay_table(relocated, relocated_output)
        for name in self.replay.OUTPUT_NAMES:
            self.assertEqual(paths[name].read_bytes(), relocated_paths[name].read_bytes())

    def test_strict_validation_rejects_missing_nonfinite_non_v_and_duplicate_rows(self) -> None:
        base_rows = manuscript_rows(self.replay)
        invalid_cases = {}

        missing_model = [dict(row) for row in base_rows]
        missing_model[0]["chgnet"] = ""
        invalid_cases["missing model"] = missing_model

        nonfinite = [dict(row) for row in base_rows]
        nonfinite[0]["reference_dft_v"] = "NaN"
        invalid_cases["non-finite"] = nonfinite

        wrong_unit = [dict(row) for row in base_rows]
        wrong_unit[0]["unit"] = "mV"
        invalid_cases["non-V unit"] = wrong_unit

        duplicate = [dict(row) for row in base_rows]
        duplicate.append(dict(duplicate[0]))
        invalid_cases["duplicate row"] = duplicate

        for name, rows in invalid_cases.items():
            with self.subTest(case=name):
                path = self.root / f"{name.replace(' ', '-')}.csv"
                write_csv(path, self.replay, rows)
                with self.assertRaises(self.replay.ReplayValidationError):
                    self.replay.build_artifacts(path.read_bytes())

        bad_header = self.root / "bad-header.csv"
        fields = tuple(field for field in self.replay.CSV_FIELDS if field != "m3gnet")
        write_csv(bad_header, self.replay, base_rows, fields=fields)
        with self.assertRaisesRegex(self.replay.ReplayValidationError, "header must exactly match"):
            self.replay.build_artifacts(bad_header.read_bytes())

    def test_cli_writes_three_artifacts_and_refuses_to_overwrite(self) -> None:
        evidence = self.root / "table.csv"
        write_csv(evidence, self.replay, manuscript_rows(self.replay))
        output = self.root / "cli-output"
        command = [
            sys.executable,
            str(REPLAY_PATH),
            "--input",
            str(evidence),
            "--output-dir",
            str(output),
        ]
        first = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(set(self.replay.OUTPUT_NAMES), set(json.loads(first.stdout)))

        second = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(2, second.returncode)
        self.assertIn("refusing to overwrite", second.stderr)
        self.assertEqual(3, len(list(output.glob("*.json"))))


class ManuscriptVoltageAdapterIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.replay = load_replay()
        self.adapter = load_adapter()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.attempt = self.root / "attempt"
        self.attempt.mkdir()
        self.evidence = self.root / "table_s11.csv"
        write_csv(self.evidence, self.replay, manuscript_rows(self.replay))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def context(self, **parameter_overrides):
        parameters = {"operation": "replay-si-table-s11"}
        parameters.update(parameter_overrides)
        return {
            "project_root": str(self.root),
            "attempt_dir": str(self.attempt),
            "inputs": {"manuscript_voltage_table": self.evidence.name},
            "parameters": parameters,
            "backend": "local",
            "resources": {},
        }

    def test_manifest_advertises_only_implemented_operations(self) -> None:
        manifest = json.loads(PLUGIN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            ["compute-from-energies", "replay-si-table-s11"],
            manifest["execution"]["operations"],
        )

    def test_standard_adapter_plan_execute_check_collect_and_no_write_replay(self) -> None:
        context = self.context()
        output = self.attempt / "manuscript-voltage-replay"
        evidence_before = self.evidence.read_bytes()

        replayed = self.adapter.replay(context)
        self.assertEqual("OK", replayed["status"])
        self.assertFalse(replayed["executable"])
        self.assertEqual("replay", replayed["mode"])
        self.assertFalse(replayed["model_execution"])
        self.assertFalse(replayed["recomputed_from_total_energies"])
        self.assertFalse(output.exists(), "Adapter.replay must not create the output directory")
        in_memory = replayed["result_manifest"]["artifacts"]
        self.assertEqual(set(self.replay.OUTPUT_NAMES), set(in_memory))
        self.assertEqual("chgnet", in_memory["model_ranking.json"]["winner"])
        self.assertEqual("WAIT", self.adapter.check(context)["status"])

        plan = self.adapter.plan(context)
        self.assertEqual("READY", plan["status"])
        self.assertEqual("replay-si-table-s11", plan["operation"])
        self.assertEqual("replay", plan["mode"])
        self.assertFalse(plan["shell"])
        self.assertFalse(plan["model_execution"])
        self.assertFalse(plan["recomputed_from_total_energies"])
        self.assertEqual(str(REPLAY_PATH), plan["argv"][1])
        self.assertEqual(3, len(plan["expected_outputs"]))
        self.assertFalse(output.exists(), "planning must be zero-write")

        completed = subprocess.run(
            plan["argv"], cwd=plan["cwd"], check=False, capture_output=True, text=True
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual("OK", self.adapter.check(context)["status"])
        collected = self.adapter.collect(context)
        self.assertEqual("OK", collected["status"])
        self.assertEqual("replay", collected["mode"])
        self.assertFalse(collected["model_execution"])
        self.assertFalse(collected["recomputed_from_total_energies"])
        self.assertEqual("chgnet", collected["selection"]["winner"])
        self.assertEqual(18, collected["metrics"]["row_count"])
        self.assertEqual(6, collected["metrics"]["model_count"])
        self.assertEqual(
            {
                "manuscript-replay-metrics",
                "manuscript-replay-ranking",
                "manuscript-replay-provenance",
            },
            {item["name"] for item in collected["artifacts"]},
        )
        self.assertEqual(
            {item["name"] for item in collected["artifacts"]},
            {item["role"] for item in collected["artifacts"]},
        )
        provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual("manuscript-table-replay", provenance["evidence_mode"])
        self.assertEqual("V", provenance["voltage_unit"])
        self.assertFalse(provenance["model_execution"])
        self.assertFalse(provenance["recomputed_from_total_energies"])
        self.assertEqual(evidence_before, self.evidence.read_bytes())

        blocked = self.adapter.plan(context)
        self.assertEqual("BLOCKED", blocked["status"])
        self.assertIn("output_exists", blocked["diagnostics"][0]["code"])

    def test_adapter_rejects_non_v_and_tampered_outputs(self) -> None:
        rows = manuscript_rows(self.replay)
        rows[0]["unit"] = "mV"
        write_csv(self.evidence, self.replay, rows)
        non_v = self.adapter.plan(self.context())
        self.assertEqual("BLOCKED", non_v["status"])
        self.assertIn(
            "manuscript_replay.input_invalid",
            {item["code"] for item in non_v["diagnostics"]},
        )

        write_csv(
            self.evidence,
            self.replay,
            manuscript_rows(self.replay),
            fields=tuple(field for field in self.replay.CSV_FIELDS if field != "m3gnet"),
        )
        wrong_columns = self.adapter.plan(self.context())
        self.assertEqual("BLOCKED", wrong_columns["status"])
        self.assertIn(
            "manuscript_replay.input_invalid",
            {item["code"] for item in wrong_columns["diagnostics"]},
        )

        write_csv(self.evidence, self.replay, manuscript_rows(self.replay))
        context = self.context()
        plan = self.adapter.plan(context)
        completed = subprocess.run(
            plan["argv"], cwd=plan["cwd"], check=False, capture_output=True, text=True
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        provenance_path = self.attempt / "manuscript-voltage-replay" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["model_execution"] = True
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        checked = self.adapter.check(context)
        self.assertEqual("FAIL", checked["status"])
        self.assertIn(
            "manuscript_replay.model_execution",
            {item["code"] for item in checked["diagnostics"]},
        )


if __name__ == "__main__":
    unittest.main()
