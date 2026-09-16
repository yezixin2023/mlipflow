from __future__ import annotations

import unittest

from mlipipe.routing import route_models


class RoutingTests(unittest.TestCase):
    def test_external_evidence_path_status_is_reported(self) -> None:
        registry = {
            "schema_version": 1,
            "models": [
                {
                    "id": "external",
                    "elements": ["Li"],
                    "tasks": ["x"],
                    "benchmarks": [
                        {
                            "task": "x",
                            "scenario": "s",
                            "run_id": "r1",
                            "validation_samples": 1,
                            "metrics": {"mae": 0.1},
                        }
                    ],
                }
            ],
            "_evidence_files": [
                {
                    "model_id": "external",
                    "run_id": "r1",
                    "status": "external",
                }
            ],
        }
        result = route_models(
            registry,
            task="x",
            elements={"Li"},
            scenario="s",
            policy={"metrics": [{"name": "mae", "direction": "minimize", "weight": 1}]},
        )
        self.assertEqual("external", result["selected_model"])
        self.assertEqual("external", result["evidence_files"][0]["status"])

    def test_task_specific_evidence_selects_model(self) -> None:
        registry = {
            "schema_version": 1,
            "models": [
                {
                    "id": "deepmd-se-atten",
                    "elements": ["Li", "P", "S"],
                    "tasks": ["ionic-transport"],
                    "benchmarks": [
                        {
                            "task": "ionic-transport",
                            "scenario": "li-thiophosphate",
                            "run_id": "b1",
                            "validation_samples": 30,
                            "metrics": {"diffusion_mae": 0.08},
                        }
                    ],
                },
                {
                    "id": "chgnet",
                    "elements": ["Li", "P", "S"],
                    "tasks": ["ionic-transport"],
                    "benchmarks": [
                        {
                            "task": "ionic-transport",
                            "scenario": "li-thiophosphate",
                            "run_id": "b2",
                            "validation_samples": 100,
                            "metrics": {"diffusion_mae": 0.21},
                        }
                    ],
                },
            ],
        }
        result = route_models(
            registry,
            task="ionic-transport",
            elements={"Li", "P", "S"},
            scenario="li-thiophosphate",
            policy={
                "metrics": [
                    {"name": "diffusion_mae", "direction": "minimize", "weight": 1.0}
                ]
            },
        )
        self.assertEqual(result["selected_model"], "deepmd-se-atten")
        self.assertEqual([item["model_id"] for item in result["ranking"]], ["deepmd-se-atten", "chgnet"])

    def test_missing_elements_and_metrics_are_explained(self) -> None:
        result = route_models(
            {
                "schema_version": 1,
                "models": [
                    {"id": "bad", "elements": ["Li"], "tasks": ["x"], "benchmarks": []}
                ],
            },
            task="x",
            elements={"Li", "S"},
            scenario="s",
            policy={"metrics": [{"name": "mae", "direction": "minimize", "weight": 1}]},
        )
        self.assertIsNone(result["selected_model"])
        self.assertIn("missing elements", " ".join(result["rejected"][0]["reasons"]))

    def test_ties_are_stable_by_model_id(self) -> None:
        models = []
        for model_id in ("z-model", "a-model"):
            models.append(
                {
                    "id": model_id,
                    "elements": ["Li"],
                    "tasks": ["x"],
                    "benchmarks": [
                        {
                            "task": "x",
                            "scenario": "s",
                            "validation_samples": 1,
                            "metrics": {"mae": 1.0},
                        }
                    ],
                }
            )
        result = route_models(
            {"schema_version": 1, "models": models},
            task="x",
            elements={"Li"},
            scenario="s",
            policy={"metrics": [{"name": "mae", "direction": "minimize", "weight": 1}]},
        )
        self.assertEqual(result["selected_model"], "a-model")


if __name__ == "__main__":
    unittest.main()
