from __future__ import annotations

from mlipflow.plugins import BUILTIN_CAPABILITIES, capability_directory, load_adapter


EXPECTED_OPERATIONS = {
    "active-learning": ("committee-evaluate", "select-candidates", "assess-round"),
    "ase-md": ("run",),
    "candidate-ranking": ("rank-candidates",),
    "dft-labeling": ("vasp-prepare", "label", "dataset-assemble"),
    "electrochemical-voltage": ("compute-from-energies", "replay-si-table-s11"),
    "high-entropy-structure": ("generate-sqs",),
    "ionic-transport": ("analyze-existing", "md-smoke-and-analyze"),
    "lammps-md": ("lammps-prepare", "execute"),
    "mlip-benchmark": (
        "evaluate-fresh",
        "evaluate-static",
        "normalize-replay",
        "normalize-execute",
    ),
    "mlip-training": ("train", "finetune"),
    "pes-sampling": (
        "direct-select",
        "lasp-input-prepare",
        "merge-structures",
        "lasp-ssw-execute",
        "lasp-ssw-normalize-replay",
    ),
}


def test_exact_builtin_capabilities_and_scientific_operations() -> None:
    assert set(BUILTIN_CAPABILITIES) == set(EXPECTED_OPERATIONS)
    assert {
        name: tuple(spec["operations"])
        for name, spec in BUILTIN_CAPABILITIES.items()
    } == EXPECTED_OPERATIONS


def test_every_builtin_capability_resolves_its_adapter() -> None:
    for capability_id in BUILTIN_CAPABILITIES:
        assert capability_directory(capability_id).is_dir()
        adapter = load_adapter(capability_id)
        assert all(
            callable(getattr(adapter, method, None))
            for method in ("validate", "plan", "check", "collect")
        )
