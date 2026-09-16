from __future__ import annotations

from mlipipe.plugins import BUILTIN_CAPABILITIES, capability_directory, load_adapter


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

EXPECTED_APPROVAL_OPERATIONS = {
    "active-learning": (),
    "ase-md": ("run",),
    "candidate-ranking": (),
    "dft-labeling": ("label",),
    "electrochemical-voltage": (),
    "high-entropy-structure": (),
    "ionic-transport": (),
    "lammps-md": ("execute",),
    "mlip-benchmark": ("evaluate-fresh",),
    "mlip-training": ("train", "finetune"),
    "pes-sampling": ("lasp-ssw-execute",),
}

EXPECTED_DEFAULT_OPERATIONS = {
    "active-learning": "",
    "ase-md": "run",
    "candidate-ranking": "rank-candidates",
    "dft-labeling": "",
    "electrochemical-voltage": "compute-from-energies",
    "high-entropy-structure": "generate-sqs",
    "ionic-transport": "analyze-existing",
    "lammps-md": "lammps-prepare",
    "mlip-benchmark": "evaluate-static",
    "mlip-training": "train",
    "pes-sampling": "direct-select",
}


def test_exact_builtin_capabilities_and_scientific_operations() -> None:
    assert set(BUILTIN_CAPABILITIES) == set(EXPECTED_OPERATIONS)
    assert {
        name: tuple(spec["operations"])
        for name, spec in BUILTIN_CAPABILITIES.items()
    } == EXPECTED_OPERATIONS
    assert {
        name: tuple(spec["approval_operations"])
        for name, spec in BUILTIN_CAPABILITIES.items()
    } == EXPECTED_APPROVAL_OPERATIONS


def test_every_builtin_capability_resolves_its_adapter() -> None:
    for capability_id in BUILTIN_CAPABILITIES:
        assert capability_directory(capability_id).is_dir()
        adapter = load_adapter(capability_id)
        assert all(
            callable(getattr(adapter, method, None))
            for method in ("operation", "validate", "plan", "check", "collect")
        )
        assert adapter.operation({"parameters": {}}) == EXPECTED_DEFAULT_OPERATIONS[
            capability_id
        ]
