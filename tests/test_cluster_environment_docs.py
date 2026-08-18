from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "CLUSTER_ENVIRONMENTS.md"


def test_cluster_environment_guide_covers_each_isolated_framework() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    expected = {
        "deepmd": "mlip-deepmd",
        "mace": "mlip-mace",
        "chgnet": "mlip-chgnet",
        "m3gnet": "mlip-m3gnet",
    }
    for framework, family in expected.items():
        assert f"`{framework}`" in text
        assert f"`{family}/run.sh`" in text
        assert f"{family}/bin/python" in text

    assert "PyYAML" in text
    assert "does not need MLIPFlow itself or the MLIPFlow `dev` extra" in text
    assert "remote_template_root" in text
    assert "work_root" in text
    assert "site.yaml" in text
    assert "PYTHON_BIN" in text


def test_install_examples_do_not_combine_frameworks_or_pin_one_accelerator_stack() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    install_lines = [line for line in text.splitlines() if "-m pip install" in line]
    framework_packages = ("deepmd-kit", "mace-torch", "chgnet", "matgl")
    for line in install_lines:
        assert sum(package in line.lower() for package in framework_packages) <= 1
        assert not re.search(r"(?:torch|cuda|cu)={0,2}[0-9]", line, re.IGNORECASE)
    assert "pip install -e" not in text


def test_primary_cluster_docs_link_to_environment_guide() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    cluster_example = (ROOT / "examples" / "training_all_models" / "CLUSTER.md").read_text(
        encoding="utf-8"
    )
    assert "(docs/CLUSTER_ENVIRONMENTS.md)" in readme
    assert "(../../docs/CLUSTER_ENVIRONMENTS.md)" in cluster_example
