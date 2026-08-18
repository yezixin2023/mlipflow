from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _extras() -> dict[str, list[str]]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def _requirements(items: list[str]) -> dict[str, Requirement]:
    return {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, items)
    }


def _minimum_version(requirement: Requirement) -> Version | None:
    lower_bounds = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator == ">="
    ]
    return max(lower_bounds, default=None)


def test_local_extra_covers_all_general_science_runtime_extras() -> None:
    extras = _extras()
    assert "local" in extras

    local = _requirements(extras["local"])
    for extra_name in ("science", "dft", "transport"):
        for dependency_name, requirement in _requirements(extras[extra_name]).items():
            assert dependency_name in local, (
                f"local does not include {extra_name}'s {dependency_name}"
            )
            required_minimum = _minimum_version(requirement)
            local_minimum = _minimum_version(local[dependency_name])
            if required_minimum is not None:
                assert local_minimum is not None
                assert local_minimum >= required_minimum


def test_local_extra_excludes_development_and_mlip_frameworks() -> None:
    names = set(_requirements(_extras()["local"]))
    assert names.isdisjoint({"pytest", "jsonschema", "ruff"})
    assert names.isdisjoint(
        {
            "deepmd",
            "deepmd-kit",
            "mace",
            "mace-torch",
            "chgnet",
            "matgl",
            "m3gnet",
        }
    )


def test_readme_recommends_the_packaged_local_extra_by_default() -> None:
    assert "local" in _extras()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install_section = readme.split("### 1. Install from source", 1)[1]
    recommended_section = install_section.split("#### Advanced / minimal installation", 1)[0]
    assert 'python -m pip install -e ".[local]"' in recommended_section
    assert "python -m pip install -e .\n" not in recommended_section
