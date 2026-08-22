"""Static enforcement of the architectural boundaries.

These boundaries used to be maintained by a comment and by convention.  Parsing
the imports turns each of them into something a change cannot silently break.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "mlipflow"


def module_imports(path: Path) -> set[str]:
    """Return the top-level module names a file imports.

    Relative imports are reported as the sibling module name, so ``from
    ..backends import X`` inside the services package yields ``backends``.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".", 1)[0])
            elif node.level:
                # ``from . import x`` — the imported names are the modules.
                found.update(alias.name for alias in node.names)
    return found


class QueryBoundaryTests(unittest.TestCase):
    """Read-only commands must not be able to reach an execution backend."""

    def test_query_module_never_imports_backends(self) -> None:
        imports = module_imports(SOURCE / "services" / "queries.py")
        self.assertNotIn("backends", imports)
        self.assertNotIn("backend_factory", imports)

    def test_query_module_never_imports_execution_paths(self) -> None:
        imports = module_imports(SOURCE / "services" / "queries.py")
        for forbidden in ("execution", "scheduled", "commands", "hpc"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, imports)

    def test_contracts_stay_free_of_backends(self) -> None:
        """Shared validation must hold identically for local and scheduled runs."""

        imports = module_imports(SOURCE / "services" / "contracts.py")
        self.assertNotIn("backends", imports)

    def test_query_command_split_is_exhaustive(self) -> None:
        """Every CLI read-only command must resolve into the queries module."""

        from mlipflow import cli
        from mlipflow.services import queries

        for command in sorted(cli.READ_ONLY_COMMANDS):
            attribute = f"query_{'workflow' if command in {'list', 'status', 'json'} else command}"
            with self.subTest(command=command):
                self.assertTrue(hasattr(queries, attribute), attribute)


class BackendConstructionTests(unittest.TestCase):
    """Scheduler backends are constructed in exactly one module."""

    def test_only_the_factory_constructs_scheduler_backends(self) -> None:
        # ``make_run_plan`` builds an ``SshSlurmBackend`` as the default remote
        # *template library* — a read-only reader, not a scheduler — so that one
        # call site is the documented exception.  Every other construction must
        # go through ``backend_factory``.
        allowed = {("commands.py", "SshSlurmBackend")}
        found: set[tuple[str, str]] = set()
        for path in sorted((SOURCE / "services").glob("*.py")):
            if path.name in {"backend_factory.py", "__init__.py"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id == "SshSlurmBackend":
                    found.add((path.name, node.func.id))
        self.assertEqual(allowed, found)


if __name__ == "__main__":
    unittest.main()
