# Contributing to MLIPFlow

Thanks for helping improve MLIPFlow. Contributions are welcome across the workflow core, built-in scientific capabilities, tests, documentation, examples, Agent Skills, and HPC integration patterns.

MLIPFlow sits between workflow orchestration and scientific software, so a good contribution should make both its **software behavior** and its **scientific assumptions** reviewable.

## Development setup

```bash
git clone https://github.com/yezixin2023/mlipflow.git
cd mlipflow

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,local]"

python -m pytest
ruff check .
```

You can install a smaller set of extras if your change does not touch scientific adapters:

```bash
python -m pip install -e ".[dev]"
```

## What to contribute

Useful contribution areas include:

- workflow configuration, minimal execution state, planning, and CLI behavior;
- new or improved built-in scientific capabilities;
- scientific completion checks and small reference fixtures;
- local or scheduler integration tests;
- Agent Skills that supervise existing deterministic capabilities;
- site-template examples for common HPC layouts;
- examples and documentation that make real workflows easier to adopt;
- reproducibility, packaging, security, and repository-quality improvements.

For larger scientific or architectural changes, opening an issue or draft pull request early can make the intended contract easier to review before implementation grows.

## Keep changes reviewable

Prefer focused changes with a clear motivation. A pull request should explain:

1. what behavior changes;
2. why the change is needed;
3. which user or developer workflow it affects;
4. how it was validated;
5. whether it changes a configuration schema, capability operation, scientific convention, artifact format, or compatibility boundary.

If a change affects scientific results, include the relevant units, normalization conventions, seeds, reference values, tolerances, software/model versions, and source provenance. Do not treat "the program ran" as sufficient scientific validation.

## Core behavior to preserve

Some behaviors are part of the public workflow contract:

- `list`, `status`, `json`, `inspect`, `logs`, `route`, and `doctor` are read-only commands.
- Approval-gated execution is reviewed with `run --dry-run` and confirmed explicitly with `--approve`.
- Retries create fresh attempts instead of overwriting prior attempts.
- Scheduler completion is followed by scientific checking before a run is treated as scientifically successful.
- Model routing is evidence- and policy-driven rather than hard-coded to a preferred model family.
- Project configuration contains scientific intent and abstract resources; private cluster details belong in user-local site profiles and site-owned templates.

Changes to these contracts are possible, but they should be explicit, documented, and covered by regression tests.

## Adding or changing a built-in scientific capability

Scientific implementations live under:

```text
mlipflow/plugins/<capability_id>/
  adapter.py
```

MLIPFlow does not discover third-party plugins. Add the capability ID, adapter entrypoint, supported backends, operations, and approval requirement to the literal built-in mapping in `mlipflow/plugins/__init__.py`. Use the capability's standard `adapter.py` entrypoint and ordinary package imports. Keep capability modules flat. Keep that entry small; scientific validation and completion logic belong in the adapter and its scientific runner. Core owns generic replay and fresh-attempt retry behavior.

Adapter execution should use explicit argv lists and controlled working directories/environment. When an external program is involved, a zero process exit code is not by itself a scientific completion criterion.

## Adding or changing an Agent Skill

Each specialist Agent Skill lives beside its capability implementation:

```text
mlipflow/plugins/<underscored_id>/
  adapter.py
  skill/
    SKILL.md
    agents/openai.yaml
```

Read the selected Skill's `SKILL.md` and its relevant linked references. The
[capability guide](README.md#scientific-capabilities-and-agent-skills) links directly
to these files.

Keep a relative directory symlink at `.agents/skills/<skill-name>` pointing to
`../../mlipflow/plugins/<underscored_id>/skill` so repository agents discover the
same source files. The cross-capability `mlip-workflow` Skill stays in
`.agents/skills/mlip-workflow/`; it has no single owning capability.

Preserve the Skill's declared name and `agents/openai.yaml` metadata when moving
it. In particular, `mlip-active-learning` belongs to the `active_learning` package.
Update `pyproject.toml` data-file sources to the canonical Skill paths. Package
data keeps specialist Skills beside installed code, and the existing
`share/mlipflow/agent-skills/<skill-name>/` exports remain available. Include new
resource types in the package-data and source-distribution rules when needed.

Skills describe how an agent should supervise a capability: what evidence to request, which MLIPFlow operation to call, when approval is required, and how to interpret results. They should not duplicate scientific computation that belongs in adapters or external tools.

When the repository's Skill validator is available, validate changed Skills before submitting them.

## Tests and fixtures

Tests should be deterministic, offline by default, and bounded in runtime/resource use.

Good fixtures are small enough to review and redistribute, and clearly identify whether they are synthetic, derived, or based on real evidence. External programs, real schedulers, large datasets, model weights, and licensed inputs should use explicit opt-in integration paths rather than becoming requirements for the default test suite.

Useful focused commands include:

```bash
python -m pytest tests/test_builtin_capabilities.py
python -m pytest tests/test_config_and_plans.py
python -m pytest tests/test_readonly_cli.py
python -m pytest tests/test_hpc_architecture.py
```

Run the broader suite when your environment supports the dependencies needed by the affected area.

## Third-party code, models, and data

When adding or wrapping third-party material, document its source, version, license, modifications, redistribution status, and required citations. Do not commit credentials, private keys, private cluster configuration, proprietary executables, VASP POTCAR data, unredistributable datasets, or model weights without explicit redistribution rights.

MLIPFlow's Apache-2.0 license does not replace the terms of software, models, or data used through its capabilities. See [`NOTICE`](NOTICE).

## AI-assisted contributions

AI-assisted code or documentation is welcome when the contributor reviews and validates the result. The submitting contributor remains responsible for correctness, licensing, provenance, tests, and scientific claims.

## Documentation changes

The top-level README is the user entry point. Keep it concise, capability-oriented, and aligned with implemented interfaces. Detailed validation boundaries belong in the relevant validation documents rather than being repeated throughout user-facing examples.

When behavior changes, update the closest source of truth as needed:

- schemas for configuration contracts;
- `mlipflow/plugins/__init__.py` for the built-in capability lookup;
- `mlipflow/plugins/<underscored_id>/skill/SKILL.md` for specialist Agent Skill contracts;
- `.agents/skills/mlip-workflow/SKILL.md` for cross-capability supervision;
- the closest adapter and deterministic tests for implementation state;
- `CHANGELOG.md` for notable user-facing changes.

## License

Unless explicitly stated otherwise, contributions submitted for inclusion in MLIPFlow are licensed under the repository's [Apache License 2.0](LICENSE).
