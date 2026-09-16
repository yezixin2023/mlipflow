# Contributing to MLIPipe

## Development setup

Use a Python 3.12+ development environment. From the repository root, install the
development and local scientific dependencies, then run the tests and lint checks:

```bash
python -m pip install -e ".[dev,local]"
python -m pytest
python -m ruff check mlipipe tests examples
```

## Contribution rules

- Keep changes focused. Leave unrelated cleanup and redesign for a separate change.
- Explain what changed and why, so a reviewer can understand the intended behavior.
- For scientific changes, state the relevant units, seeds, reference values,
  tolerances or conventions when applicable.
- Add or update regression tests for changed behavior.
- Do not commit credentials, private cluster configuration, POTCAR, proprietary
  software, unredistributable datasets or model weights.

## Public interface

The supported user interfaces are the `mlipipe` CLI, documented configuration formats, CLI/result formats, and Agent Skills. Internal Python modules are not a stable public SDK.

## License

Contributions submitted for inclusion in MLIPipe are licensed under [Apache-2.0](../LICENSE).
