# Run a small local task

This example executes the bundled candidate-ranking program and checks its result.
The scores are dimensionless demonstration inputs, not material properties.
It needs only MLIPFlow's base installation (Python and PyYAML), no model or cluster.

From a checkout, use a fresh work directory:

```bash
cp -R examples/local_ranking /tmp/mlipflow-ranking-demo
mlipflow --project /tmp/mlipflow-ranking-demo init
mlipflow --project /tmp/mlipflow-ranking-demo --format json run rank --dry-run
mlipflow --project /tmp/mlipflow-ranking-demo --format json run rank
mlipflow --project /tmp/mlipflow-ranking-demo json rank
```

For a pip installation, locate the installed examples first:

```bash
python -c 'import sysconfig; print(sysconfig.get_path("data") + "/share/mlipflow/examples")'
```

Copy `local_ranking` from that directory in the first command. Choose a fresh
destination if `/tmp/mlipflow-ranking-demo` already exists.

Expected run output: exit code `0`, `ok: true`, `data.state: OK`,
`data.metrics.selected_count: 2`, and `data.summary.ranked_candidates` containing
`b` (0.8), then `c` (0.5). `missing` appears under `excluded_missing`.
These values come from the declared input rule, not a saved replay result.

`data.artifacts` contains role `ranking-result` and its absolute `path`:
`/tmp/mlipflow-ranking-demo/.mlipflow/runs/rank/attempt-1/ranking-result.json`.
Use the returned path for a downstream task. `manifest_path`, `logs`, `check`, and
`collection` explain how the result was obtained. On failure, read `reason` and
`mlipflow --project /tmp/mlipflow-ranking-demo logs rank` before deciding on a retry.
