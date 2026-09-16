# Small SQS input example

This idealized two-site binary prototype illustrates the input contract; it is not a
validated alloy geometry or evidence of SQS convergence. Its lattice/cutoff units are
Å, and the manifest declares exact integer occupations and a bounded search of 100
steps. Replace the prototype, counts and search controls for your scientific task.
Use the existing environment with `mlipipe[science]` (ASE, icet and NumPy).

Copy this directory to a fresh work project, then run:

```bash
mlipipe --project /PATH/TO/sqs-project init
mlipipe --project /PATH/TO/sqs-project --format json run generate --dry-run
mlipipe --project /PATH/TO/sqs-project --format json run generate
mlipipe --project /PATH/TO/sqs-project json generate
```

Read `state`, `check`, and `artifacts` for the actual generated structures and
manifest. Scientific composition and count checks must pass before the candidate
is reused. The seed records the deterministic search input, not global optimality.
