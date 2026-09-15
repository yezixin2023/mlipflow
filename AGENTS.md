# MLIPFlow Agent Operating Guidelines

This document applies only to the current `mlipflow/` repository. The Agent acts as the supervisor; MLIPFlow and its computational plugins are the deterministic execution layer.

## Non-Negotiable Boundaries

1. All numerical work must be performed through MLIPFlow computational plugins or external scientific programs explicitly specified by the user. The LLM must not estimate, fill in, infer, or fabricate energies, forces, diffusion coefficients, ionic conductivities, voltages, or model rankings.

2. Prefer invoking `mlipflow`. Do not bypass the state store by directly calling `sbatch`, `scancel`, deleting run directories, or overwriting models.

3. The following commands are strictly read-only and may be executed directly:

   ```text
   mlipflow list
   mlipflow status
   mlipflow json
   mlipflow inspect
   mlipflow logs
   mlipflow route
   mlipflow doctor
   ```

   These commands must not submit jobs, fetch results, retry runs, cancel jobs, modify inputs, write state, or trigger automatic workflow advancement.

4. `init/run/advance/retry/stop` modify workflow state or external systems. Before starting an expensive or external computation that is declared to require approval, first use `run --dry-run` to review its execution semantics, and then use `--approve` to indicate explicit approval.

   Scheduler observation, bounded fetch, and check/collect operations do not require a second approval.

   `retry` must create a fresh attempt.

   An explicit `stop NODE` command itself constitutes cancellation intent.

5. The following actions always require separate and explicit user intent:

   - deleting or cleaning data;
   - overwriting models or datasets;
   - destructive reruns;
   - remote staging;
   - DFT or AIMD calculations;
   - long-duration MD;
   - training or fine-tuning;
   - large-scale screening.

   An explicit `stop NODE` may cancel known jobs associated with that node, but cancellation must not be expanded to unrelated jobs.

6. Do not write passwords, private-key paths, tokens, POTCAR files, model weights, or private large datasets into the repository.

   SSH backends must reference only profile names defined in the user's `~/.ssh/config`.

7. `retry` must create a new attempt and preserve all previous results. Never emulate retry behavior using `rm`.

8. A scheduler status of `COMPLETED` does not imply that the scientific result is `OK`.

   A node may be marked `OK` only when both:

   - the capability adapter's completion criteria pass; and
   - the required output schema is valid.

9. Uncertain scientific parameters, units, fitting windows, random seeds, dataset splits, or reference energies must be reported and left unresolved until clarified. Do not guess them.

10. Built-in Python adapters are trusted code, and `run --dry-run` may load them. Do not generate execution plans from untrusted repository code.

    `dft-labeling.label` static calculations are integrated with the generic SSH-SLURM profile/template/workspace contract.

    Other built-in adapters support local execution only.

    The Agent should specify only the scientific task and abstract resource requirements. It must not guess:

    - SSH hosts;
    - partitions;
    - modules;
    - executables;
    - template roots;
    - work roots.

    The Agent must not use `submit_script` or `remote_cwd` to bypass core staging, fetch, or completion-check restrictions.

11. The same physical computing site must reuse its canonical `remote_template_root` and `work_root`.

    Do not create sibling template roots or work roots based on:

    - plugin;
    - framework;
    - target;
    - partition;
    - validation round.

    If a required template family is missing, first inspect and verify the existing canonical template root. Then add the required family or scheduler-specific subdirectory under that canonical root.

    Existing templates must never be silently overwritten.

## Recommended Supervision Workflow

1. Run:

   ```text
   mlipflow json
   ```

   to obtain the persistent workflow state.

2. Use the corresponding repository Skill to understand the task and its scientific considerations.

3. Run:

   ```text
   mlipflow inspect NODE
   ```

   to inspect the plugin contract, inputs, and known limitations.

4. If model selection is involved, run:

   ```text
   mlipflow route --task ...
   ```

   Review:

   - candidate models;
   - exclusion reasons;
   - benchmark metric contributions.

   Do not select a model based on plots, model branding, or reputation alone.

5. For a `run` operation that requires approval, first execute it with:

   ```text
   --dry-run
   ```

   Review and explain, item by item:

   - execution semantics;
   - computational scale;
   - backend;
   - resource requirements;
   - actual inputs;
   - staged scripts.

   Then request explicit boolean approval.

   Ordinary `advance` and `retry` operations, as well as an explicit `stop NODE`, do not require an additional approval field.

6. After execution, determine the outcome using both the process exit code and structured JSON output.

   Never describe a non-zero exit code as a successful execution.

7. For a `FAIL` result, inspect the logs and manifest first.

   Diagnose the failure before proposing a retry or parameter correction.

   Do not retry automatically.

## Replay Mode

Replay mode may only:

- read existing small manifests or artifacts;
- validate them;
- reference them by path.

Replay mode must not:

- start numerical programs;
- submit jobs;
- copy large datasets;
- modify source files.

Replay results must be explicitly described as:

> structured collection of existing results

They must not be described as:

- recomputation;
- independent scientific validation.

## Skills and Plugins

- `mlipflow/plugins/<capability>/skill/` contains each capability's Agent Skill,
  with relative directory symlinks under `.agents/skills/` for discovery.
  The cross-capability `mlip-workflow` Skill remains in
  `.agents/skills/mlip-workflow/`. Skills teach the Agent:
  - when a capability should be used;
  - what inputs are required;
  - how results should be interpreted;
  - when user approval or clarification is required.

  Maintain each specialist Skill beside its capability implementation. Skills do
  not contain the primary numerical implementation.

- `mlipflow/plugins/` defines deterministic computational units, including:
  - dependencies;
  - execution backends;
  - executable operations;
  - completion criteria.

  Core owns generic result-manifest replay and fresh retry attempts.

Do not elevate DeepMD, CHGNet, an individual MSD fitting procedure, SLURM, or OUTCAR parsing into separate top-level Agent Skills.
