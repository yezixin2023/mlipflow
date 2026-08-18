# Security Policy

MLIPFlow coordinates local programs, scientific adapters, SSH connections, schedulers, model files, and research artifacts. Security reports are therefore especially useful when they identify a way to cross an execution, approval, path, credential, or trust boundary.

## Supported versions

MLIPFlow is currently an alpha project. Security fixes are targeted at the latest package version and the current `main` branch. Older versions do not currently have a long-term support commitment.

## Reporting a vulnerability

Please report exploitable vulnerabilities privately.

If GitHub private vulnerability reporting is enabled for this repository, use **Security → Advisories → Report a vulnerability**. If that option is not available, contact the repository maintainers through a private channel available to collaborators or project users. Do not publish exploitable details, credentials, private cluster information, or proof-of-concept payloads in a public issue.

A useful report includes:

- affected version or commit;
- a minimal reproduction;
- expected and observed behavior;
- impact and required privileges;
- whether the issue affects local, Slurm, or SSH + Slurm execution;
- any known mitigation or workaround.

Use fictional hostnames and revoked/test credentials in reproductions.

## Security-relevant issues

Examples include:

- a documented read-only command performing writes, launching a process, accessing the network, or mutating scheduler state;
- bypassing or replaying an execution approval against a different plan;
- command, argument, path, manifest, SSH-profile, or template injection;
- path traversal or artifact handling that can overwrite files outside the intended project/attempt workspace;
- leakage of passwords, tokens, private keys, credentials, or sensitive cluster data into logs or manifests;
- unsafe plugin discovery or configuration loading that executes code unexpectedly;
- scheduler staging/fetch behavior that accepts the wrong project, node, attempt, or content identity;
- unsafe loading of model or serialized data that creates an unexpected code-execution path in MLIPFlow-controlled behavior.

A disagreement about numerical accuracy is normally a scientific bug rather than a security vulnerability. If a scientific-data path also enables code execution, destructive writes, credential exposure, or another security impact, report it privately as a security issue.

## Trust model

MLIPFlow separates workflow control from scientific execution, but it does not sandbox arbitrary third-party Python adapters or scientific software.

Treat the following as trusted executable inputs:

- Python adapters and plugin code;
- site-owned scheduler templates;
- external wrappers and scientific executables;
- model formats that may deserialize executable objects;
- scripts referenced by your research workflow.

Review third-party plugins before running `run --dry-run`: planning a selected adapter imports and executes its Python planning code even though the dry-run is intended to avoid scientific execution or scheduler submission.

## Operational guidance

- Keep SSH authentication in your normal SSH/agent configuration and user-local site configuration; do not place secrets in `project.yaml`.
- Review dry-run plans before approving expensive work, including commands, inputs, backend, resources, remote paths, and staged files.
- Use least-privilege cluster accounts and filesystem permissions.
- Keep site-owned templates and scientific environments under normal change control.
- Review logs and manifests before publishing them.
- Only run projects, plugins, models, wrappers, and serialized artifacts from sources you trust.

## Current safeguards

The current codebase includes controls such as explicit argv execution for local external programs, bounded project/attempt artifact paths, content identities for staged/fetched artifacts, plan-bound approvals for approval-gated execution, immutable attempt lineage, and scientific completion checks after scheduler completion.

These controls reduce orchestration risk; they do not constitute a security audit of VASP, LAMMPS, LASP, MLIP frameworks, model files, site scripts, SSH, Slurm, or other external components used through MLIPFlow.
