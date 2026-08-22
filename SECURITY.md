# Security Policy

MLIPFlow coordinates local programs, scientific adapters, SSH connections, schedulers, model files, and research artifacts. Security reports are therefore especially useful when they identify a way to cross an execution, approval, path, credential, or trust boundary.

## Supported versions

Security fixes target the latest published package version and the current `main` branch. The project does not currently maintain long-term-support branches for older releases.

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

Use fictional hostnames and revoked or test credentials in reproductions.

## Security-relevant issues

Examples include:

- a documented read-only command performing writes, launching a process, accessing the network, or mutating scheduler state;
- bypassing the explicit approval requirement for external or expensive work;
- command, argument, path, manifest, SSH-profile, or template injection;
- path traversal or artifact handling that can overwrite files outside the intended project or attempt workspace;
- leakage of passwords, tokens, private keys, credentials, or sensitive cluster data into logs or manifests;
- configuration loading that executes code unexpectedly;
- scheduler staging or fetch behavior that accepts the wrong project, node, attempt, or output path;
- unsafe loading of model or serialized data that creates an unexpected code-execution path in MLIPFlow-controlled behavior.

A disagreement about numerical accuracy is normally a scientific bug rather than a security vulnerability. If a scientific-data path also enables code execution, destructive writes, credential exposure, or another security impact, report it privately as a security issue.

## Trust model

MLIPFlow separates workflow control from scientific execution, but it does not sandbox its built-in Python adapters or scientific software.

Treat the following as trusted executable inputs:

- bundled Python adapters;
- site-owned scheduler templates;
- external wrappers and scientific executables;
- model formats that may deserialize executable objects;
- scripts referenced by your research workflow.

Planning a selected built-in capability imports and executes its Python adapter even though the dry run is intended to avoid scientific execution or scheduler submission.

## Operational guidance

- Keep SSH authentication in your normal SSH or agent configuration and user-local site configuration; do not place secrets in `project.yaml`.
- Review dry-run plans before approving expensive work, including commands, inputs, backend, resources, remote paths, and staged files.
- Use least-privilege cluster accounts and filesystem permissions.
- Keep site-owned templates and scientific environments under normal change control.
- Review logs and manifests before publishing them.
- Only run projects, models, wrappers, and serialized artifacts from sources you trust.

## Current safeguards

The current codebase includes controls such as explicit argv execution for local external programs, project and attempt path containment where execution constructs paths, explicit approval for approval-gated execution, fresh attempt directories, and scientific completion checks after scheduler completion.

These controls reduce orchestration risk; they do not constitute a security audit of VASP, LAMMPS, LASP, MLIP frameworks, model files, site scripts, SSH, Slurm, or other external components used through MLIPFlow.
