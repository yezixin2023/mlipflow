# Security Policy

## Reporting a vulnerability

Report security vulnerabilities privately. If GitHub private vulnerability reporting
is enabled, use **Security → Advisories → Report a vulnerability**; otherwise use an
existing private channel to contact the maintainers.

Command or path injection, credential leakage, unexpected writes, and bypasses of
SSH or scheduler boundaries are security issues. Public reports must not contain
credentials or private cluster information.

## Execution boundary and support

MLIPipe does not sandbox external scientific software, scripts or model files.
Security fixes target the current supported codebase.
