"""User-facing entry point for Zhi-Pan Liu LASP stochastic surface walking.

LASP SSW is the random-walk PES sampling backend.  The implementation lives in
``lasp_ssw.py``; this module only gives that existing, reviewed contract an
explicit random-walk name and defaults to the execute path.
"""
from __future__ import annotations

import sys

if __package__:
    from . import lasp_ssw
else:
    import lasp_ssw


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"execute", "normalize-replay"}:
        args.insert(0, "execute")
    return lasp_ssw.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
