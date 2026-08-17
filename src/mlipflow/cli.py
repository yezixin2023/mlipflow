"""MLIPFlow command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import sysconfig
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import load_project
from .errors import ApprovalError, MLIPFlowError
from .presentation import compact_output, render_text
from .services import (
    advance,
    initialize,
    make_advance_plan,
    make_retry_plan,
    make_run_plan,
    query_doctor,
    query_inspect,
    query_logs,
    query_route,
    query_workflow,
    retry,
    run_node,
    stop,
)


READ_ONLY_COMMANDS = frozenset({"list", "status", "json", "inspect", "logs", "route", "doctor"})
MUTATING_COMMANDS = frozenset({"init", "run", "advance", "retry", "stop"})


def default_plugin_root() -> Path:
    source_root = Path(__file__).resolve().parents[2] / "plugins"
    if source_root.is_dir():
        return source_root
    return Path(sysconfig.get_path("data")) / "share" / "mlipflow" / "plugins"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlipflow",
        description="Deterministic, agent-supervised MLIP workflows",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="project directory or project.yaml (no implicit upward search)",
    )
    parser.add_argument(
        "--plugins",
        type=Path,
        default=default_plugin_root(),
        help="plugin manifest root",
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=None,
        help="local site.yaml (defaults to ~/.mlipflow/site.yaml only when HPC resolution needs it)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="include detailed provenance and internal planning information",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create/initialize a project (writes)")
    init_parser.add_argument("path", nargs="?", type=Path)

    subparsers.add_parser("list", help="list workflow nodes (strictly read-only)")
    status_parser = subparsers.add_parser("status", help="show workflow status (strictly read-only)")
    status_parser.add_argument("node", nargs="?")
    json_parser = subparsers.add_parser("json", help="emit status JSON (strictly read-only)")
    json_parser.add_argument("node", nargs="?")
    inspect_parser = subparsers.add_parser("inspect", help="inspect a node and plugin (read-only)")
    inspect_parser.add_argument("node")
    logs_parser = subparsers.add_parser("logs", help="read saved logs (read-only)")
    logs_parser.add_argument("node")
    logs_parser.add_argument("--tail", type=int, default=80)
    route_parser = subparsers.add_parser("route", help="rank models from benchmark evidence (read-only)")
    route_parser.add_argument("--task", required=True)
    route_parser.add_argument("--elements", nargs="+", required=True)
    route_parser.add_argument("--scenario", required=True)
    subparsers.add_parser("doctor", help="run non-mutating diagnostics")

    run_parser = subparsers.add_parser("run", help="plan or execute one READY node")
    run_parser.add_argument("node")
    _approval_arguments(run_parser)
    advance_parser = subparsers.add_parser(
        "advance", help="observe and reconcile workflow state; never launches nodes"
    )
    advance_parser.add_argument(
        "--dry-run", action="store_true", help="return observations and planned transitions"
    )
    retry_parser = subparsers.add_parser("retry", help="create a new attempt without overwriting")
    retry_parser.add_argument("node")
    retry_parser.add_argument(
        "--dry-run", action="store_true", help="describe the fresh retry attempt without creating it"
    )
    stop_parser = subparsers.add_parser("stop", help="explicitly cancel/stop a node")
    stop_parser.add_argument("node")
    for command_parser in subparsers.choices.values():
        command_parser.add_argument(
            "--audit",
            action="store_true",
            default=argparse.SUPPRESS,
            help="include detailed provenance and internal planning information",
        )
    return parser


def _approval_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="return a plan and write nothing")
    group.add_argument("--approve", metavar="TOKEN", help="approve the exact dry-run execution")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_format = "json" if args.command == "json" else args.format
    try:
        raw_data = dispatch(args)
        audit = bool(getattr(args, "audit", False))
        data = (
            raw_data
            if audit
            else compact_output(
                args.command,
                raw_data,
                dry_run=bool(getattr(args, "dry_run", False)),
            )
        )
        envelope = {
            "schema_version": 1,
            "ok": True,
            "command": args.command,
            "read_only": args.command in READ_ONLY_COMMANDS,
            "audit": audit,
            "data": data,
        }
        _emit(envelope, output_format, error=False)
        if args.command == "doctor" and not raw_data.get("ok", False):
            return 1
        return 0
    except (MLIPFlowError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, MLIPFlowError) else "IO_OR_VALUE_ERROR"
        envelope = {
            "schema_version": 1,
            "ok": False,
            "command": args.command,
            "read_only": args.command in READ_ONLY_COMMANDS,
            "error": {"code": code, "message": str(exc)},
        }
        _emit(envelope, output_format, error=True)
        return 2


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    command = args.command
    if command == "init":
        return initialize(args.path or args.project)
    if command == "doctor":
        return query_doctor(args.project, args.plugins, args.site)
    project = load_project(args.project)
    if command == "list":
        data = query_workflow(project)
        return {
            "project": data["project"],
            "initialized": data["initialized"],
            "nodes": [
                {
                    "node_id": step["node_id"],
                    "plugin_id": step["plugin_id"],
                    "state": step["state"],
                    "attempt": step["attempt"],
                }
                for step in data["steps"]
            ],
        }
    if command in {"status", "json"}:
        return query_workflow(project, args.node)
    if command == "inspect":
        return query_inspect(project, args.node, args.plugins)
    if command == "logs":
        return query_logs(project, args.node, args.tail)
    if command == "route":
        return query_route(
            project, task=args.task, elements=args.elements, scenario=args.scenario
        )
    if command == "run":
        plan = make_run_plan(project, args.node, args.plugins, args.site)
        if args.dry_run:
            return plan
        if plan.get("approval_required") is True and not args.approve:
            raise ApprovalError(
                "this run launches approval-required work; use --dry-run and --approve TOKEN"
            )
        return run_node(project, args.node, args.plugins, args.approve, args.site)
    if command == "advance":
        if args.dry_run:
            return make_advance_plan(project, args.plugins)
        return advance(project, args.plugins)
    if command == "retry":
        if args.dry_run:
            return make_retry_plan(project, args.node, args.plugins)
        return retry(project, args.node, args.plugins)
    if command == "stop":
        return stop(project, args.node, args.site)
    raise MLIPFlowError(f"unsupported command: {command}")


def _emit(envelope: dict[str, Any], output_format: str, error: bool) -> None:
    stream = sys.stderr if error else sys.stdout
    if output_format == "json":
        print(json.dumps(envelope, sort_keys=True, ensure_ascii=False), file=stream)
        return
    if error:
        item = envelope["error"]
        print(f"ERROR [{item['code']}]: {item['message']}", file=stream)
        return
    print(
        render_text(
            str(envelope["command"]),
            envelope["data"],
            audit=envelope.get("audit") is True,
        ),
        file=stream,
    )


if __name__ == "__main__":
    raise SystemExit(main())
