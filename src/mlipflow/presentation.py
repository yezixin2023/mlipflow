"""Agent-facing projections and text rendering for CLI results.

Query and planning services return the single detailed source of truth used by
execution and audit workflows. This module never creates an alternative plan
model; it only projects those objects for normal CLI display.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_HASH_KEY_PARTS = ("fingerprint", "digest", "sha256", "hash", "provenance")
_DIGEST_PREFIXES = ("sha256:", "tree-sha256:")
_DIGEST_VALUE = re.compile(r"(?:tree-)?sha256:[A-Za-z0-9._:+/=-]+", re.IGNORECASE)


def compact_output(command: str, data: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Return the compact semantic view for one detailed service payload."""

    if command in {"status", "json"}:
        return _compact_workflow(data)
    if command == "list":
        return _compact_list(data)
    if command == "inspect":
        return _compact_inspect(data)
    if command == "route":
        return _compact_route(data)
    if command == "run" and dry_run:
        return _compact_run_plan(data)
    if dry_run and data.get("action") in {"advance", "retry"}:
        return _compact_action_plan(data)
    if command == "run":
        return _compact_run_result(data)
    if command in {"advance", "retry", "stop"}:
        return _compact_mutation_result(command, data)
    return _without_hashes(data)


def render_text(command: str, data: dict[str, Any], *, audit: bool = False) -> str:
    """Render command output as readable text rather than serialized JSON."""

    if audit:
        return "\n".join([f"{command} (audit)", *_mapping_lines(data)])
    if command in {"status", "json", "list"}:
        return _render_status(data)
    if command == "inspect":
        return _render_inspect(data)
    if command == "route":
        return _render_route(data)
    if command == "run" and data.get("action") == "run-plan":
        return _render_run_plan(data)
    if command == "logs":
        return _render_logs(data)
    if command == "doctor":
        return _render_doctor(data)
    return "\n".join(_mapping_lines(data))


def _compact_workflow(data: dict[str, Any]) -> dict[str, Any]:
    project = data.get("project")
    project_id = project.get("id") if isinstance(project, dict) else project
    nodes: list[dict[str, Any]] = []
    for raw in data.get("steps", []):
        if not isinstance(raw, dict):
            continue
        node = {
            "node_id": raw.get("node_id"),
            "plugin": raw.get("plugin_id"),
            "state": raw.get("state"),
            "attempt": raw.get("attempt"),
            "backend": raw.get("backend"),
        }
        if raw.get("job_id"):
            node["job_id"] = raw["job_id"]
        reason = _concise_reason(raw.get("diagnostic"))
        if reason:
            node["reason"] = reason
        roles = sorted(
            {
                str(item["role"])
                for item in raw.get("artifacts", [])
                if isinstance(item, dict) and item.get("role")
            }
        )
        if roles:
            node["outputs"] = roles
        nodes.append(node)
    return {
        "project_id": project_id,
        "initialized": bool(data.get("initialized")),
        "counts": data.get("counts", {}),
        "nodes": nodes,
    }


def _compact_list(data: dict[str, Any]) -> dict[str, Any]:
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    return {
        "project_id": project.get("id"),
        "initialized": bool(data.get("initialized")),
        "nodes": [
            {
                "node_id": item.get("node_id"),
                "plugin": item.get("plugin_id"),
                "state": item.get("state"),
                "attempt": item.get("attempt"),
            }
            for item in data.get("nodes", [])
            if isinstance(item, dict)
        ],
    }


def _compact_inspect(data: dict[str, Any]) -> dict[str, Any]:
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    plugin_record = data.get("plugin") if isinstance(data.get("plugin"), dict) else {}
    manifest = (
        plugin_record.get("manifest")
        if isinstance(plugin_record.get("manifest"), dict)
        else {}
    )
    parameters = _without_hashes(node.get("parameters", {}))
    result: dict[str, Any] = {
        "node_id": node.get("id"),
        "uses": node.get("uses"),
        "state": state.get("state"),
        "attempt": state.get("attempt"),
        "action": _node_action(node),
        "backend": node.get("backend", "local"),
        "needs": node.get("needs", []),
        "inputs": _without_hashes(node.get("inputs", {})),
        "parameters": parameters,
        "resources": _without_hashes(node.get("resources", {})),
        "plugin": {
            key: value
            for key, value in {
                "id": manifest.get("id"),
                "name": manifest.get("display_name", manifest.get("name")),
                "version": manifest.get("version"),
                "category": manifest.get("category"),
                "implementation": manifest.get("implementation", {}).get("kind")
                if isinstance(manifest.get("implementation"), dict)
                else None,
                "backends": manifest.get("execution", {}).get("backends")
                if isinstance(manifest.get("execution"), dict)
                else None,
                "operations": manifest.get("execution", {}).get("operations")
                if isinstance(manifest.get("execution"), dict)
                else None,
            }.items()
            if value not in (None, [], {})
        },
    }
    selected = _selected_model(node)
    if selected is not None:
        result["selected_model"] = selected
    reason = _concise_reason(state.get("diagnostic"))
    if reason:
        result["reason"] = reason
    return result


def _compact_route(data: dict[str, Any]) -> dict[str, Any]:
    ranking = []
    for raw in data.get("ranking", []):
        if not isinstance(raw, dict):
            continue
        ranking.append(
            {
                "model_id": raw.get("model_id"),
                "score": raw.get("score"),
                "metrics": raw.get("metrics", {}),
            }
        )
    rejected = []
    for raw in data.get("rejected", []):
        if not isinstance(raw, dict):
            continue
        reasons = raw.get("reasons", [])
        reason = "; ".join(str(item) for item in reasons) if isinstance(reasons, list) else str(reasons)
        rejected.append({"model_id": raw.get("model_id"), "reason": reason})
    return {
        "task": data.get("task"),
        "elements": data.get("elements", []),
        "scenario": data.get("scenario"),
        "selected_model": data.get("selected_model"),
        "ranking": ranking,
        "rejected": rejected,
    }


def _compact_run_plan(data: dict[str, Any]) -> dict[str, Any]:
    adapter = data.get("adapter_plan") if isinstance(data.get("adapter_plan"), dict) else {}
    status = adapter.get("status") or "READY"
    result: dict[str, Any] = {
        "action": "run-plan",
        "node_id": data.get("node_id"),
        "plugin": data.get("plugin", {}).get("id")
        if isinstance(data.get("plugin"), dict)
        else None,
        "state": status,
        "mode": data.get("mode"),
        "backend": data.get("backend"),
        "will_run": _will_run(data, adapter),
        "inputs": _without_hashes(data.get("inputs", {})),
        "parameters": _without_hashes(data.get("parameters", {})),
        "resources": _without_hashes(data.get("resources", {})),
        "approval_required": data.get("approval_required") is True,
    }
    if data.get("backend_profile"):
        result["backend_profile"] = data["backend_profile"]
    selected = _selected_model(data)
    if selected is not None:
        result["selected_model"] = selected
    reason = _plan_reason(adapter, data.get("adapter_diagnostics"))
    if reason:
        result["reason"] = reason
    if data.get("approval_required") is True:
        result["approval_token"] = data.get("plan_digest")
    return result


def _compact_action_plan(data: dict[str, Any]) -> dict[str, Any]:
    details = data.get("details") if isinstance(data.get("details"), dict) else {}
    result: dict[str, Any] = {
        "action": data.get("action"),
        "node_id": data.get("node_id"),
    }
    if isinstance(details.get("transitions"), list):
        result["changes"] = [
            {
                key: item.get(key)
                for key in ("node_id", "action", "to", "scheduler_state", "reason")
                if key in item
            }
            for item in details["transitions"]
            if isinstance(item, dict)
        ]
    else:
        result.update(_without_hashes(details))
    return result


def _compact_run_result(data: dict[str, Any]) -> dict[str, Any]:
    step = data.get("step") if isinstance(data.get("step"), dict) else {}
    result_data = data.get("result") if isinstance(data.get("result"), dict) else {}
    result: dict[str, Any] = {
        "action": "run",
        "node_id": step.get("node_id"),
        "state": step.get("state"),
        "attempt": step.get("attempt"),
        "backend": step.get("backend"),
    }
    if step.get("job_id"):
        result["job_id"] = step["job_id"]
    metrics = result_data.get("metrics")
    if isinstance(metrics, dict) and metrics:
        result["metrics"] = _without_hashes(metrics)
    reason = _concise_reason(step.get("diagnostic"))
    if reason:
        result["reason"] = reason
    return result


def _compact_mutation_result(command: str, data: dict[str, Any]) -> dict[str, Any]:
    if command == "advance":
        changed = data.get("changed", [])
        return {
            "action": "advance",
            "changed": [
                {
                    key: item.get(key)
                    for key in ("node_id", "state", "attempt", "backend", "job_id", "diagnostic")
                    if item.get(key) is not None
                }
                for item in changed
                if isinstance(item, dict)
            ],
        }
    step = data.get("step") if isinstance(data.get("step"), dict) else {}
    return {
        key: value
        for key, value in {
            "action": command,
            "node_id": step.get("node_id"),
            "state": step.get("state"),
            "attempt": step.get("attempt"),
            "backend": step.get("backend"),
            "job_id": step.get("job_id"),
        }.items()
        if value is not None
    }


def _will_run(plan: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    if plan.get("mode") == "replay":
        return {"kind": "replay", "summary": "read existing results; no numerical program"}
    scheduled = adapter.get("scheduled_execution")
    if isinstance(scheduled, dict):
        hpc_submissions = plan.get("hpc_executions")
        if isinstance(hpc_submissions, list):
            scripts = sorted(
                {
                    str(path)
                    for item in hpc_submissions
                    if isinstance(item, dict)
                    for execution in [item.get("hpc_execution")]
                    if isinstance(execution, dict)
                    for path in (
                        execution.get("template_paths", {}).values()
                        if isinstance(execution.get("template_paths"), dict)
                        else []
                    )
                }
            )
            return {
                "kind": "scheduler-jobs",
                "submission_strategy": scheduled.get("submission_strategy"),
                "job_count": len(hpc_submissions),
                "resources_per_job": (
                    hpc_submissions[0].get("hpc_execution", {}).get("resources")
                    if hpc_submissions
                    and isinstance(hpc_submissions[0], dict)
                    and isinstance(hpc_submissions[0].get("hpc_execution"), dict)
                    else None
                ),
                "scripts": scripts,
            }
        hpc = plan.get("hpc_execution") if isinstance(plan.get("hpc_execution"), dict) else {}
        scripts = hpc.get("template_paths") if isinstance(hpc.get("template_paths"), dict) else {}
        return {
            "kind": "scheduler-job",
            "execution_model": hpc.get("execution_model", scheduled.get("execution_model")),
            "scripts": sorted(str(value) for value in scripts.values()),
        }
    argv = adapter.get("argv")
    if isinstance(argv, list):
        return {"kind": "local-command", "command": _command_summary(argv)}
    return {"kind": "unavailable", "summary": "execution is blocked or not implemented"}


def _command_summary(argv: list[Any]) -> list[str]:
    result: list[str] = []
    for index, raw in enumerate(argv[:10]):
        value = str(raw)
        if value.startswith(_DIGEST_PREFIXES):
            value = "<content-id>"
        else:
            value = _DIGEST_VALUE.sub("<content-id>", value)
            if Path(value).is_absolute():
                value = Path(value).name or "/"
        result.append(value)
    if len(argv) > 10:
        result.append(f"… (+{len(argv) - 10} args)")
    return result


def _without_hashes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, raw in value.items()
            if not _is_hash_key(str(key))
            and (cleaned := _without_hashes(raw)) is not None
        }
    if isinstance(value, list):
        return [cleaned for raw in value if (cleaned := _without_hashes(raw)) is not None]
    if isinstance(value, str):
        if value.startswith(_DIGEST_PREFIXES):
            return None
        return _DIGEST_VALUE.sub("<content-id>", value)
    return value


def _is_hash_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _HASH_KEY_PARTS)


def _selected_model(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    parameters = value.get("parameters") if isinstance(value.get("parameters"), dict) else {}
    inputs = value.get("inputs") if isinstance(value.get("inputs"), dict) else {}
    for mapping in (parameters, inputs):
        for key in ("model_id", "selected_model", "model"):
            candidate = mapping.get(key)
            if isinstance(candidate, str) and not candidate.endswith((".json", ".yaml", ".yml")):
                return candidate
    return None


def _node_action(node: dict[str, Any]) -> str:
    parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
    operation = parameters.get("operation")
    return str(operation) if operation is not None else str(node.get("mode", "execute"))


def _plan_reason(adapter: dict[str, Any], diagnostics: Any) -> str | None:
    for source in (adapter.get("diagnostics"), diagnostics):
        if not isinstance(source, list):
            continue
        messages = []
        for item in source:
            if isinstance(item, dict):
                message = item.get("message", item.get("detail"))
            else:
                message = item
            if message:
                messages.append(str(message))
        if messages:
            return _concise_reason("; ".join(messages))
    return None


def _concise_reason(value: Any, limit: int = 240) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    line = " ".join(value.split())
    line = _DIGEST_VALUE.sub("<content-id>", line)
    if line == "preview: project has not been initialized":
        return None
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _render_status(data: dict[str, Any]) -> str:
    if "nodes" not in data and "nodes" in data.get("data", {}):
        data = data["data"]
    project = data.get("project_id")
    initialized = "initialized" if data.get("initialized") else "not initialized"
    lines = [f"project {project} — {initialized}"]
    counts = data.get("counts", {})
    if isinstance(counts, dict) and counts:
        lines.append("states: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    nodes = data.get("nodes", [])
    for node in nodes:
        if not isinstance(node, dict):
            continue
        line = f"{node.get('state', 'UNKNOWN'):>9}  {node.get('node_id')}  {node.get('plugin')}"
        if node.get("attempt") is not None:
            line += f"  attempt {node['attempt']}"
        if node.get("job_id"):
            line += f"  job {node['job_id']}"
        lines.append(line)
        if node.get("reason"):
            lines.append(f"           reason: {node['reason']}")
        if node.get("outputs"):
            lines.append(f"           outputs: {', '.join(node['outputs'])}")
    return "\n".join(lines)


def _render_inspect(data: dict[str, Any]) -> str:
    lines = [f"{data.get('node_id')} — {data.get('uses')}"]
    lines.append(f"state: {data.get('state')} (attempt {data.get('attempt')})")
    lines.append(f"action: {data.get('action')}")
    lines.append(f"backend: {data.get('backend')}")
    if data.get("selected_model"):
        lines.append(f"selected model: {data['selected_model']}")
    if data.get("needs"):
        lines.append("needs: " + ", ".join(str(item) for item in data["needs"]))
    _append_section(lines, "inputs", data.get("inputs"))
    _append_section(lines, "parameters", data.get("parameters"))
    _append_section(lines, "resources", data.get("resources"))
    plugin = data.get("plugin")
    if isinstance(plugin, dict):
        summary = " ".join(
            str(value) for value in (plugin.get("id"), plugin.get("version")) if value
        )
        if plugin.get("implementation"):
            summary += f" ({plugin['implementation']})"
        lines.append(f"plugin: {summary}")
        if plugin.get("backends"):
            lines.append("supported backends: " + ", ".join(plugin["backends"]))
    if data.get("reason"):
        lines.append(f"reason: {data['reason']}")
    return "\n".join(lines)


def _render_route(data: dict[str, Any]) -> str:
    elements = ", ".join(str(item) for item in data.get("elements", []))
    lines = [
        f"route {data.get('task')} — {data.get('scenario')} [{elements}]",
        f"selected model: {data.get('selected_model') or 'none'}",
    ]
    ranking = data.get("ranking", [])
    if ranking:
        lines.append("ranking:")
        for index, item in enumerate(ranking, 1):
            metrics = _inline_mapping(item.get("metrics", {}))
            suffix = f"; {metrics}" if metrics else ""
            lines.append(f"  {index}. {item.get('model_id')}  score={item.get('score')}{suffix}")
    rejected = data.get("rejected", [])
    if rejected:
        lines.append("rejected:")
        for item in rejected:
            lines.append(f"  {item.get('model_id')}: {item.get('reason')}")
    return "\n".join(lines)


def _render_run_plan(data: dict[str, Any]) -> str:
    lines = [f"run {data.get('node_id')} — {data.get('state')}"]
    lines.append(f"plugin: {data.get('plugin')}")
    lines.append(f"mode/backend: {data.get('mode')} / {data.get('backend')}")
    if data.get("selected_model"):
        lines.append(f"selected model: {data['selected_model']}")
    will_run = data.get("will_run", {})
    if isinstance(will_run, dict):
        if will_run.get("kind") == "local-command":
            lines.append("will run: " + " ".join(will_run.get("command", [])))
        elif will_run.get("kind") == "scheduler-job":
            model = will_run.get("execution_model") or "scheduled command"
            lines.append(f"will run: scheduler job ({model})")
            if will_run.get("scripts"):
                lines.append("scripts: " + ", ".join(will_run["scripts"]))
        else:
            lines.append(f"will run: {will_run.get('summary', will_run.get('kind'))}")
    _append_section(lines, "inputs", data.get("inputs"))
    _append_section(lines, "parameters", data.get("parameters"))
    _append_section(lines, "resources", data.get("resources"))
    if data.get("reason"):
        lines.append(f"reason: {data['reason']}")
    if data.get("approval_required"):
        lines.append("approval: required")
        lines.append(f"token: {data.get('approval_token')}")
    else:
        lines.append("approval: not required")
    return "\n".join(lines)


def _render_logs(data: dict[str, Any]) -> str:
    lines = [f"logs {data.get('node_id')} — attempt {data.get('attempt')}"]
    for name, record in data.get("logs", {}).items():
        if not isinstance(record, dict):
            continue
        lines.append(f"{name}: {'available' if record.get('exists') else 'missing'}")
        lines.extend(f"  {str(line).rstrip()}" for line in record.get("lines", []))
    return "\n".join(lines)


def _render_doctor(data: dict[str, Any]) -> str:
    lines = ["doctor: " + ("OK" if data.get("ok") else "FAIL")]
    for item in data.get("diagnostics", []):
        if isinstance(item, dict):
            mark = "OK" if item.get("ok") else "FAIL"
            lines.append(f"  {mark} {item.get('check')}: {item.get('detail')}")
    return "\n".join(lines)


def _append_section(lines: list[str], label: str, value: Any) -> None:
    if value in (None, {}, []):
        return
    if isinstance(value, dict):
        lines.append(f"{label}:")
        lines.extend(_mapping_lines(value, indent=2))
        return
    lines.append(f"{label}: {_scalar(value)}")


def _mapping_lines(value: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, raw in value.items():
        if isinstance(raw, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_mapping_lines(raw, indent + 2))
        elif isinstance(raw, list) and any(isinstance(item, (dict, list)) for item in raw):
            lines.append(f"{prefix}{key}:")
            for item in raw:
                if isinstance(item, dict):
                    lines.append(f"{' ' * (indent + 2)}-")
                    lines.extend(_mapping_lines(item, indent + 4))
                else:
                    lines.append(f"{' ' * (indent + 2)}- {_scalar(item)}")
        else:
            lines.append(f"{prefix}{key}: {_scalar(raw)}")
    return lines


def _inline_mapping(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(f"{key}={_scalar(raw)}" for key, raw in value.items())


def _scalar(value: Any) -> str:
    if value is None:
        return "none"
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, list):
        return ", ".join(_scalar(item) for item in value) if value else "none"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
