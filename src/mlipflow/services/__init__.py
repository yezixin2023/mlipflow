"""Read and mutation services.

The query and command halves used to live in one 1891-line module separated only
by a comment.  They are now separate modules, and the "queries never reach a
backend" guarantee is a real import boundary: ``services.queries`` does not
import ``mlipflow.backends`` at all, and ``tests/test_module_boundaries.py``
enforces that statically.

Layout::

    paths            project-scoped paths, state/config drift assertions
    queries          read-only: list/status/json/inspect/logs/route/doctor
    contracts        pure validation, fingerprinting, context assembly
    backend_factory  the one place scheduler backends are constructed
    execution        replay / local / scheduled-submit attempt lifecycles
    scheduled        stage, submit, observe, fetch, finalize for schedulers
    commands         init/run/advance/retry/stop approval surface

This package re-exports the previous flat module's names, so
``from mlipflow.services import run_node`` and
``patch("mlipflow.services.LocalBackend.run")`` keep working unchanged.
"""

from __future__ import annotations

# Every import below is a deliberate compatibility re-export of a name that used
# to live in the flat ``services.py``, including the underscore-prefixed helpers
# that tests reach for directly.  ``noqa: F401`` is therefore correct here rather
# than a suppression of a real problem.
#
# Backend classes are re-exported because callers (and tests) address them
# through this namespace; they are patched as class attributes, so the binding
# here and the one used inside the submodules are the same object.
from ..backends import (  # noqa: F401
    LocalBackend,
    SchedulerBackend,
    SlurmBackend,
    SshSlurmBackend,
)

from .backend_factory import (  # noqa: F401
    SchedulerFactory,
    default_scheduler_factory,
    scheduler_for_node,
    scheduler_from_cluster_record,
)
from .commands import (  # noqa: F401
    DEFAULT_PROJECT,
    advance,
    initialize,
    make_advance_plan,
    make_retry_plan,
    make_run_plan,
    make_stop_plan,
    retry,
    run_node,
    stop,
    _require_approval,
)
from .contracts import (  # noqa: F401
    _adapter_command_identities,
    _adapter_context,
    _cluster_profile,
    _is_within,
    _load_result,
    _manifest_context,
    _normalize_adapter_artifacts,
    _planned_attempt,
    _project_scoped_result_path,
    _scheduled_contract,
    _sha256_file,
)
from .execution import _execute_ready, _replay  # noqa: F401
from .paths import (  # noqa: F401
    STATE_RELATIVE,
    attempt_directory,
    state_path,
    _assert_state_matches_project,
)
from .queries import (  # noqa: F401
    query_doctor,
    query_inspect,
    query_logs,
    query_route,
    query_workflow,
    _tail_lines,
)
from .scheduled import (  # noqa: F401
    _finalize_scheduled_adapter,
    _finalize_scheduler_manifest,
    _load_pinned_scheduled_plan,
    _materialize_hpc_scripts,
    _observe_scheduled_step,
    _remote_output_inventory,
    _scheduler_expected_identity,
    _stage_and_submit_scheduled_adapter,
    _validate_hpc_completion,
    _HPC_RUN_SCRIPT,
    _HPC_SUBMIT_SCRIPT,
)


__all__ = [
    # queries
    "query_doctor",
    "query_inspect",
    "query_logs",
    "query_route",
    "query_workflow",
    # commands
    "advance",
    "initialize",
    "make_advance_plan",
    "make_retry_plan",
    "make_run_plan",
    "make_stop_plan",
    "retry",
    "run_node",
    "stop",
    # paths
    "attempt_directory",
    "state_path",
    "STATE_RELATIVE",
    # backends and their construction
    "LocalBackend",
    "SchedulerBackend",
    "SchedulerFactory",
    "SlurmBackend",
    "SshSlurmBackend",
    "default_scheduler_factory",
    "scheduler_for_node",
    "scheduler_from_cluster_record",
    "DEFAULT_PROJECT",
]
