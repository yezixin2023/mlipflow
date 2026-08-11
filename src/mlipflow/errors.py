"""Domain exceptions with stable machine-readable error codes."""


class MLIPFlowError(Exception):
    code = "MLIPFLOW_ERROR"


class ConfigError(MLIPFlowError):
    code = "CONFIG_ERROR"


class StateError(MLIPFlowError):
    code = "STATE_ERROR"


class ApprovalError(MLIPFlowError):
    code = "APPROVAL_REQUIRED"


class PluginError(MLIPFlowError):
    code = "PLUGIN_ERROR"


class BackendError(MLIPFlowError):
    code = "BACKEND_ERROR"

