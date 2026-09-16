"""Domain exceptions with stable machine-readable error codes."""


class MLIPipeError(Exception):
    code = "MLIPIPE_ERROR"


class ConfigError(MLIPipeError):
    code = "CONFIG_ERROR"


class StateError(MLIPipeError):
    code = "STATE_ERROR"


class ApprovalError(MLIPipeError):
    code = "APPROVAL_REQUIRED"


class CapabilityError(MLIPipeError):
    code = "CAPABILITY_ERROR"


class BackendError(MLIPipeError):
    code = "BACKEND_ERROR"
