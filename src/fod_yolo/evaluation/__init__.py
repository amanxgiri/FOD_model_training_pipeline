"""Framework and project-controlled model evaluation."""


class EvaluationError(RuntimeError):
    """Base class for project-controlled evaluation failures."""


class EvaluationConfigurationError(EvaluationError):
    """Raised when evaluation configuration or CLI intent is invalid."""


class EvaluationDataError(EvaluationError):
    """Raised when ground truth or model predictions violate the contract."""


class EvaluationExecutionError(EvaluationError):
    """Raised when framework evaluation or output finalization fails."""
