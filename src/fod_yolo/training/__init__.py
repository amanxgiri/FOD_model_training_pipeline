"""Model training, run metadata, and resume support."""


class TrainingError(RuntimeError):
    """Base class for project-controlled training failures."""


class TrainingConfigurationError(TrainingError):
    """Raised when the Phase 1 training contract is invalid."""


class TrainingResumeError(TrainingError):
    """Raised when a checkpoint cannot safely resume its original run."""


class TrainingExecutionError(TrainingError):
    """Raised when model execution or candidate checkpoint handling fails."""
