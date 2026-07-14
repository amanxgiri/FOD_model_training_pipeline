"""Stable image and video inference interfaces."""


class InferenceError(RuntimeError):
    """Base error for project-controlled inference."""


class InferenceConfigurationError(InferenceError):
    """Raised when inference configuration is missing or invalid."""


class DetectorError(InferenceError):
    """Raised when a model cannot be loaded or returns invalid predictions."""


class VideoInferenceError(InferenceError):
    """Raised when video input, processing, or output fails."""
