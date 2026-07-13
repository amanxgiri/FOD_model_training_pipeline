"""Dataset acquisition, conversion, validation, and statistics."""


class DatasetError(RuntimeError):
    """Base exception for dataset pipeline failures."""


class DatasetDownloadError(DatasetError):
    """Raised when Kaggle download or archive extraction fails."""


class KaggleAuthenticationError(DatasetDownloadError):
    """Raised when the required Kaggle credential contract is not satisfied."""


class DatasetDiscoveryError(DatasetError):
    """Raised when a Pascal VOC root or source image is ambiguous or missing."""


class VocParseError(DatasetError):
    """Raised when a Pascal VOC XML annotation is malformed or unsafe."""


class DatasetSplitError(DatasetError):
    """Raised when deterministic split construction cannot be completed."""


class DatasetConversionError(DatasetError):
    """Raised when a source annotation or image cannot be converted safely."""


class DatasetValidationError(DatasetError):
    """Raised when strict processed-dataset validation fails."""
