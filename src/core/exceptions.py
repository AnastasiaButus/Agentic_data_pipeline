"""Core exception types used by the pipeline infrastructure layer."""


class ConfigError(ValueError):
    """Raised when a configuration file is missing required data or is invalid."""


class ArtifactError(RuntimeError):
    """Raised when a pipeline artifact cannot be created or accessed."""


class ValidationError(ValueError):
    """Raised when input data fails a validation check."""