class SpectraCompressError(Exception):
    def __init__(self, message: str = "", code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self):
        return self.message
    
class ValidationError(SpectraCompressError):
    def __init__(self, message: str = "", code: int = 400):
        super().__init__(message, code)

class BadRequestError(SpectraCompressError):
    def __init__(self, message: str = "", code: int = 400):
        super().__init__(message, code)

class InputValidation(SpectraCompressError):
    def __init__(self, message: str = "", code: int = 400):
        super().__init__(message, code)

class SingularMatrixError(SpectraCompressError):
    def __init__(self, message: str = "", code: int = 400):
        super().__init__(message, code)

class ShapeMismatchError(SpectraCompressError, ValueError):
    def __init__(self, message: str = "Shape mismatch", code: int = 400):
        ValueError.__init__(self, message)
        SpectraCompressError.__init__(self, message=message, code=code)

class NotFittedError(SpectraCompressError):
    def __init__(self, message: str = "", code: int = 400):
        super().__init__(message, code)

class NonConvergenceError(InputValidation):
    pass

class MemoryGuardError(SpectraCompressError, ValueError):
    def __init__(self, message: str = "Memory limit exceeded", code: int = 400):
        ValueError.__init__(self, message)
        SpectraCompressError.__init__(self, message=message, code=code)

class InvalidFoldConfigError(ValueError, InputValidation):
    """Raised when cross-validation or fold configuration parameters are invalid."""
    def __init__(self, message: str = "Invalid fold configuration", code: int = 400):
        ValueError.__init__(self, message)
        InputValidation.__init__(self, message=message, code=code)

class ArtifactIntegrityError(SpectraCompressError, ValueError):
    """Raised when artifact checksum, schema, shape, or dtype verification fails."""
    def __init__(self, message: str = "Artifact integrity verification failed", code: int = 400):
        ValueError.__init__(self, message)
        SpectraCompressError.__init__(self, message=message, code=code)

class ArtifactVersionError(SpectraCompressError, ValueError):
    """Raised when artifact library or schema version is incompatible under strict mode."""
    def __init__(self, message: str = "Artifact version mismatch", code: int = 400):
        ValueError.__init__(self, message)
        SpectraCompressError.__init__(self, message=message, code=code)
