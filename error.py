class SpectraCompressError(Exception):
    def __init__(self,message=str,code=int):
        self.message=message
        self.code=code
        # super().__init__(self.message,self.code)
    
class ValidationError (SpectraCompressError):
    def __init__(self, message=str, code=int):
        super().__init__(message, code)

class BadRequestError(SpectraCompressError):
    def __init__(self, message=str, code=int):
        super().__init__(message, code)

class InputValidation(SpectraCompressError):
    def __init__(self, message=str, code=int):
        super().__init__(message, code)

class SingularMatrixError(SpectraCompressError):
    def __init__(self, message=str, code=int):
        super().__init__(message, code)

class ShapeMismatchError(SpectraCompressError):
    def __init__(self, message=str, code=int):
        super().__init__(message, code)

class NotFittedError(SpectraCompressError):
    def __init__(self, message=str, code=int):
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




