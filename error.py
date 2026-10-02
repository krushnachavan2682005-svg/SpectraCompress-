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