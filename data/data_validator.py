from data.data_contracts import DataContract
from error import InputValidation
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationReport:
    is_valid: bool
    total_rows: int
    duplicate_rows_count: int
    message: str

class SchemaValidator:
    @staticmethod
    def validate_schema(data: np.ndarray, contract: DataContract):
        if data.shape != contract.expected_shape:
            raise InputValidation("Shape Mismatch With the Contract", 400)
        if data.dtype != contract.dtype:
            raise InputValidation("Data Type Mismatch", 400)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise InputValidation("Data Consist the Nan", 400)
        unique_values=np.unique(data,axis=0)
        duplicate_rows_count=len(data)-len(unique_values)
        max_value=np.max(data)
        min_value=np.min(data)
        contract_min=contract.valid_range[0]
        contract_max=contract.valid_range[1]
        if max_value>contract_max or min_value<contract_min:
            raise InputValidation("Value is Out of the Valid Range",400)
        return ValidationReport(
            is_valid=True,
            total_rows=len(data),
            duplicate_rows_count=duplicate_rows_count,
            message="Data successfully validated against contract."
        )