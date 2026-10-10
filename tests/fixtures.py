import numpy as np
from data.data_contracts import DataContract

def generate_synthetic_dataset(n_rows=200, n_cols=20, seed=42):
    """
    Generates a reproducible synthetic dataset for testing.
    Matches the schema requirements of DataContract and SchemaValidator.
    """
    rng = np.random.default_rng(seed)
    
    # Generate data in range [0, 255] as typical for image-like data,
    # but use float32 to avoid overflow during covariance/eigen calculations.
    data = rng.uniform(0, 255, size=(n_rows, n_cols)).astype(np.float32)
    
    contract = DataContract(
        expected_shape=(n_rows, n_cols),
        dtype=np.float32,
        valid_range=(0, 255)
    )
    
    return data, contract
