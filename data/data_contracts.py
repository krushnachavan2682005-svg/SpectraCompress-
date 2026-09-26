from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class DataContract:
    expected_shape: Tuple[int,int]
    dtype:np.dtype
    valid_range:Tuple[int,int]
    target_column:None=None
    
# हे का?
# expected_shape → (N, 784)
# dtype → expected NumPy dtype, उदा. np.uint8 / np.float32
# valid_range → (0, 255) सारखी minimum आणि maximum value
# target_column → None, कारण unsupervised आहे
# validation method नाही → कारण हा file फक्त contract define करतो.
# valid range input mhanunn yenar ahe mhaneje eg-(0,255) the minimum range is 0 ani maximum 255 