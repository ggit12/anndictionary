"""
unit tets for anndict.utils.scanpy_
"""

import pytest
import numpy as np
import pandas as pd

from anndict.utils.scanpy_ import (
    subsample_adata_dict,
    resample_adata,
)
