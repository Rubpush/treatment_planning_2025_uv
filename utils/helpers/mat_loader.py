from typing import Any
import scipy.io
import numpy as np

def _mat_to_dict(obj: Any) -> dict:
    """Recursively construct python dictionary from matlab struct."""
    if isinstance(obj, scipy.io.matlab.mat_struct):
        dest = {}
        for field in obj._fieldnames:
            value = obj.__dict__[field]
            dest[field] = _mat_to_dict(value)
        return dest
    
    if (
        isinstance(obj, np.ndarray)
        and len(obj) > 0
        and isinstance(obj[0], scipy.io.matlab.mat_struct)
    ):
        return np.array([_mat_to_dict(elem) for elem in obj])

    return obj
    
def loadmat(filename: str) -> dict:
    """Load a MATLAB .mat file and convert it to a Python dictionary."""
    try:
        data = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise ValueError(f"Failed to load MATLAB file '{filename}': {e}")
    
    result = {}
    # Filter out MATLAB metadata and convert structures
    for key in data:
        if not key.startswith('__'):
            result[key] = _mat_to_dict(data[key])

    return result
