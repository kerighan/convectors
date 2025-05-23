import re
from typing import Any, List, Union

import numpy as np
from scipy.sparse import issparse, spmatrix, vstack


def camel_to_snake(name: str) -> str:
    """
    Convert a camelCase string to snake_case.
    
    Parameters
    ----------
    name : str
        The camelCase string to convert.
        
    Returns
    -------
    str
        The converted snake_case string.
        
    Examples
    --------
    >>> camel_to_snake("CamelCase")
    'camel_case'
    >>> camel_to_snake("camelCase")
    'camel_case'
    >>> camel_to_snake("CamelCaseString")
    'camel_case_string'
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def to_matrix(series: Any) -> Union[np.ndarray, spmatrix]:
    """
    Convert a series of vectors to a matrix.
    
    This function handles different types of input series and converts them
    to either a numpy array or a sparse matrix, depending on the input type.
    
    Parameters
    ----------
    series : Any
        The series to convert. Can be a list of numpy arrays, a list of sparse
        matrices, a numpy array, or a sparse matrix.
        
    Returns
    -------
    Union[np.ndarray, spmatrix]
        The converted matrix.
    """
    from scipy.sparse import issparse, vstack
    
    # If already a matrix (sparse or dense), return as is
    if issparse(series) or isinstance(series, np.ndarray):
        return series
    
    # Handle list/series of numpy arrays
    if isinstance(series[0], np.ndarray):
        if isinstance(series, list):
            return np.array(series)
        else:
            return np.array(series.tolist())
    # Handle list/series of sparse matrices
    elif issparse(series[0]):
        return vstack(series.tolist())
    
    return series
