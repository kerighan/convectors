import re
import numpy as np


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def to_matrix(series):
    from scipy.sparse import issparse, vstack
    if issparse(series) or isinstance(series, np.ndarray):
        pass
    else:
        if isinstance(series[0], np.ndarray):
            if isinstance(series, list):
                series = np.array(series)
            else:
                series = np.array(series.tolist())
        elif issparse(series[0]):
            series = vstack(series.tolist())
    return series
