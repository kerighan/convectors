import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm.contrib import tmap
from tqdm.contrib.concurrent import process_map


def identity(x):
    return x


def input_series(func):
    def wrapper(*args, **kwargs):
        return_string = False
        if not isinstance(args[1], pd.Series):
            if isinstance(args[1], str):
                return_string = True
            elif str(type(args[1])) != (
                    "<class 'networkx.classes.graph.Graph'>"):
                if issparse(args[1]) or isinstance(args[1], np.ndarray):
                    pass
                else:
                    args = list(args)
                    try:
                        args[1] = pd.Series(args[1])
                    except ValueError:
                        args[1] = pd.Series(list(args[1]))
        res = func(*args, **kwargs)
        if return_string:
            return res[0]
        return res
    return wrapper


def parallel_apply(series, func, name=None, verbose=True):
    res = process_map(func, series, desc=name)
    return pd.Series(res, index=series.index)


def progress_apply(series, func, name=None, verbose=True):
    if verbose:
        res = tmap(func, series, desc=name)
    else:
        res = [func(x) for x in series]
    return pd.Series(res, index=series.index)


pd.Series.parallel_apply = parallel_apply
pd.Series.progress_apply = progress_apply
