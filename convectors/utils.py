import pandas as pd
from tqdm.contrib import tmap
from tqdm.contrib.concurrent import process_map


def identity(x):
    return x


def input_series(func):
    def wrapper(*args, **kwargs):
        return_string = False
        if args[0].parallel and not isinstance(args[1], pd.Series):
            if isinstance(args[1], str):
                return_string = True

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
