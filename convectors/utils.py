import pandas as pd
from tqdm.contrib import tmap
from tqdm.contrib.concurrent import process_map


def identity(x):
    return x


def input_series(func):
    def wrapper(*args, **kwargs):
        if args[0].parallel and not isinstance(args[1], pd.Series):
            args = list(args)
            args[1] = pd.Series(args[1])
        return func(*args, **kwargs)
    return wrapper


def parallel_apply(series, func, name=None):
    res = process_map(func, series, desc=name)
    return pd.Series(res, index=series.index)


def progress_apply(series, func, name=None):
    res = tmap(func, series, desc=name)
    return pd.Series(res, index=series.index)


pd.Series.parallel_apply = parallel_apply
pd.Series.progress_apply = progress_apply
