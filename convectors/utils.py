import pandas as pd
from tqdm.contrib.concurrent import process_map
from tqdm.contrib import tmap


def input_series(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], pd.Series):
            args = list(args)
            args[1] = pd.Series(args[1])
        return func(*args, **kwargs)
    return wrapper


def parallel_apply(series, func, name=None):
    res = process_map(func, series, desc=name)
    return pd.Series(res)


def progress_apply(series, func, name=None):
    res = tmap(func, series, desc=name)
    return pd.Series(res)


pd.Series.parallel_apply = parallel_apply
pd.Series.progress_apply = progress_apply
