import pandas as pd
from tqdm.contrib.concurrent import process_map


def input_series(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], pd.Series):
            args = list(args)
            args[1] = pd.Series(args[1])
        return func(*args, **kwargs)
    return wrapper


def parallel_apply(series, func, **kwargs):
    if "name" in kwargs:
        res = process_map(func, series, desc=kwargs["name"])
    else:
        res = process_map(func, series)
    return pd.Series(res)


pd.Series.parallel_apply = parallel_apply
