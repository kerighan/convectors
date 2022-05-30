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


class PairCounter:
    def __init__(self, undirected=True):
        from collections import defaultdict
        self.undir = undirected
        self.pair_counts = defaultdict(int)
        self.pair_total = 0
        self.counts = defaultdict(int)
        self.counts_total = 0

    def count(self, series):
        import itertools
        from collections import Counter
        self.counts = Counter(itertools.chain(*series))
        self.counts_total = sum(self.counts.values())

    def get_pair(self, a, b):
        if self.undir:
            pair = (a, b) if a < b else (b, a)
        else:
            pair = (a, b)
        return pair

    def increment(self, a, b):
        pair = self.get_pair(a, b)
        self.pair_counts[pair] += 1
        self.pair_total += 1

    def get(self, a, b=None):
        if b is not None:
            pair = self.get_pair(a, b)
            return self.pair_counts[pair]
        return self.counts[a]

    def get_pmi(
        self, normalize=False, threshold=0, min_cooc=2, ignore_self=True
    ):
        import numpy as np

        edges = {}
        for (a, b), f_ab in self.pair_counts.items():
            if f_ab == 0 or f_ab < min_cooc or (ignore_self and (a == b)):
                continue
            f_a = self.counts[a] / self.counts_total
            f_b = self.counts[b] / self.counts_total
            f_ab /= self.pair_total
            _pmi = np.log(f_ab / (f_a * f_b))
            if normalize:
                _pmi /= -np.log(f_ab)
                _pmi = min(1, _pmi)

            if threshold is None:
                edges[a, b] = _pmi
            elif _pmi > threshold:
                edges[a, b] = _pmi
        return edges

    def __repr__(self):
        return str(self.pair_counts)
