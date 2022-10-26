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


def get_features_from_documents(
    documents,
    max_features=10000,
    min_tf=0,
    max_tf=float("inf"),
    min_df=0,
    max_df=float("inf"),
    index=1,
    unk_token=False,
    mask_token=False,
    empty_token=False
):
    from collections import defaultdict
    tf = defaultdict(int)
    df = defaultdict(int)
    for doc in documents:
        seen = set()
        for token in doc:
            tf[token] += 1
            if token not in seen:
                df[token] += 1
                seen.add(token)

    # restrict by term frequency
    if min_df < 1:
        min_df *= len(documents)
    if max_df < 1:
        max_df *= len(documents)

    n_features = 0
    id2feature = {}
    feature2id = {}
    if unk_token:
        id2feature[index] = "_UNK_"
        feature2id["_UNK_"] = index
        index += 1
        n_features += 1
    if mask_token:
        id2feature[index] = "_MASK_"
        feature2id["_MASK_"] = index
        index += 1
        n_features += 1
    if empty_token:
        id2feature[index] = "_EMPTY_"
        feature2id["_EMPTY_"] = index
        index += 1
        n_features += 1

    for feature, feature_tf in tf.items():
        if feature_tf < min_tf:
            continue
        if feature_tf > max_tf:
            continue
        if min_df > 0 or max_df < float("inf"):
            feature_df = df[feature]
            if feature_df < min_df:
                continue
            if feature_df > max_df:
                continue
        id2feature[index] = feature
        feature2id[feature] = index
        index += 1
        n_features += 1
        if n_features >= max_features:
            break
    return feature2id, id2feature
