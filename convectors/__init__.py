import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm

from .utils import input_series


class Model:
    def __init__(self, layers=[], name="Model", verbose=True):
        self.name = name
        self.verbose = verbose
        self.layers = []
        self.layer2num = {}
        for layer in layers:
            self.add(layer)

    def copy(self):
        nlp = Model()
        for layer in self.layers:
            nlp.add(layer)
        return nlp

    def unload(self):
        for layer in self.layers:
            layer.unload()

    def reload(self, **kwargs):
        for layer in self.layers:
            layer.reload(**kwargs)

    def add(self, layer):
        layer.verbose = False
        layer_name = layer.name

        if layer_name in self.layer2num:
            self.layer2num[layer_name] += 1
            layer_name += f"_{self.layer2num[layer_name]}"
        else:
            self.layer2num[layer_name] = 1
        layer.name = layer_name
        self.layers.append(layer)

    def remove(self, layer_name):
        for i, layer in enumerate(self.layers):
            if layer.name == layer_name:
                self.layers.pop(i)
                del self.layer2num[layer_name]
                return

        raise KeyError(f"layer named {layer} not found")

    def apply(self, series, *args, y=None):
        if self.verbose:
            t = tqdm(self.layers)
            for layer in t:
                t.set_description(layer.name.ljust(12))
                t.refresh()
                series = layer.apply(series, *args, y=y)
        else:
            for layer in self.layers:
                series = layer.apply(series, *args, y=y)
        return series

    def process(self, df, *args, y=None):
        for layer in self.layers:
            layer.process(df, *args, y=y)

    def fit(self, df, *args, y=None):
        for layer in self.layers:
            layer.trained = False
        self.__call__(df, *args, y=y)

    def __getitem__(self, key):
        for layer in self.layers:
            if layer.name == key:
                return layer
        raise KeyError(f"Layer {key} not found")

    def __call__(self, df, *args, y=None):
        if not isinstance(df, pd.DataFrame):
            return self.apply(df, *args, y=y)
        else:
            self.process(df, *args, y=y)

    def __add__(self, model_2):
        name = f"{self.name}+{model_2.name}"
        model = Model(verbose=self.verbose, name=name)
        for layer in self.layers:
            model.add(layer)

        if isinstance(model_2, Model):
            for layer in model_2.layers:
                model.add(layer)
        else:
            model.add(model_2)
        return model

    def __iadd__(self, layer):
        self.add(layer)
        return self

    def __repr__(self):
        length = 40
        t = "\n" + "+" + "-" * length + "+\n"
        u = f"| {self.name}"
        u += " " * (length + 1 - len(u)) + "|"
        t += u + "\n" + "+" + "-" * length + "+"
        for layer in self.layers:
            u = "\n| " + layer.__repr__()
            u += " " * (length + 2 - len(u)) + "|"
            t += u
        t += "\n" + "+" + "-" * length + "+\n"
        return t

    def summary(self):
        print(self.__repr__())

    def save(self, filename):
        import dill
        self.unload()
        with open(filename, "wb") as f:
            dill.dump(self, f)


class Layer:
    document_wise = True
    multi = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        parallel=False
    ):
        self.input = input
        self.output = output
        self.verbose = verbose
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.trained = False
        self.run_parallel = parallel

    def get_numpy_matrix(self, series):
        # get data
        from scipy.sparse import issparse
        X = to_matrix(series)
        if issparse(X):
            X = np.array(X.todense())
        return X

    def unload(self):
        pass

    def reload(self, **_):
        pass

    def process(self, df, y=None):
        if self.input is None:
            raise KeyError(f"{self.name}: no input defined")
        elif self.output is None:
            self.output = self.input

        res = self.apply(df[self.input], y=y)
        if isinstance(res, np.ndarray) or issparse(res):
            df[self.output] = list(res)
        else:
            df[self.output] = res

    @input_series
    def apply(self, series, *args, y=None):
        if self.multi:
            return self.process_series(series, *args)
        else:
            # if layer must be trained
            if self.trainable and not self.trained:
                if self.verbose:
                    iterable = tqdm(range(1), desc=f"{self.name} (fitting)")
                else:
                    iterable = range(1)
                for _ in iterable:
                    self.fit(series, y=y)
                    self.trained = True
            # differing apply procedure depending on layer's logic
            if self.document_wise:
                if self.parallel and self.run_parallel:
                    return series.parallel_apply(self.process_doc,
                                                 name=self.name,
                                                 verbose=self.verbose)
                else:
                    return series.progress_apply(self.process_doc,
                                                 name=self.name,
                                                 verbose=self.verbose)
            else:
                if self.verbose:
                    iterable = tqdm(range(1), desc=self.name)
                else:
                    iterable = range(1)
                for _ in iterable:
                    res = self.process_series(series)
                return res

    def __call__(self, df, *args, y=None):
        if not isinstance(df, pd.DataFrame):
            return self.apply(df, *args, y=y)
        else:
            self.process(df, *args, y=y)

    def __iadd__(self, obj):
        model = Model(verbose=self.verbose)
        model.add(self)
        model.add(obj)
        return model

    def __repr__(self):
        if self.input is None and self.output is None:
            return f"{self.name}"
        elif self.input is not None and self.output is not None:
            return f"{self.name}: {self.input} -> {self.output}"
        elif self.output is None:
            return f"{self.name}: {self.input} -> {self.input}"
        return f"{self.name}"

    def save(self, filename):
        import dill
        self.unload()
        with open(filename, "wb") as f:
            dill.dump(self, f)


class WordVectors:
    def __init__(
        self,
        model=None
    ):
        if model is not None:
            from tqdm import tqdm
            self.id2word = {}
            self.word2id = {}
            weights = [np.zeros_like(model.wv[0])]
            for i, word in tqdm(
                    enumerate(model.wv.key_to_index, 1),
                    total=len(model.wv)):
                vector = model.wv[word]
                weights.append(vector)
                self.word2id[word] = i
                self.id2word[i] = word
            self.weights = np.vstack(weights)

    def unload(self):
        pass

    def reload(self, **_):
        pass

    def normalize(self, norm="l2"):
        if norm == "l1":
            self.weights /= np.sum(self.weights, axis=1)[:, None]
        elif norm == "l2":
            self.weights /= (
                np.linalg.norm(self.weights, axis=1)[:, None].clip(1e-6, None)
            )

    def save(self, filename):
        import dill
        with open(filename, "wb") as f:
            dill.dump(self, f)

    def __iter__(self):
        for key in self.word2id:
            yield key

    def __len__(self):
        return self.num_words

    def __contains__(self, key):
        return key in self.word2id

    def __getitem__(self, key):
        if isinstance(key, int):
            if 0 <= key < self.num_words:
                return self.weights[key, :]
            else:
                raise IndexError(key)
        if key not in self.word2id:
            raise KeyError(key)
        return self.weights[self.word2id[key]]


# =============================================================================
# Functions
# =============================================================================

def load_model(filename, **kwargs):
    import dill
    with open(filename, "rb") as f:
        obj = dill.load(f)
    obj.reload(**kwargs)
    return obj


def to_matrix(series):
    from scipy.sparse import issparse, vstack
    if issparse(series) or isinstance(series, np.ndarray):
        pass
    else:
        if isinstance(series[0], np.ndarray):
            series = np.array(series.tolist())
        elif issparse(series[0]):
            series = vstack(series.tolist())
    return series
