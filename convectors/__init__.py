import pandas as pd
import numpy as np
from .utils import input_series
from scipy.sparse import issparse
from tqdm import tqdm


class Model:
    def __init__(self, name="Model", verbose=True):
        self.name = name
        self.verbose = verbose
        self.layers = []
        self.layer2num = {}

    def add(self, layer):
        layer.verbose = self.verbose
        layer_name = layer.name

        if layer_name in self.layer2num:
            self.layer2num[layer_name] += 1
            layer_name += f"_{self.layer2num[layer_name]}"
        else:
            self.layer2num[layer_name] = 1
        layer.name = layer_name
        self.layers.append(layer)

    def apply(self, series):
        for layer in self.layers:
            series = layer.apply(series)
        return series

    def process(self, df):
        for layer in self.layers:
            layer.process(df)

    def fit(self, df):
        for layer in self.layers:
            layer.trained = False
        self.__call__(df)

    def __call__(self, df):
        if not isinstance(df, pd.DataFrame):
            return self.apply(df)
        else:
            self.process(df)

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
        t += "\n" + "+" + "-" * length + "+"
        return t

    def save(self, filename):
        import dill

        with open(filename, "wb") as f:
            dill.dump(self, f)


class Layer:
    document_wise = True

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

    def process(self, df):
        if self.input is None:
            raise KeyError(f"{self.name}: no input defined")
        elif self.output is None:
            self.output = self.input

        res = self.apply(df[self.input])
        if isinstance(res, np.ndarray) or issparse(res):
            df[self.output] = list(res)
        else:
            df[self.output] = res

    @input_series
    def apply(self, series):
        if self.trainable and not self.trained:
            for _ in tqdm(range(1), desc=f"Fitting {self.name}"):
                self.fit(series)
        if self.document_wise:
            if self.parallel and self.run_parallel:
                return series.parallel_apply(self.process_doc, name=self.name)
            else:
                return series.progress_apply(self.process_doc, name=self.name)
        else:
            for _ in tqdm(range(1), desc=self.name):
                res = self.process_series(series)
            return res

    def __call__(self, df):
        if not isinstance(df, pd.DataFrame):
            return self.apply(df)
        else:
            self.process(df)

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
        with open(filename, "wb") as f:
            dill.dump(self, f)


# =============================================================================
# Functions
# =============================================================================

def load_model(filename):
    import dill
    with open(filename, "rb") as f:
        obj = dill.load(f)
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
