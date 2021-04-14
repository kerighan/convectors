import pandas as pd
from .utils import input_series


class Sequential:
    def __init__(self, name=None, verbose=True):
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

    def __call__(self, df):
        if not isinstance(df, pd.DataFrame):
            return self.apply(df)
        else:
            self.process(df)

    def __iadd__(self, layer):
        self.add(layer)
        return self

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
        df[self.output] = self.apply(df[self.input])

    @input_series
    def apply(self, series):
        if self.trainable and not self.trained:
            self.fit(series)
        if self.document_wise:
            if self.parallel and self.run_parallel:
                return series.parallel_apply(self.process_doc, name=self.name)
            else:
                return series.progress_apply(self.process_doc, name=self.name)
        else:
            return self.process_series(series)

    def __call__(self, df):
        if not isinstance(df, pd.DataFrame):
            return self.apply(df)
        else:
            self.process(df)

    def __iadd__(self, obj):
        model = Sequential(verbose=self.verbose)
        model.add(self)
        model.add(obj)
        return model

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
