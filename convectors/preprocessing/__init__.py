from .. import Layer, to_matrix


class Normalize(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        norm="l2",
        name=None,
        verbose=True,
    ):
        super(Normalize, self).__init__(input, output, name, verbose, False)
        self.norm = norm

    def process_series(self, series):
        from sklearn.preprocessing import normalize
        return normalize(to_matrix(series), norm=self.norm)


class Scaler(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
    ):
        super(Scaler, self).__init__(input, output, name, verbose, False)

    def fit(self, series, y=None):
        X = to_matrix(series)
        self.scaler.fit(X)

    def process_series(self, series):
        X = to_matrix(series)
        return self.scaler.transform(X)


class RobustScaler(Scaler):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
    ):
        super(RobustScaler, self).__init__(input, output, name, verbose)
        from sklearn.preprocessing import RobustScaler as RS
        self.scaler = RS()


class MinMaxScaler(Scaler):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
    ):
        super(MinMaxScaler, self).__init__(input, output, name, verbose)
        from sklearn.preprocessing import MinMaxScaler as MM
        self.scaler = MM()


class QuantileTransformer(Scaler):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
    ):
        super(QuantileTransformer, self).__init__(
            input, output, name, verbose)
        from sklearn.preprocessing import QuantileTransformer as QT
        self.scaler = QT()
