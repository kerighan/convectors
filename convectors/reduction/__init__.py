from .. import Layer, to_matrix


class ReduceLayer(Layer):
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
        super(ReduceLayer, self).__init__(input, output, name, verbose, False)

    def fit(self, series):
        self.reducer.fit(to_matrix(series))
        self.trained = True

    def process_series(self, series):
        return self.reducer.transform(to_matrix(series))


class SVD(ReduceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        n_components=2,
        name=None,
        verbose=True,
        **kwargs,
    ):
        super(SVD, self).__init__(input, output, name, verbose)

        from sklearn.decomposition import TruncatedSVD
        self.reducer = TruncatedSVD(
            n_components=n_components,
            **kwargs)
