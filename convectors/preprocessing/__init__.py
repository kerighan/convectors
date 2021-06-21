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
