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

    def fit(self, series, y=None):
        self.reducer.fit(to_matrix(series))

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


class PCA(ReduceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        n_components=2,
        name=None,
        verbose=True,
        **kwargs,
    ):
        super(PCA, self).__init__(input, output, name, verbose)

        from sklearn.decomposition import PCA as PCA_sk
        self.reducer = PCA_sk(
            n_components=n_components,
            **kwargs)


class NMF(ReduceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        n_components=2,
        name=None,
        verbose=True,
        **kwargs,
    ):
        super(NMF, self).__init__(input, output, name, verbose)

        from sklearn.decomposition import NMF as NMF_sk
        self.reducer = NMF_sk(
            n_components=n_components,
            **kwargs)


class UMAP(ReduceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        n_components=2,
        n_neighbors=10,
        min_dist=0,
        name=None,
        verbose=True,
        **kwargs,
    ):
        super(UMAP, self).__init__(input, output, name, verbose)

        from umap import UMAP as UMAP_
        self.reducer = UMAP_(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            **kwargs)
