from ..base_layer import Layer
from typing import Any, Optional


class ReductionLayer(Layer):
    def fit(self, documents: Any, labels: Any = None) -> None:
        self._reducer.fit(documents)
        self._trained = True

    def process_documents(self, documents: Any) -> Any:
        res = self._reducer.transform(documents)
        return res


class SVD(ReductionLayer):
    def __init__(
        self,
        n_components: int = 2,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(name, verbose)
        from sklearn.decomposition import TruncatedSVD

        self.n_components = n_components
        self._reducer = TruncatedSVD(n_components=n_components, **kwargs)


class PCA(ReductionLayer):
    def __init__(
        self,
        n_components: int = 2,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(name, verbose)
        from sklearn.decomposition import PCA

        self.n_components = n_components
        self._reducer = PCA(n_components=n_components, **kwargs)
