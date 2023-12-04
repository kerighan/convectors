from ..base_layer import Layer
from typing import Any, Callable, List, Optional, Set, Union, Dict


class Annoy(Layer):
    _trainable = True

    def __init__(
        self,
        k: int = 10,
        threshold: float = None,
        n_trees: int = 10,
        metric: str = "euclidean",
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(name, verbose)
        self._k = k
        self._n_trees = n_trees
        self._threshold = threshold
        self._metric = metric

    def fit(self, X, docs=None, y=None):
        from annoy import AnnoyIndex

        self._model = AnnoyIndex(X.shape[1], self._metric)
        self._index_to_doc = {}
        if docs is None:
            docs = range(len(X))
        for i, (row, doc) in enumerate(zip(X, docs)):
            self._model.add_item(i, row)
            self._index_to_doc[i] = doc
        self._model.build(self._n_trees)

        self._trained = True
        return self

    def process_documents(self, X):
        res = []
        for row in X:
            ids, dists = self._model.get_nns_by_vector(
                row, self._k, include_distances=True
            )
            if self._threshold is not None:
                ids = [i for i, d in zip(ids, dists) if d < self._threshold]
                dists = [d for d in dists if d < self._threshold]
            if self._k == 1:
                if len(ids) == 0:
                    res.append(None)
                else:
                    res.append(self._index_to_doc[ids[0]])
            else:
                res.append([self._index_to_doc[i] for i in ids])
        return res


class HNSW(Layer):
    _trainable = True

    def __init__(
        self,
        k: int = 10,
        ef: int = 50,
        ef_construction: int = 200,
        metric: str = "l2",
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(name, verbose)
        self._k = k
        self._ef = ef
        self._ef_construction = ef_construction
        self._metric = metric

    def fit(self, X, docs=None, y=None):
        import hnswlib

        self._model = hnswlib.Index(space=self._metric, dim=X.shape[1])
        self._index_to_doc = {}
        if docs is None:
            docs = range(len(X))
        self._model.add_items(X, range(len(X)))
        for i, doc in enumerate(docs):
            self._index_to_doc[i] = doc
        # for i, (row, doc) in enumerate(zip(X, docs)):
        #     self._model.add_items([row], [i])
        #     self._index_to_doc[i] = doc
        self._model.set_ef(self._ef)
        self._model.set_ef_construction(self._ef_construction)

        self._trained = True
        return self

    def process_documents(self, X):
        res = []
        for row in X:
            ids, dists = self._model.knn_query(row, k=self._k)
            if self._k == 1:
                if len(ids) == 0:
                    res.append(None)
                else:
                    res.append(self._index_to_doc[ids[0]])
            else:
                res.append([self._index_to_doc[i] for i in ids])
        return res


class NNDescent(Layer):
    _trainable = True

    def __init__(
        self,
        k: int = 10,
        metric: str = "euclidean",
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(name, verbose)
        self._k = k
        self._metric = metric

    def fit(self, X, docs=None, y=None):
        from pynndescent import NNDescent

        self._model = NNDescent(X, metric=self._metric)
        self._index_to_doc = {}
        if docs is None:
            docs = range(len(X))
        for i, doc in enumerate(docs):
            self._index_to_doc[i] = doc

        self._trained = True
        return self

    def process_documents(self, X):
        res = []
        for row in X:
            ids, dists = self._model.query(row, k=self._k)
            if self._k == 1:
                if len(ids) == 0:
                    res.append(None)
                else:
                    res.append(self._index_to_doc[ids[0]])
            else:
                res.append([self._index_to_doc[i] for i in ids])
        return res


class KDTree(Layer):
    _trainable = True

    def __init__(
        self,
        k: int = 10,
        threshold: float = None,
        metric: str = "euclidean",
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(name, verbose)
        self._k = k
        self._metric = metric
        self._threshold = threshold

    def fit(self, X, docs=None, y=None):
        from sklearn.neighbors import KDTree

        self._model = KDTree(X, metric=self._metric)
        self._index_to_doc = {}
        if docs is None:
            docs = range(len(X))
        for i, doc in enumerate(docs):
            self._index_to_doc[i] = doc

        self._trained = True
        return self

    def process_documents(self, X):
        distances, ids = self._model.query(X, k=self._k)
        res = []
        for row, dists in zip(ids, distances):
            for i, d in zip(row, dists):
                if self._threshold is not None and d > self._threshold:
                    res.append(None)
                res.append(self._index_to_doc[i])
        return res
