import pandas as pd

from . import Layer, to_matrix


class KNNGraph(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        k=10,
        algorithm="hnswlib",
        metric="l2",
        name=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.k = k
        self.metric = metric
        self.algorithm = algorithm

    def process_series(self, series):
        X = to_matrix(series)

        if self.algorithm == "hnswlib":
            G = self.get_hnswlib_graph(X)
        return G

    def get_hnswlib_graph(self, X):
        import hnswlib
        import networkx as nx
        import numpy as np

        # possible options are l2, cosine or ip
        index = hnswlib.Index(space='l2', dim=X.shape[1])
        index.init_index(max_elements=X.shape[0], ef_construction=200, M=16)
        index.add_items(X, np.arange(len(X)))
        index.set_ef(50)

        labels, _ = index.knn_query(X, k=self.k)
        edges = []
        for i, row in enumerate(labels):
            for j in row:
                edges.append((i, j))
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_edges_from(edges)
        return G


class AdaptiveGraph(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        k=10,
        ratio=.8,
        reject=.2,
        n_samples=100,
        algorithm="hnswlib",
        metric="l2",
        name=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.ratio = ratio
        self.reject = reject
        self.n_samples = n_samples

    def process_series(self, series):
        X = to_matrix(series)

        if self.algorithm == "hnswlib":
            G = self.get_hnswlib_graph(X)
        return G

    def get_hnswlib_graph(self, X):
        import hnswlib
        import networkx as nx
        import numpy as np

        n, dim = X.shape

        # possible options are l2, cosine or ip
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=16)
        index.add_items(X, np.arange(n))
        index.set_ef(50)

        labels, distances = index.knn_query(X, k=self.k)

        # compute rejection threshold
        x_sample = np.random.randint(0, n, size=self.n_samples)
        y_sample = np.random.randint(0, self.k, size=self.n_samples)
        samples = np.sort(distances[x_sample, y_sample])
        rejection_threshold = samples[int(round(self.n_samples * self.reject))]

        # build graph
        edges = []
        for i, (row, dists) in enumerate(zip(labels, distances)):
            threshold = self.ratio * np.mean(dists)
            for j, d in zip(row, dists):
                if d < threshold or d < rejection_threshold:
                    continue
                edges.append((i, j))
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_edges_from(edges)
        return G


class Community(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        resolution=1,
        algorithm="louvain",
        name=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.resolution = resolution
        self.algorithm = algorithm

    def process_series(self, G):
        if self.algorithm == "louvain":
            try:
                from cylouvain import best_partition
            except ImportError:
                from community import best_partition
            labels = best_partition(G, resolution=self.resolution)
        labels = relabel(labels)
        return pd.Series(labels[i] for i in range(len(labels)))


def relabel(labels):
    from collections import Counter
    count = Counter(labels.values())
    old2new = {}
    for i, (old_label, _) in enumerate(count.most_common()):
        old2new[old_label] = i
    labels = {node: old2new[label] for node, label in labels.items()}
    return labels
