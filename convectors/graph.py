import numpy as np
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
        import networkx as nx

        labels, _ = get_hnswlib_nns(X, self.k)

        edges = []
        for i, row in enumerate(labels):
            for j in row:
                if i == j:
                    continue
                edges.append((i, j))
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_edges_from(edges)
        avg_degree = len(G.edges)/len(G.nodes)
        print(f"avg_degree={avg_degree:.2f}")
        return G


class DensityGraph(Layer):
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
        centrifugal=False,
        name=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.centrifugal = centrifugal

    def process_series(self, series):
        X = to_matrix(series)

        if self.algorithm == "hnswlib":
            G = self.get_hnswlib_graph(X)
        return G

    def get_hnswlib_graph(self, X):
        import networkx as nx
        import numpy as np

        n = X.shape[0]
        labels, distances = get_hnswlib_nns(X, self.k)

        edges = []
        density = np.zeros(n)
        for i, (row, dists) in enumerate(zip(labels, distances)):
            density[i] = -np.sum(dists)
            gap = np.argmax(dists[2:] - dists[:-2])
            for j in row[:gap + 2]:
                if i == j:
                    continue
                edges.append((i, j))
        if self.centrifugal:
            edges = [(i, j) for i, j in edges if density[i] >= density[j]]
        else:
            edges = [(i, j) for i, j in edges if density[i] <= density[j]]

        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_edges_from(edges)
        # if self.verbose:
        avg_degree = len(G.edges)/len(G.nodes)
        print(f"avg_degree={avg_degree:.2f}")
        return G


class AdaptiveDensityGraph(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        k=10,
        ratio=1,
        n_samples=100,
        algorithm="hnswlib",
        centrifugal=False,
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
        self.n_samples = n_samples
        self.centrifugal = centrifugal

    def process_series(self, series):
        X = to_matrix(series)

        if self.algorithm == "hnswlib":
            G = self.get_hnswlib_graph(X)
        return G

    def get_hnswlib_graph(self, X):
        import networkx as nx
        import numpy as np

        n = X.shape[0]
        labels, distances = get_hnswlib_nns(X, self.k)

        # build graph
        edges = []
        density = np.zeros(n)
        for i, (row, dists) in enumerate(zip(labels, distances)):
            threshold = self.ratio * np.mean(dists)
            density[i] = -np.sum(dists)
            for j, d in zip(row, dists):
                if i == j:
                    continue
                if d > threshold:
                    break
                edges.append((i, j))
        if self.centrifugal:
            edges = [(i, j) for i, j in edges if density[i] >= density[j]]
        else:
            edges = [(i, j) for i, j in edges if density[i] <= density[j]]

        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_edges_from(edges)
        avg_degree = len(G.edges)/len(G.nodes)
        print(f"avg_degree={avg_degree:.2f}")
        return G


class ExpGraph(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        k=100,
        algorithm="hnswlib",
        min_sim=.001,
        max_sim=.99,
        threshold=.05,
        metric="l2",
        centrifugal=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.min_sim = min_sim
        self.max_sim = max_sim
        self.threshold = threshold
        self.centrifugal = centrifugal

    def process_series(self, series):
        X = to_matrix(series)
        G = self.build_graph(X)
        return G

    def build_graph(self, X):
        import networkx as nx

        labels, distances = get_hnswlib_nns(X, self.k)
        scale = 1 / (1 + np.max(distances))

        density = np.zeros(X.shape[0])
        candidates = []
        for i, (nns, dists) in enumerate(zip(labels, distances)):
            dists *= scale
            sims, b = self.get_tanh_similarities(dists)
            density[i] = b
            for j, similarity in zip(nns, sims):
                if i == j:
                    continue
                candidates.append((i, j, similarity))

        if self.centrifugal:
            def compare(i, j):
                return i > j
        else:
            def compare(i, j):
                return j > i

        edges = []
        for i, j, w in candidates:
            density_i = density[i]
            density_j = density[j]
            weight = w * (2 * density_j) / (density_i + density_j)
            if weight < self.threshold:
                continue

            if compare(density_i, density_j):
                continue
            edges.append((i, j, weight))

        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_weighted_edges_from(edges)

        # if self.verbose:
        avg_degree = len(G.edges)/len(G.nodes)
        print(f"avg_degree={avg_degree:.2f}")
        return G

    def get_exp_similarities(self, row):
        d_k = row[-1]
        d_k = max(d_k, 1e-6)
        d_1 = row[1]
        d_1 = max(d_1, 1e-6)
        log_m = np.log(self.min_sim)
        log_M = np.log(self.max_sim)
        loglog_mM = np.log(log_m / log_M)

        b = loglog_mM / np.log(d_k / d_1)
        a = -log_m/(d_k**b)

        similarities = np.exp(-a*row**b)
        return similarities, b

    def get_tanh_similarities(self, row):
        d_k = row[-1]
        d_1 = row[1]
        if d_k == d_1:
            return row**0

        d_k = max(d_k, 1e-12)
        d_1 = max(d_1, 1e-12)
        numerator = np.log(np.arctanh(1 - self.max_sim) /
                           np.arctanh(1 - self.min_sim))
        denominator = np.log(d_1 / d_k)

        b = numerator / denominator
        if d_k**b == 0:
            return row**0

        a = np.arctanh(1 - self.min_sim) / (d_k**b)
        similarities = 1 - np.tanh(a*row**b)
        return similarities, b


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
        relabel_communities=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.resolution = resolution
        self.algorithm = algorithm
        self.relabel_communities = relabel_communities

    def process_series(self, G):
        if self.algorithm == "louvain":
            try:
                from cylouvain import best_partition
            except ImportError:
                from community import best_partition, modularity
            labels = best_partition(G, resolution=self.resolution)
            # if self.verbose:
            Q = modularity(labels, G)
            print(f"Q={Q:.2f}")
        elif self.algorithm == "infomap":
            from nodl.community import infomap
            labels = infomap(G)
        elif self.algorithm == "lpa":
            from nodl.community import RWLPA, label_propagation

            labels = RWLPA(G, k=2)
            # labels = label_propagation(G)

        if self.relabel_communities:
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


def get_hnswlib_nns(X, k):
    import hnswlib
    import numpy as np

    n, dim = X.shape

    # possible options are l2, cosine or ip
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.add_items(X, np.arange(n))
    index.set_ef(50)

    labels, distances = index.knn_query(X, k=k)
    return labels, distances
