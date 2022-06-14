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


def adamic_adar_rewire(G, k=2, avg_degree=10):
    import networkx as nx
    A = nx.adjacency_matrix(G)
    degree = 1/np.log(np.array(A.sum(axis=1) + 1)).flatten()
    # degree[np.isnan(degree)] = 0

    id2node = dict(enumerate(G.nodes))
    node2id = {node: i for i, node in id2node.items()}
    neighbors = {}
    for i, node in enumerate(G.nodes):
        neighbors[i] = set(G.neighbors(node))

    # reach farther
    A **= k
    # list edges
    edges = []
    indptr = A.indptr
    indices = A.indices
    n = len(indptr) - 1
    for i in range(n):
        neighbors_i = neighbors[i]

        start, end = indices[i:i+2]
        for index in range(start, end):
            j = indices[index]
            neighbors_j = neighbors[j]

            adamic_adar = 0
            for common_neighbor in neighbors_i.intersection(neighbors_j):
                neighbor_id = node2id[common_neighbor]
                adamic_adar += degree[neighbor_id]
            edges.append((i, j, adamic_adar))
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:avg_degree*n]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(edges)
    return G
