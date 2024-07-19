import numpy as np
import networkx as nx
import pandas as pd
from scipy import sparse
from convectors.layers import Tokenize, TfIdf
from .duplicates import remove_near_duplicates
from convectors.algorithms.summarization import summarize
from cityhash import CityHash64

from louvain_numba import best_partition

# from community import best_partition
from joblib import Parallel, delayed


def process_topic(cm, nodes, G, X, features, data, top_n_docs, n_sentences=10):
    from community import best_partition

    H = G.subgraph(nodes)
    try:
        pr = nx.eigenvector_centrality(H, max_iter=20, tol=1e-4)
    except:
        pr = nx.degree_centrality(H)

    best_docs = sorted(nodes, key=lambda x: pr[x], reverse=True)[:top_n_docs]

    feats = X[nodes].sum(axis=0).A1
    best_feats = np.argsort(feats)[::-1][:10]
    words = [features[i] for i in best_feats]

    topic_text = "\n".join([data[node] for node in best_docs])
    topic_summary = summarize(topic_text, n=n_sentences, boost_words=words)
    topic_hash = str(CityHash64(topic_summary.encode()) % 100000000)

    return {
        "summary": topic_summary,
        "docs": [data[node] for node in best_docs],
        "words": words,
        "topic_id": topic_hash,
        "n_docs": len(nodes),
    }


def tfidf_graph_topics(
    data,
    min_count=4,
    min_df=3,
    max_df=0.25,
    avg_degree=3,
    max_features=5000,
    min_docs=2,
    top_n_docs=10,
    n_sentences=5,
    max_n_topics=20,
    stopwords=["fr", "en", "url", "media"],
    shuffle=False,
    verbose=True,
    n_jobs=-1,
):
    # Tokenization and TF-IDF
    tokenize = Tokenize(stopwords=stopwords)
    processed_data = tokenize(data)

    if shuffle:
        processed_data = [
            np.random.permutation(text).tolist() for text in processed_data
        ]

    n = len(data)
    try:
        tfidf = TfIdf(
            min_df=min_df, max_df=max_df, max_features=max_features, verbose=verbose
        )
        X = tfidf(processed_data)
    except:
        tfidf = TfIdf(min_df=1, max_df=1, max_features=max_features, verbose=verbose)
        X = tfidf(processed_data)
        min_docs = 1

    # Compute similarity matrix
    sim = X @ X.T

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges
    edges = sparse.triu(sim, k=1)
    edges = sparse.coo_matrix(edges)
    top_edges = sorted(
        zip(edges.row, edges.col, edges.data), key=lambda x: x[2], reverse=True
    )
    top_edges = top_edges[: avg_degree * n]
    G.add_weighted_edges_from(top_edges)

    # Community detection
    node_to_cm = best_partition(G, resolution=1.25)

    # Group nodes by community
    cm_to_nodes = {}
    for node, cm in node_to_cm.items():
        cm_to_nodes.setdefault(cm, []).append(node)

    cm_to_nodes = {
        cm: nodes for cm, nodes in cm_to_nodes.items() if len(nodes) >= min_docs
    }
    cm_to_nodes = dict(
        sorted(cm_to_nodes.items(), key=lambda x: len(x[1]), reverse=True)[
            :max_n_topics
        ]
    )

    # Extract features
    features = tfidf._vectorizer.get_feature_names_out()

    # Process topics in parallel
    topics = Parallel(n_jobs=n_jobs)(
        delayed(process_topic)(cm, nodes, G, X, features, data, top_n_docs, n_sentences)
        for cm, nodes in cm_to_nodes.items()
    )

    if verbose:
        print(pd.DataFrame(topics))

    return topics
