import numpy as np


def tfidf_graph_topics(
    data,
    min_count=4,
    min_df=3,
    max_df=0.25,
    # minimum_tfidf=0.1,
    avg_degree=5,
    max_features=5000,
    min_docs=4,
    top_n_docs=10,
    stopwords=["fr", "en", "url", "media"],
    shuffle=False,
    verbose=True,
):
    from convectors.layers import Tokenize, TfIdf, SnowballStem
    from .duplicates import remove_near_duplicates
    from convectors.algorithms.summarization import summarize
    import networkx as nx
    import matplotlib.pyplot as plt
    from cityhash import CityHash64

    # from cdlib import algorithms
    from louvain_numba import best_partition
    import pandas as pd
    import random

    tokenize = Tokenize(stopwords=stopwords)
    # tokenize += SnowballStem(lang="fr")
    processed_data = tokenize(data)
    if shuffle:
        processed_data = [random.sample(text, len(text)) for text in processed_data]

    tfidf = TfIdf(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        verbose=verbose,
    )
    X = tfidf(processed_data)
    sim = X @ X.T

    features = tfidf._vectorizer.get_feature_names_out()
    # threshold
    # sim[sim < minimum_tfidf] = 0
    # print(sim)

    node_to_cm = best_partition(sim)

    G = nx.Graph()
    G.add_nodes_from(range(len(node_to_cm)))
    edges = []
    for i in range(len(node_to_cm)):
        for j in range(i + 1, len(node_to_cm)):
            w = sim[i, j]
            # if w > minimum_tfidf:
            edges.append((i, j, w))
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    edges = edges[: avg_degree * len(node_to_cm)]
    G.add_weighted_edges_from(edges)

    cm_to_nodes = {}
    for node, cm in node_to_cm.items():
        if cm not in cm_to_nodes:
            cm_to_nodes[cm] = []
        cm_to_nodes[cm].append(node)
    cm_to_nodes = {
        cm: nodes for cm, nodes in cm_to_nodes.items() if len(nodes) >= min_docs
    }

    # reorder by size
    cm_to_nodes = {
        cm: nodes
        for cm, nodes in sorted(
            cm_to_nodes.items(), key=lambda x: len(x[1]), reverse=True
        )
    }

    topics = []
    for cm, nodes in cm_to_nodes.items():
        H = nx.subgraph(G, nodes)
        try:
            pr = nx.eigenvector_centrality(H)
        except:
            pr = nx.degree_centrality(H)

        best_docs = sorted(nodes, key=lambda x: pr[x], reverse=True)

        feats = np.array(X[nodes].sum(axis=0)).flatten()
        best_feats = np.argsort(feats)[::-1]
        words = [features[it] for it in best_feats[:10]]
        topic_text = "\n".join([data[node] for node in best_docs[:top_n_docs]])
        topic_summary = summarize(topic_text, n=10, boost_words=words)
        topic_hash = str(CityHash64(topic_summary.encode()) % 100000000)
        topics.append(
            {
                "summary": topic_summary,
                "docs": [data[node] for node in best_docs[:top_n_docs]],
                "words": words,
                "topic_id": topic_hash,
                "n_docs": len(nodes),
            }
        )
    import pandas as pd

    print(pd.DataFrame(topics))
    return topics
