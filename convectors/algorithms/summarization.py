import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import networkx as nx
from convectors.layers import Tokenize, TfIdf, SnowballStem
from .duplicates import remove_near_duplicates


def summarize(
    text,
    n=5,
    itemize=True,
    boost_words=None,
    boost_value=2,
    similarity_threshold=0.05,
    remove_duplicates=False,
):
    # Tokenization
    sentence_tokenize = Tokenize(
        sentence_tokenize=True,
        word_tokenize=False,
        strip_accents=False,
        strip_punctuation=False,
        lower=False,
        split_lines=True,
    )
    sentences = sentence_tokenize(text)

    # Pair adjacent sentences
    sentences = [
        f"{sentences[i]} {sentences[i+1]}" for i in range(0, len(sentences) - 1, 2)
    ]

    # Vectorization
    vectorizer = Tokenize(stopwords=["fr", "en", "media", "url"])
    vectorizer += SnowballStem()
    vectorizer += TfIdf(max_features=2000, min_df=1)

    X = vectorizer(sentences)

    # Boost words if provided
    if boost_words:
        boost_words_ids = vectorizer(" ".join(boost_words)).indices
        X[:, boost_words_ids] *= boost_value

        # Normalize
        X = normalize(X, norm="l2", axis=1, copy=False)

    # Compute similarity matrix
    A = X @ X.T

    # Create graph and compute centrality
    G = nx.from_scipy_sparse_array(A)
    # remove edges where weight < threshold
    for u, v, d in list(G.edges(data=True)):
        if d["weight"] < 0.12:
            G.remove_edge(u, v)
    pr = nx.pagerank(G)

    # Select top sentences
    top_indices = sorted(pr, key=pr.get, reverse=True)
    top_sentences = [
        sentences[i].strip() for i in top_indices if len(sentences[i]) > 30
    ]

    # Remove near duplicates
    if remove_duplicates:
        unique_indices = remove_near_duplicates(top_sentences, threshold=0.7)
        top_sentences = [top_sentences[i] for i in unique_indices][:n]

    if itemize:
        return "- " + "\n- ".join(top_sentences)
    return top_sentences
