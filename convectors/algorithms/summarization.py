def summarize(text, n=5, itemize=True, boost_words=None, boost_value=2):
    from convectors.layers import Tokenize, TfIdf, SnowballStem
    from .duplicates import remove_near_duplicates
    from sklearn.preprocessing import normalize
    import networkx as nx

    sentence_tokenize = Tokenize(
        sentence_tokenize=True,
        word_tokenize=False,
        strip_accents=False,
        strip_punctuation=False,
        lower=False,
        split_lines=True,
    )
    sentences = sentence_tokenize(text)
    # make pairs of sentences joined by a space
    sentences = [" ".join(sentences[i : i + 2]) for i in range(len(sentences) - 1)]

    vectorizer = Tokenize(stopwords=["fr", "en", "media", "url"])
    vectorizer += SnowballStem()
    vectorizer += TfIdf()

    X = vectorizer(sentences)

    if boost_words is not None:
        boost_words_ids = vectorizer(" ".join(boost_words)).indices
        for x in boost_words_ids:
            X[:, x] *= boost_value
        normalize(X, norm="l2", axis=1, copy=False)

    A = X.dot(X.T)
    # threshold
    # A[A < 0.05] = 0
    # A.data[A.data < 0.05] = 0
    # remove diag
    A.setdiag(0)

    # create graph from adjacency sparse array
    G = nx.from_scipy_sparse_array(A)
    try:
        pr = nx.eigenvector_centrality(G)
    except nx.exception.PowerIterationFailedConvergence:
        pr = nx.pagerank(G)
    top_sentences = sorted(pr, key=pr.get, reverse=True)
    top_sentences = [sentences[i].strip() for i in top_sentences]

    # remove near duplicates
    indices = remove_near_duplicates(top_sentences, threshold=0.7)
    top_sentences = [top_sentences[i] for i in indices]
    top_sentences = [it for it in top_sentences if len(it) > 30][:n]

    if itemize:
        top_sentences = "- " + "\n- ".join(top_sentences)

    return top_sentences
