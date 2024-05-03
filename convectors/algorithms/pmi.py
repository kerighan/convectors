import itertools
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
import networkx as nx
import math
from tqdm import tqdm


def pmi(
    series,
    window_size=3,
    min_count=2,
    min_df=2,
    max_df=0.5,
    max_features=None,
    minimum_pmi=0.3,
    normalize=True,
    verbose=True,
):
    # Flatten the series and count individual word frequencies
    # freq = Counter(itertools.chain.from_iterable(series))
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        max_features=max_features,
    )
    X = vectorizer.fit_transform(series)
    freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))

    texts = [[word for word in text if word in freq] for text in series]
    total_words = sum(freq.values())  # Total number of words

    cooc = defaultdict(int)
    n_pairs = 0
    if verbose:
        iterator = tqdm(texts, desc="Calculating co-occurrences")
    else:
        iterator = texts
    for words in iterator:
        for i in range(len(words) - window_size + 1):
            window = words[i : i + window_size]
            for j, w1 in enumerate(window):
                for w2 in window[j + 1 :]:
                    if w1 == w2:
                        continue
                    pair = tuple(sorted((w1, w2)))
                    cooc[pair] += 1
                    n_pairs += 1

    # Calculate PMI or NPMI for each pair
    pmi_dict = {}
    for (x, y), count in cooc.items():
        if count >= min_count:
            p_x = freq[x] / total_words
            p_y = freq[y] / total_words
            p_xy = count / n_pairs
            if p_xy > 0:
                pmi_value = math.log(p_xy / (p_x * p_y))
                if normalize:
                    # Normalize PMI to [-1, 1] interval using log base e
                    npmi_value = pmi_value / (-math.log(p_xy))
                    if minimum_pmi is None or npmi_value > minimum_pmi:
                        pmi_dict[(x, y)] = npmi_value
                elif minimum_pmi is None or pmi_value > minimum_pmi:
                    pmi_dict[(x, y)] = pmi_value
    G_pmi = nx.Graph()
    G_pmi.add_nodes_from(freq.keys())
    G_pmi.add_edges_from([(x, y, {"weight": pmi}) for (x, y), pmi in pmi_dict.items()])
    return G_pmi, freq


def text_graph(
    series,
    window_size=3,
    min_count=2,
    min_df=2,
    max_df=0.5,
    max_features=None,
    minimum_pmi=0.4,
    minimum_tfidf=0.2,
    normalize=True,
    verbose=True,
):
    from convectors.layers import TfIdf

    G_pmi, freq = pmi(
        series,
        window_size=window_size,
        min_count=min_count,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        minimum_pmi=minimum_pmi,
        normalize=normalize,
        verbose=verbose,
    )
    tfidf = TfIdf(max_df=max_df, min_df=min_df, max_features=max_features)
    X = tfidf(series)
    features = tfidf._vectorizer.get_feature_names_out()

    G_pmi.add_nodes_from(range(X.shape[0]))
    edges = []
    for word in features:
        if word in freq:
            i = tfidf._vectorizer.vocabulary_[word]
            slice_ = X[:, i].tocsc()
            weights = slice_.data
            doc_ids = slice_.indices

            for doc_id, weight in zip(doc_ids, weights):
                if weight > minimum_tfidf:
                    edges.append((word, doc_id, weight))
        else:
            print(word)
    G_pmi.add_weighted_edges_from(edges)
    return G_pmi


def summarize(text, n=5, itemize=True, boost_words=None, boost_value=2):
    from convectors.layers import Tokenize, TfIdf, SnowballStem
    from .duplicates import remove_near_duplicates
    from sklearn.preprocessing import normalize

    sentence_tokenize = Tokenize(
        sentence_tokenize=True,
        word_tokenize=False,
        strip_accents=False,
        strip_punctuation=False,
        lower=False,
        split_lines=True,
    )
    sentences = sentence_tokenize(text)

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


def text_graph_topics(
    data,
    window_size=20,
    min_count=4,
    min_df=3,
    max_df=0.25,
    minimum_tfidf=0.2,
    minimum_pmi=0.33,
    max_features=5000,
    min_docs=4,
    top_n_docs=10,
    stopwords=["fr", "en", "url", "media"],
    shuffle=False,
    verbose=True,
):
    from convectors.layers import Tokenize
    from .duplicates import remove_near_duplicates
    from cdlib import algorithms
    import pandas as pd
    import random

    tokenize = Tokenize(stopwords=stopwords)
    processed_data = tokenize(data)
    if shuffle:
        processed_data = [random.sample(text, len(text)) for text in processed_data]

    G = text_graph(
        processed_data,
        window_size=window_size,
        min_count=min_count,
        min_df=min_df,
        max_df=max_df,
        minimum_tfidf=minimum_tfidf,
        minimum_pmi=minimum_pmi,
        max_features=max_features,
        verbose=verbose,
    )

    coms = algorithms.louvain(G, weight="weight")
    node2cm = coms.to_node_community_map()
    cm2words = {}
    cm2docs = {}

    for node, cm in node2cm.items():
        cm = cm[0]
        if cm not in cm2words:
            cm2words[cm] = set()
        if cm not in cm2docs:
            cm2docs[cm] = set()
        if isinstance(node, str):
            cm2words[cm].add(node)
        else:
            cm2docs[cm].add(node)

    topic_words = []
    topic_docs = []
    topic_summaries = []
    for cm in cm2docs:
        docs = cm2docs[cm]
        if len(docs) < min_docs:
            continue

        nodes = docs.union(cm2words[cm])
        H = nx.subgraph(G, nodes)
        try:
            pr = nx.eigenvector_centrality(H)
        except:
            pr = nx.degree_centrality(H)
        sorted_words = sorted(cm2words[cm], key=lambda x: pr[x], reverse=True)

        doc_scores = {}
        for doc in docs:
            doc_text_tokens = processed_data[doc]
            score = sum(
                pr[word] for word in sorted_words if word in doc_text_tokens
            )  # Use PageRank values as weights
            doc_scores[doc] = score

        best_docs = sorted(
            doc_scores, key=doc_scores.get, reverse=True
        )  # Also keep top_n_docs sorted by their score

        # topic_text = "\n".join([data[doc] for doc in topic_docs[-1][:top_n_docs]])
        topic_text = "\n".join([data[doc] for doc in best_docs[:top_n_docs]])
        topic_summary = summarize(topic_text, boost_words=sorted_words)

        topic_words.append(sorted_words)
        topic_docs.append(best_docs)
        topic_summaries.append(topic_summary)

    topic_info = pd.DataFrame()
    topic_info["words"] = topic_words
    topic_info["docs"] = topic_docs
    topic_info["summary"] = topic_summaries
    topic_info["n_docs"] = topic_info["docs"].apply(len)
    topic_info = topic_info.sort_values("n_docs", ascending=False).reset_index(
        drop=True
    )
    return topic_info
