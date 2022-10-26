import numpy as np
import pandas as pd

from .. import Layer
from ..utils import identity


class VectorizerLayer(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        **kwargs,
    ):
        super(VectorizerLayer, self).__init__(
            input, output, name, verbose, False)
        self.n_features = None

    def fit(self, series, y=None):
        self.vectorizer.fit(series)
        self.n_features = len(self.vectorizer.get_feature_names_out())

    def process_series(self, series):
        res = self.vectorizer.transform(series)
        if self.sparse:
            return res
        return np.array(res.todense())

    def get_graph(
        self,
        documents,
        threshold=.1,
        pmi_threshold=.5,
        window_size=5,
        k=1
    ):

        import networkx as nx
        from convectors.linguistics import pmi

        edges = []
        X = self.vectorizer.transform(documents)
        features = self.vectorizer.get_feature_names_out()
        features_set = set(features)
        for i in range(len(documents)):
            start, end = X.indptr[i], X.indptr[i+1]
            for ind, w in zip(X.indices[start:end], X.data[start:end]):
                if w < threshold:
                    continue
                word = features[ind]
                edges.append((i, word, w))

        pmi_ = pmi(documents,
                   undirected=True,
                   threshold=pmi_threshold,
                   window_size=window_size,
                   k=k)
        for (a, b), w in pmi_.items():
            if a in features_set and b in features_set:
                edges.append((a, b, w))

        G = nx.Graph()
        G.add_nodes_from(range(len(documents)))
        G.add_weighted_edges_from(edges)
        return G


class TfIdf(VectorizerLayer):
    def __init__(
        self,
        input=None,
        output=None,
        max_features=None,
        min_df=0,
        max_df=1.,
        sparse=True,
        name=None,
        verbose=True,
        **kwargs,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer
        super(TfIdf, self).__init__(input, output, name, verbose)

        self.sparse = sparse
        self.vectorizer = TfidfVectorizer(
            preprocessor=identity,
            tokenizer=identity,
            max_features=max_features,
            min_df=min_df, max_df=max_df,
            token_pattern=None,
            lowercase=False,
            **kwargs)


class OddsVectorizer(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        max_features=None,
        norm="l2",
        threshold=2,
        ci=.95,
        name=None,
        verbose=True
    ):
        from scipy.stats import norm as scipy_norm
        super().__init__(input, output, name, verbose)
        self.max_features = max_features
        self.norm = norm
        self.threshold = threshold
        self.ci_factor = -scipy_norm.isf((1 - ci) / 2.)

    def fit(self, documents, y=None):
        import itertools
        from collections import Counter
        self.tf = Counter(itertools.chain(*documents))
        if self.max_features is not None:
            self.tf = dict(self.tf.most_common(self.max_features))
        self.token2id = {token: i for i, token in enumerate(self.tf.keys())}
        self.dim = len(self.token2id)
        self.total_count = sum(self.tf.values())
        self.n_features = len(self.tf)

    def fit_transform(self, documents):
        self.fit(documents)
        X = self.transform(documents)
        return X

    def process_series(self, series):
        from scipy.sparse import csr_matrix
        from sklearn.preprocessing import normalize

        xs, ys, data = [], [], []
        for i, doc in enumerate(series):
            xs_, ys_, data_ = self.vectorize(doc, i)
            xs.extend(xs_)
            ys.extend(ys_)
            data.extend(data_)

        X = csr_matrix((data, (xs, ys)),
                       dtype=float,
                       shape=(len(series), self.dim))
        if self.norm is not None:
            normalize(X, axis=1, norm=self.norm, copy=False)
        return X

    def get_graph(
        self,
        documents,
        threshold=2,
        pmi_threshold=.5,
        window_size=5,
        k=1
    ):
        from collections import Counter

        import networkx as nx
        from convectors.linguistics import pmi

        edges = []
        for i, doc in enumerate(documents):
            count = Counter(doc)
            doc_count = sum(count.values())
            for word, tf in count.items():
                if word not in self.tf:
                    continue
                odds = self.compute_odds(
                    tf, doc_count, self.tf[word], self.total_count)
                if odds < threshold:
                    continue
                edges.append((i, word, np.log(odds)))

        pmi_ = pmi(documents,
                   undirected=True,
                   threshold=pmi_threshold,
                   window_size=window_size,
                   k=k)
        for (a, b), w in pmi_.items():
            if a in self.tf and b in self.tf:
                edges.append((a, b, w))

        G = nx.Graph()
        G.add_nodes_from(range(len(documents)))
        G.add_weighted_edges_from(edges)
        return G

    def vectorize(self, doc, i):
        from collections import Counter
        count = Counter(doc)
        doc_count = sum(count.values())
        xs, ys, data = [], [], []
        for word, tf in count.items():
            if word not in self.tf:
                continue
            odds = self.compute_odds(
                tf, doc_count, self.tf[word], self.total_count)
            if odds <= self.threshold:
                continue
            xs.append(i)
            ys.append(self.token2id[word])
            data.append(np.log(odds))
            # data.append(odds)

        return xs, ys, data

    def compute_odds(self, a, b, c, d):
        eps = 1
        b = max(b - a, eps)
        d = max(d - c, eps)
        odds_ratio = (a / b) / (c / d)

        uncertainty = np.sqrt(1/a + 1/b + 1/c + 1/d)
        uncertainty = np.exp(self.ci_factor*uncertainty)
        odds_ratio *= uncertainty
        return odds_ratio


class CountVectorizer(VectorizerLayer):
    def __init__(
        self,
        input=None,
        output=None,
        max_features=None,
        min_df=0,
        max_df=1.,
        sparse=True,
        name=None,
        verbose=True,
        **kwargs,
    ):
        from sklearn.feature_extraction.text import CountVectorizer as CV
        super(CountVectorizer, self).__init__(input, output, name, verbose)

        self.sparse = sparse
        self.vectorizer = CV(
            preprocessor=identity,
            tokenizer=identity,
            max_features=max_features,
            min_df=min_df, max_df=max_df,
            token_pattern=None,
            **kwargs)


class Sequence(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        max_features=None,
        min_tf=0,
        max_tf=float("inf"),
        min_df=0,
        max_df=float("inf"),
        maxlen=None,
        pad=True,
        padding="post",
        unk_token=False,
        mask_token=False,
        empty_token=True,
        ragged=False,
        feature2id=None,
        model=None,
        name=None,
        verbose=True
    ):
        super(Sequence, self).__init__(input, output, name, verbose, False)
        if max_features is None:
            max_features = float("inf")
        self.max_features = max_features
        self.min_tf = min_tf
        self.max_tf = max_tf
        self.min_df = min_df
        self.max_df = max_df
        self.maxlen = maxlen
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.empty_token = empty_token
        self.ragged = ragged
        self.padding = padding
        if ragged:
            pad = False

        if pad and maxlen is not None:
            self.process_series = self._crop_pad_sequences
        elif pad:
            self.process_series = self._pad_sequences
        elif maxlen is not None:
            self.process_series = self._crop_sequences
        else:
            self.process_series = self._sequences

    def _crop_pad_sequences(self, documents):
        unk_token_id = self.feature2id.get("_UNK_", -1)
        empty_token_id = self.feature2id.get("_EMPTY_", -1)
        maxlen = self.maxlen
        results = np.zeros((len(documents), maxlen), dtype=np.int32)
        for i, doc in enumerate(documents):
            doc = [self.feature2id.get(t, unk_token_id) for t in doc]
            if not self.unk_token:
                doc = [t for t in doc if t != -1]
            if len(doc) > maxlen:
                doc = doc[:maxlen]

            if len(doc) == 0 and empty_token_id != -1:
                doc = [empty_token_id]

            doc_len = len(doc)
            if doc_len <= maxlen:
                if self.padding == "post":
                    # doc += [0]*(maxlen-doc_len)
                    results[i, :doc_len] = doc
                else:
                    results[i, -doc_len:] = doc
        return results

    def _pad_sequences(self, documents):
        unk_token_id = self.feature2id.get("_UNK_", -1)
        empty_token_id = self.feature2id.get("_EMPTY_", -1)
        results = []
        maxlen = -float("inf")
        for doc in documents:
            doc = [self.feature2id.get(t, unk_token_id) for t in doc]
            if not self.unk_token:
                doc = [t for t in doc if t != -1]

            if len(doc) == 0 and empty_token_id != -1:
                doc = [empty_token_id]

            doc_len = len(doc)
            if doc_len > maxlen:
                maxlen = doc_len
            results.append(doc)
        if self.padding == "post":
            results = np.array([doc + [0]*(maxlen - len(doc))
                                for doc in results], dtype=np.int64)
        else:
            results = np.array([[0]*(maxlen - len(doc)) + doc
                                for doc in results], dtype=np.int64)
        return results

    def _crop_sequences(self, documents):
        unk_token_id = self.feature2id.get("_UNK_", -1)
        empty_token_id = self.feature2id.get("_EMPTY_", -1)
        maxlen = self.maxlen
        results = []
        for doc in documents:
            doc = [self.feature2id.get(t, unk_token_id) for t in doc]
            if not self.unk_token:
                doc = [t for t in doc if t != -1]

            if len(doc) > maxlen:
                doc = doc[:maxlen]
            if len(doc) == 0 and empty_token_id != -1:
                doc = [empty_token_id]
            results.append(doc)
        if self.ragged:
            import tensorflow as tf
            return tf.ragged.constant(results, dtype=tf.int64)
        return results

    def _sequences(self, documents):
        unk_token_id = self.feature2id.get("_UNK_", -1)
        empty_token_id = self.feature2id.get("_EMPTY_", -1)
        results = []
        for doc in documents:
            doc = [self.feature2id.get(t, unk_token_id) for t in doc]
            if not self.unk_token:
                doc = [t for t in doc if t != -1]
            if len(doc) == 0 and empty_token_id != -1:
                doc = [empty_token_id]
            results.append(doc)
        if self.ragged:
            import tensorflow as tf
            return tf.ragged.constant(results, dtype=tf.int64)
        return results

    def fit(self, series, y=None):
        if hasattr(self, "feature2id"):
            return

        from ..utils import get_features_from_documents
        self.feature2id, self.id2feature = get_features_from_documents(
            series, max_features=self.max_features, min_tf=self.min_tf,
            max_tf=self.max_tf, min_df=self.min_df, max_df=self.max_df,
            unk_token=self.unk_token, mask_token=self.mask_token,
            empty_token=self.empty_token)
        self.n_features = len(self.feature2id)


class OneHot(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        to_categorical=False,
        threshold=None,
        unk_token=None,
        max_features=None,
        verbose=True
    ):
        super(OneHot, self).__init__(input, output, name, verbose, False)
        self.to_categorical = to_categorical
        self.class2id = {}
        self.multilabel = False
        self.decode_mode = False
        self.threshold = threshold
        self.unk_token = unk_token
        self.max_features = max_features

    def fit(self, series, y=None):
        import itertools
        from collections import Counter

        if isinstance(series, list):
            zero_doc = series[0]
        else:
            zero_doc = series.iloc[0]
        if isinstance(zero_doc, list):
            tf = Counter(itertools.chain(*series))
            self.multilabel = True
        else:
            tf = Counter(series)
            self.multilabel = False

        if self.max_features is None:
            self.class2id = {w: i for i, (w, _) in enumerate(tf.most_common())}
        else:
            self.class2id = {w: i for i, (w, _) in enumerate(
                tf.most_common(self.max_features))}
        self.id2class = {i: w for w, i in self.class2id.items()}
        self.n_features = len(self.class2id)

    def process_series(self, series):
        if not self.decode_mode:
            N = len(series)
            n_features = len(self.class2id)
            if not self.multilabel:
                if self.to_categorical:
                    X = np.zeros((N, n_features), dtype=np.bool_)
                    for i, class_ in enumerate(series):
                        X[i, self.class2id[class_]] = 1
                else:
                    X = np.zeros((N,), dtype=np.uint64)
                    for i, class_ in enumerate(series):
                        X[i] = self.class2id[class_]
            else:
                if self.to_categorical:
                    X = np.zeros((N, n_features), dtype=np.bool_)
                    for i, classes in enumerate(series):
                        for class_ in classes:
                            c = self.class2id.get(class_, None)
                            if c is None:
                                continue
                            X[i, c] = 1
                else:
                    X = []
                    for i, classes in enumerate(series):
                        tmp = []
                        for class_ in classes:
                            c = self.class2id.get(class_, None)
                            if c is None:
                                continue
                            tmp.append(c)
                        X.append(tmp)
        else:
            if not self.multilabel:
                if len(series.shape) == 2:
                    if self.threshold is None:
                        x_nan, _ = np.where(np.isnan(series))
                        x_nan = np.unique(x_nan)

                        X = np.argmax(series, axis=1)
                        X = [self.id2class[c] for c in X]
                        if self.unk_token is not None:
                            for idx in x_nan:
                                X[idx] = self.unk_token
                        else:
                            for idx in x_nan:
                                X[idx] = None
                    else:
                        t = self.threshold
                        assert self.unk_token is not None

                        x_nan, _ = np.where(np.isnan(series))
                        x_nan = np.unique(x_nan)

                        X = np.argmax(series, axis=1)
                        max_list = np.max(series, axis=1)
                        X = [
                            self.id2class[c] if m >= t else self.unk_token
                            for c, m in zip(X, max_list)
                        ]
                        for idx in x_nan:
                            X[idx] = self.unk_token
                elif len(series.shape) == 1:
                    X = [self.id2class[c] for c in series]
                else:
                    raise ValueError(
                        f"Layer {self.name} got an input of wrong dimension")
        return X

    def get_decoder(self):
        from copy import deepcopy
        obj = deepcopy(self)
        obj.decode_mode = True
        return obj


class Doc2Vec(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        n_components=200,
        min_tf=2,
        epochs=40,
        model=None,
        name=None,
        verbose=True
    ):
        super(Doc2Vec, self).__init__(input, output, name, verbose, False)
        self.n_components = n_components
        self.min_tf = min_tf
        self.epochs = epochs
        self.model = model

    def fit(self, series, y=None):
        import gensim

        train_corpus = [gensim.models.doc2vec.TaggedDocument(
            tokens, [i]) for i, tokens in enumerate(series)]

        self.model = gensim.models.doc2vec.Doc2Vec(
            vector_size=self.n_components,
            min_count=self.min_tf,
            epochs=self.epochs)
        self.model.build_vocab(train_corpus)
        self.model.train(
            train_corpus,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs)

    def process_series(self, series):
        if self.model is None:
            self.fit(series)
        return np.array(
            [self.model.infer_vector(doc) for doc in series])


class SWEM(Layer):
    parallel = True
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        model=None,
        method="concat",
        window=3,
        sample_points=np.linspace(.1, 10, 10),
        name=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(input, output, name, verbose)
        self.model = model
        self.window = window
        self.method = method
        self.sample_points = sample_points
        self.dim = self.model.weights.shape[1]

        if method == "concat":
            self.process_series = self.process_concat
        elif method == "hier":
            self.process_series = self.process_hierarchical
        elif method == "char-hier":
            self.process_series = self.process_characteristic_hierarchical
        elif method == "characteristic":
            self.process_series = self.process_characteristic

    def process_concat(self, series):
        res = []
        for doc in series:
            words = [self.model[w] for w in doc if w in self.model.word2id]
            if len(words) == 0:
                res.append(np.zeros((2*self.dim,)))
                continue
            words_avg = np.mean(words, axis=0)
            words_max = np.max(words, axis=0)
            vector = np.concatenate([words_avg, words_max], axis=0)
            res.append(vector)
        return np.array(res)

    def process_hierarchical(self, series):
        res = []
        for doc in series:
            words = np.array([self.model[w]
                             for w in doc if w in self.model.word2id])
            if len(words) == 0:
                res.append(np.zeros((self.dim,)))
                continue

            ngrams = [np.mean(words[i:i+self.window], axis=0)
                      for i in range(max(1, len(words)-self.window+1))]
            vector = np.max(ngrams, axis=0)
            res.append(vector)
        return np.array(res)

    def process_characteristic(self, series):
        res = []
        for doc in series:
            words = np.array([self.model[w]
                             for w in doc if w in self.model.word2id])
            if len(words) == 0:
                words = np.zeros((1, self.dim))
            vector = self.sample_characteristic_function(words)
            res.append(vector)
        return np.array(res)

    def process_characteristic_hierarchical(self, series):
        res = []
        for doc in series:
            words = np.array([self.model[w]
                              for w in doc if w in self.model.word2id])

            if len(words) == 0:
                words = np.zeros((1, self.dim))

            ngrams = np.array(
                [np.mean(words[i:i+self.window], axis=0)
                 for i in range(max(1, len(words)-self.window+1))])
            vector = self.sample_characteristic_function(ngrams)
            res.append(vector)
        return np.array(res)

    def sample_characteristic_function(self, ngrams):
        vector = []
        for t in self.sample_points:
            attributes = t*ngrams
            real = np.mean(np.cos(attributes), axis=0)
            imag = np.mean(np.sin(attributes), axis=0)
            vector.append(real)
            vector.append(imag)
        vector = np.concatenate(vector)
        return vector
