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
        sparse=True,
        name=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__(input, output, name, verbose)
        self.max_features = max_features

    def fit(self, documents, y=None):
        import itertools
        from collections import Counter
        self.tf = Counter(itertools.chain(*documents))
        if self.max_features is not None:
            self.tf = dict(self.tf.most_common(self.max_features))
        self.token2id = {token: i for i, token in enumerate(self.tf.keys())}
        self.dim = len(self.token2id)
        self.total_count = sum(self.tf.values())

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
        normalize(X, axis=1, norm="l2", copy=False)
        return X

    def get_graph(
        self, documents, threshold=2, pmi_threshold=.5, window_size=5
    ):
        from collections import Counter

        import networkx as nx
        from convectors.linguistics import pmi

        edges = []
        for i, doc in enumerate(documents):
            count = Counter(doc)
            count_total = sum(count.values())
            for word, tf in count.items():
                if word not in self.tf:
                    continue
                odds = self.get_low_odds(
                    tf, count_total, self.tf[word], self.total_count)
                if odds < threshold:
                    continue
                edges.append((i, word, np.log(odds)))

        pmi_ = pmi(documents, undirected=True,
                   threshold=pmi_threshold, window_size=window_size)
        for (a, b), w in pmi_.items():
            if a in self.tf and b in self.tf:
                edges.append((a, b, w))

        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        return G

    def vectorize(self, doc, i):
        from collections import Counter
        count = Counter(doc)
        count_total = sum(count.values())
        xs, ys, data = [], [], []
        for word, tf in count.items():
            if word not in self.tf:
                continue
            odds = self.get_low_odds(
                tf, count_total, self.tf[word], self.total_count)
            if odds <= 1:
                continue
            xs.append(i)
            ys.append(self.token2id[word])
            data.append(np.log(odds))
        return xs, ys, data

    def get_low_odds(self, a, c, b, d):
        eps = 1e-6
        c = max(c - a, eps)
        d = max(d - b, eps)
        a = max(a, eps)
        b = max(b, eps)
        odds_ratio = (a/c) / (b/d)

        uncertainty = np.sqrt(1/a+1/b+1/c+1/d)
        low = odds_ratio * np.exp(-1.96*uncertainty)
        return low


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
        maxlen=None,
        pad=True,
        padding="pre",
        unk_token=True,
        mask_token=True,
        word2id=None,
        model=None,
        name=None,
        verbose=True
    ):
        super(Sequence, self).__init__(input, output, name, verbose, False)
        if max_features is None:
            max_features = float("inf")
        self.max_features = max_features
        self.min_tf = min_tf
        self.maxlen = maxlen
        self.pad = pad
        if padding == "pre":
            self.padding = True
        elif padding == "post":
            self.padding = False
        else:
            raise ValueError("padding argument should be 'pre' or 'post'")
        self.unk_token = unk_token
        self.mask_token = mask_token

        if model is not None:  # gensim model provided
            if hasattr(model, "word2id"):  # convectors WordVectors instance
                self.word2id = {k: v for k, v in model.word2id.items()}
            else:
                self.word2id = {}
                for word, index in model.wv.key_to_index.items():
                    self.word2id[word] = index + 1  # offset index by 1
            self.unk_token = "<UNK>" in self.word2id
            self.mask_token = "<MASK>" in self.word2id
            self.n_features = len(self.word2id)
        elif word2id is not None:  # custom word2id mapping
            self.word2id = word2id
            self.unk_token = "<UNK>" in self.word2id
            self.mask_token = "<MASK>" in self.word2id
            self.n_features = len(self.word2id)
        else:
            self.word2id = None

    def fit(self, series, y=None):
        if self.word2id is not None:
            return

        import itertools
        from collections import Counter

        self.word2id = {}
        tf = Counter(itertools.chain(*series))
        n_words = 1
        threshold = (
            self.max_features + 1
            - self.unk_token - self.mask_token)
        for word, freq in tf.most_common():
            if freq < self.min_tf:
                break
            if n_words >= threshold:
                break
            self.word2id[word] = n_words
            n_words += 1

        # add special token
        if self.unk_token and "<UNK>" not in self.word2id:
            self.word2id["<UNK>"] = n_words
            n_words += 1
        if self.mask_token and "<MASK>" not in self.word2id:
            self.word2id["<MASK>"] = n_words
            n_words += 1
        self.n_features = len(self.word2id)

    def process_series(self, series):
        if self.unk_token:
            unk_id = self.word2id["<UNK>"]
            doc_ids = [
                [self.word2id.get(w, unk_id) for w in text]
                for text in series
            ]
        else:
            doc_ids = [
                [self.word2id[w] for w in text if w in self.word2id]
                for text in series
            ]
        if self.pad:
            if self.maxlen is None:
                max_length = max((len(d) for d in doc_ids))
                if self.padding:
                    doc_ids = [
                        [0]*(max_length - len(d)) + d
                        for d in doc_ids
                    ]
                else:
                    doc_ids = [
                        d + [0]*(max_length - len(d))
                        for d in doc_ids
                    ]
            else:
                for i in range(len(doc_ids)):
                    d = doc_ids[i]
                    if len(d) < self.maxlen:
                        if self.padding:
                            doc_ids[i] = [0] * (self.maxlen - len(d)) + d
                        else:
                            doc_ids[i] = d + [0] * (self.maxlen - len(d))
                    elif len(d) > self.maxlen:
                        if self.padding:
                            doc_ids[i] = d[:self.maxlen]
                        else:
                            doc_ids[i] = d[-self.maxlen:]
            return np.array(doc_ids, dtype=np.uint64)
        elif self.maxlen is not None:
            if self.padding:
                doc_ids = [d[:self.maxlen] for d in doc_ids]
            else:
                doc_ids = [d[-self.maxlen:] for d in doc_ids]
        if isinstance(series, pd.Series):
            return pd.Series(doc_ids, index=series.index)
        return doc_ids


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
