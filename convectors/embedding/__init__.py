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
        self.n_features = len(self.vectorizer.get_feature_names())

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
            **kwargs)


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
        verbose=True
    ):
        super(OneHot, self).__init__(input, output, name, verbose, False)
        self.to_categorical = to_categorical
        self.class2id = {}
        self.multilabel = False
        self.decode_mode = False
        self.threshold = threshold
        self.unk_token = unk_token

    def fit(self, series, y=None):
        import itertools
        from collections import Counter
        if isinstance(series[0], list):
            tf = Counter(itertools.chain(*series))
            self.multilabel = True
        else:
            tf = Counter(series)
            self.multilabel = False

        self.class2id = {w: i for i, (w, _) in enumerate(tf.most_common())}
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
                            X[i, self.class2id[class_]] = 1
                else:
                    X = []
                    for i, classes in enumerate(series):
                        tmp = []
                        for class_ in classes:
                            tmp.append(self.class2id[class_])
                        X.append(tmp)
        else:
            if not self.multilabel:
                if len(series.shape) == 2:
                    if self.threshold is None:
                        X = np.argmax(series, axis=1)
                        X = [self.id2class[c] for c in X]
                    else:
                        t = self.threshold
                        assert self.unk_token is not None
                        X = np.argmax(series, axis=1)
                        max_list = np.max(series, axis=1)
                        X = [
                            self.id2class[c] if m >= t else self.unk_token
                            for c, m in zip(X, max_list)

                        ]
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
