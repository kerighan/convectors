from .. import Layer
import numpy as np
import pandas as pd


class TfIdf(Layer):
    parallel = False
    trainable = True
    document_wise = False

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
        super(TfIdf, self).__init__(input, output, name, verbose, False)

        self.sparse = sparse
        self.vectorizer = TfidfVectorizer(
            preprocessor=lambda x: x,
            tokenizer=lambda x: x,
            max_features=max_features,
            min_df=min_df, max_df=max_df,
            token_pattern=None,
            **kwargs)

    def fit(self, series):
        self.vectorizer.fit(series)
        self.trained = True

    def process_series(self, series):
        res = self.vectorizer.transform(series)
        if self.sparse:
            return res
        return np.array(res.todense())


class Count(Layer):
    parallel = False
    trainable = True
    document_wise = False

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
        from sklearn.feature_extraction.text import CountVectorizer
        super(Count, self).__init__(input, output, name, verbose, False)

        self.sparse = sparse
        self.vectorizer = CountVectorizer(
            preprocessor=lambda x: x,
            tokenizer=lambda x: x,
            max_features=max_features,
            min_df=min_df, max_df=max_df,
            token_pattern=None,
            **kwargs)

    def fit(self, series):
        self.vectorizer.fit(series)
        self.trained = True

    def process_series(self, series):
        res = self.vectorizer.transform(series)
        if self.sparse:
            return res
        return np.array(res.todense())


class Embedding(Layer):
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
        pad=False,
        padding="pre",
        unk_token=True,
        name=None,
        verbose=True
    ):
        super(Embedding, self).__init__(input, output, name, verbose, False)
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
        self.word2id = {}
        self.unk_token = unk_token

    def fit(self, series):
        from collections import Counter
        import itertools

        tf = Counter(itertools.chain(*series))
        n_words = 1
        threshold = self.max_features + 1 - self.unk_token
        for word, freq in tf.most_common():
            if freq < self.min_tf:
                break
            if n_words >= threshold:
                break
            self.word2id[word] = n_words
            n_words += 1

        # add special token
        if self.unk_token:
            self.word2id["<UNK>"] = n_words
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
