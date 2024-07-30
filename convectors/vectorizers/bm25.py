import numpy as np
from scipy.sparse import csr_matrix, find
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


class BM25Vectorizer:
    def __init__(
        self,
        k1=1.5,
        b=0.75,
        max_features=None,
        dtype=np.float32,
        min_df=1,
        max_df=1.0,
    ):
        self.k1 = k1
        self.b = b
        self._vectorizer = CountVectorizer(
            token_pattern=None,
            max_features=max_features,
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            min_df=min_df,
            max_df=max_df,
        )
        self.avg_doc_length = 0.0
        self.weights = None
        self.dtype = dtype

    def fit(self, raw_documents, y=None):
        """
        Learn vocabulary and idf from training set.
        """
        X = self._vectorizer.fit_transform(raw_documents)
        self.doc_count, _ = X.shape
        self.avg_doc_length = X.sum() / self.doc_count
        doc_lengths = X.sum(axis=1)

        # Compute IDF values
        df = np.diff(X.tocsc().indptr)
        idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        self.idf = np.asarray(idf, dtype=self.dtype)

        # Compute document-specific denominators for scoring formula
        denom = self.k1 * (1 - self.b + self.b * doc_lengths / self.avg_doc_length)
        self.denom = np.asarray(denom, dtype=np.float64).squeeze()
        self._trained = True
        return self

    def get_feature_names_out(self):
        return self._vectorizer.get_feature_names_out()

    def _bm25_weight(self, row, col, data, doc_id):
        """Compute individual BM25 weights"""
        return (data * (self.k1 + 1)) / (data + self.denom[doc_id]) * self.idf[col]

    def transform(self, raw_documents):
        """
        Transform documents to BM25 feature matrix using vectorized operations.
        """
        X = self._vectorizer.transform(raw_documents)
        rows, cols, data = find(X)

        # Vectorized computation of BM25 weights
        values = self._bm25_weight(rows, cols, data, rows)

        A = csr_matrix((values, (rows, cols)), shape=X.shape, dtype=self.dtype)
        normalize(A, norm="l2", axis=1, copy=False)
        return A

    def fit_transform(self, raw_documents, y=None):
        """
        Fit to data, then transform it.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)
