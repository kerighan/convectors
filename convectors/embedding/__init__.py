from .. import Layer
import numpy as np


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
