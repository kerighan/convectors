from ..base_layer import Layer
import numpy as np
from typing import Any, Optional


class VectorizerLayer(Layer):
    def fit(self, documents: Any, labels: Any = None) -> None:
        self._vectorizer.fit(documents)
        self._trained = True

    @property
    def n_features(self) -> int:
        return len(self._vectorizer.get_feature_names_out())

    def process_documents(self, documents: Any) -> Any:
        res = self._vectorizer.transform(documents)
        if self._sparse:
            return res
        return np.array(res.todense())


class TfIdf(VectorizerLayer):
    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: float = 0.0,
        max_df: float = 1.0,
        lowercase: bool = True,
        sparse: bool = True,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(name, verbose)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._sparse = sparse
        self._vectorizer = TfidfVectorizer(
            preprocessor=lambda x: x,
            tokenizer=lambda x: x,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            token_pattern=None,
            lowercase=lowercase,
            **kwargs
        )


class BM25(VectorizerLayer):
    _sparse = True

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: Optional[int] = None,
        min_df: int = 0,
        max_df: float = 1.0,
        lowercase: bool = True,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(name, verbose)
        from .bm25 import BM25Vectorizer

        self._vectorizer = BM25Vectorizer(
            k1=k1, b=b, max_features=max_features, **kwargs
        )


class CountVectorizer(VectorizerLayer):
    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: int = 0,
        max_df: float = 1.0,
        sparse: bool = True,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(name, verbose)
        from sklearn.feature_extraction.text import CountVectorizer

        self._sparse = sparse
        self._vectorizer = CountVectorizer(
            preprocessor=lambda x: x,
            tokenizer=lambda x: x,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            token_pattern=None,
            lowercase=False,
            **kwargs
        )


class HashingVectorizer(VectorizerLayer):
    def __init__(
        self,
        n_features: int = 2**18,
        binary: bool = False,
        lowercase: bool = True,
        sparse: bool = True,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(name, verbose)
        from sklearn.feature_extraction.text import HashingVectorizer

        self._sparse = sparse
        self._vectorizer = HashingVectorizer(
            preprocessor=lambda x: x,
            tokenizer=lambda x: x,
            n_features=n_features,
            binary=binary,
            token_pattern=None,
            lowercase=lowercase,
            **kwargs
        )
        self._n_features = n_features

    @property
    def n_features(self) -> int:
        return self._n_features
