from ..base_layer import Layer
from typing import Any, Optional
from .utils import NAME_TO_REGEX
import numpy as np
import re


class Prefix(Layer):
    def __init__(
        self,
        prefix: str,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self.prefix = prefix

    def process_document(self, document: Any) -> Any:
        return self.prefix + document


class Suffix(Layer):
    def __init__(
        self,
        suffix: str,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self.suffix = suffix

    def process_document(self, document: Any) -> Any:
        return document + self.suffix


class Limit(Layer):
    def __init__(
        self,
        limit: int,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self.limit = limit

    def process_document(self, document: Any) -> Any:
        return document[: self.limit]


class Pad(Layer):
    def __init__(
        self,
        maxlen: int,
        pad: str = "post",
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self.pad = pad
        self.maxlen = maxlen

    def process_documents(self, documents: Any) -> Any:
        if self.maxlen is None:
            maxlen = max(len(doc) for doc in documents)
        else:
            maxlen = self.maxlen

        padded_documents = []
        for doc in documents:
            if len(doc) >= maxlen:
                padded_documents.append(doc[:maxlen])
            else:
                if self.pad == "post":
                    padded_doc = np.pad(doc, (0, maxlen - len(doc)), mode="constant")
                elif self.pad == "pre":
                    padded_doc = np.pad(doc, (maxlen - len(doc), 0), mode="constant")
                padded_documents.append(padded_doc)
        return np.array(padded_documents)


class Sub(Layer):
    def __init__(
        self,
        regex: str = "url",
        replace: str = "",
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        if regex in NAME_TO_REGEX:
            regex = NAME_TO_REGEX[regex]
        else:
            regex = re.compile(regex)
        self.regex = regex
        self.replace = replace

    def process_document(self, document: Any) -> Any:
        return re.sub(self.regex, self.replace, document)


class FindAll(Layer):
    def __init__(
        self,
        regex: str = "url",
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        if regex in NAME_TO_REGEX:
            regex = NAME_TO_REGEX[regex]
        else:
            regex = re.compile(regex)
        self.regex = regex

    def process_document(self, document: Any) -> Any:
        res = re.findall(self.regex, document)
        return res


class DocumentSplitter(Layer):
    def __init__(
        self,
        maxlen: int = 128,
        overlap: int = 0,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        assert maxlen > overlap
        self.maxlen = maxlen
        self.overlap = overlap

    def _split_document(self, tokens):
        start = 0
        seq = []
        while start < len(tokens):
            end = start + self.maxlen
            if end > len(tokens):
                end = len(tokens)

            sequence = tokens[start:end]
            if len(sequence) < self.maxlen:
                break
            seq.append(sequence)
            start += self.maxlen - self.overlap
        return seq

    def process_documents(self, documents: Any) -> Any:
        splitted_documents = []
        for doc in documents:
            seq = self._split_document(doc)
            if len(seq) > 0:
                splitted_documents.extend(seq)
        return splitted_documents


class OneHot(Layer):
    def __init__(
        self,
        to_categorical: bool = False,
        threshold: float = None,
        unk_token: Any = None,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self._to_categorical = to_categorical
        self._threshold = threshold
        self._unk_token = unk_token
        self._decode_mode = False

    def fit(self, series):
        import itertools
        from collections import Counter

        # get first document
        if isinstance(series, list):
            first_doc = series[0]
        else:
            first_doc = series.iloc[0]

        if isinstance(first_doc, list):
            tf = Counter(itertools.chain(*series))
            self._multilabel = True
        else:
            tf = Counter(series)
            self._multilabel = False

        self.id_to_class = {i: c for i, (c, _) in enumerate(tf.most_common())}
        self.class_to_id = {c: i for i, c in self.id_to_class.items()}
        self.n_features = len(self.id_to_class)

    def process_documents(self, series: Any) -> Any:
        if not self._decode_mode:
            N = len(series)
            n_features = len(self.class_to_id)
            if not self._multilabel:
                if self._to_categorical:
                    X = np.zeros((N, n_features), dtype=np.bool_)
                    for i, class_ in enumerate(series):
                        X[i, self.class_to_id[class_]] = 1
                else:
                    X = np.zeros((N,), dtype=np.uint64)
                    for i, class_ in enumerate(series):
                        X[i] = self.class_to_id[class_]
            else:
                if self._to_categorical:
                    X = np.zeros((N, n_features), dtype=np.bool_)
                    for i, classes in enumerate(series):
                        for class_ in classes:
                            c = self.class_to_id.get(class_, None)
                            if c is None:
                                continue
                            X[i, c] = 1
                else:
                    X = []
                    for i, classes in enumerate(series):
                        tmp = []
                        for class_ in classes:
                            c = self.class_to_id.get(class_, None)
                            if c is None:
                                continue
                            tmp.append(c)
                        X.append(tmp)
        else:
            if not self._multilabel:
                if isinstance(series[0], list):
                    if self._threshold is None:
                        x_nan, _ = np.where(np.isnan(series))
                        x_nan = np.unique(x_nan)

                        X = np.argmax(series, axis=1)
                        X = [self.id_to_class[c] for c in X]
                        if self.unk_token is not None:
                            for idx in x_nan:
                                X[idx] = self.unk_token
                        else:
                            for idx in x_nan:
                                X[idx] = None
                    else:
                        t = self._threshold
                        assert self.unk_token is not None

                        x_nan, _ = np.where(np.isnan(series))
                        x_nan = np.unique(x_nan)

                        X = np.argmax(series, axis=1)
                        max_list = np.max(series, axis=1)
                        X = [
                            self.id_to_class[c] if m >= t else self.unk_token
                            for c, m in zip(X, max_list)
                        ]
                        for idx in x_nan:
                            X[idx] = self.unk_token
                elif isinstance(series, np.ndarray):
                    if len(series.shape) == 2:
                        X = np.argmax(series, axis=1)
                        X = [self.id_to_class[c] for c in X]
                    else:
                        X = [self.id_to_class[c] for c in series]
                else:
                    X = [self.id_to_class[c] for c in series]
        return X

    def get_decoder(self):
        from copy import deepcopy

        obj = deepcopy(self)
        obj._decode_mode = True
        return obj


class Normalize(Layer):
    def __init__(
        self,
        norm: str = "l2",
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self.norm = norm

    def process_documents(self, documents: Any) -> Any:
        from sklearn.preprocessing import normalize

        return normalize(documents, norm=self.norm)
