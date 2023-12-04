from ..base_layer import Layer
from functools import partial
from .utils import tokenize
from typing import Any, Callable, List, Optional, Set, Union, Dict
import numpy as np


class Tokenize(Layer):
    """
    This class represents a layer that tokenizes the input. It inherits from
    the Layer class and overrides the constructor to handle tokenization
    parameters.

    Parameters
    ----------
    stopwords : set or list, optional
        A set of stopwords to be excluded from the tokenization. If a list is
        given, it should contain names of predefined stopwords sets.
    strip_accents : bool, optional
        If True, accents will be removed from the input during tokenization.
        Default is False.
    strip_punctuation : bool, optional
        If True, punctuation will be removed from the input during
        tokenization. Default is True.
    sentence_tokenize : bool, optional
        If True, the input will be tokenized into sentences. Default is False.
    word_tokenize : bool, optional
        If True, the input will be tokenized into words. Default is True.
    lower : bool, optional
        If True, the input will be converted to lower case during tokenization.
        Default is True.
    name : str, optional
        The name of the layer. If not given, the name will be derived from the
        class name.
    verbose : bool, optional
        If True, the layer will output verbose messages during execution.
        Default is True.

    """

    _trainable = False

    def __init__(
        self,
        stopwords: Optional[Union[Set[str], List[str]]] = None,
        strip_accents: bool = False,
        strip_punctuation: bool = True,
        sentence_tokenize: bool = False,
        word_tokenize: bool = True,
        lower: bool = True,
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(name, verbose)
        if stopwords is not None:
            if isinstance(stopwords, set):
                self._stopwords = stopwords
            else:
                from .stopwords import stopwords as sw

                self._stopwords = set()
                for item in stopwords:
                    self._stopwords.update(sw[item])
        else:
            self._stopwords = stopwords
        self._strip_accents = strip_accents
        self._strip_punctuation = strip_punctuation
        self._sentence_tokenize = sentence_tokenize
        self._word_tokenize = word_tokenize
        self._lower = lower

        # create document processing partial function
        self.process_document: Callable[..., List[str]] = partial(
            tokenize,
            stopwords=self._stopwords,
            strip_accents=self._strip_accents,
            strip_punctuation=self._strip_punctuation,
            sentence_tokenize=self._sentence_tokenize,
            word_tokenize=self._word_tokenize,
            lower=self._lower,
        )


class NGrams(Layer):
    """
    This class represents a layer that creates n-grams from the input. It
    inherits from the Layer class and overrides the constructor to handle
    n-grams parameters.

    Parameters
    ----------
    n : int, optional
        The size of the n-grams to create. Default is 2.
    lower : bool, optional
        If True, the input will be converted to lower case during tokenization.
    name : str, optional
        The name of the layer. If not given, the name will be derived from the
        class name.
    verbose : bool, optional
        If True, the layer will output verbose messages during execution.
        Default is True.

    """

    _trainable = False

    def __init__(
        self,
        n: int = 2,
        lower: Optional[bool] = True,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self._n = n
        self._lower = lower

    def process_document(self, text: List[str]) -> List[str]:
        """
        Process a document by creating n-grams from it.

        Parameters
        ----------
        text : list
            The document to process, as a list of words.

        Returns
        -------
        ngrams : list
            The n-grams of the document, as a list of words.
        """
        if self._lower:
            text = text.lower()
        return [text[i : i + self._n] for i in range(len(text) - self._n + 1)]


class SnowballStem(Layer):
    """
    This class represents a layer that applies Snowball stemming to the input.
    It inherits from the Layer class and overrides the constructor to handle
    stemming parameters.

    Parameters
    ----------
    lang : str, optional
        The language of the input. Default is 'en' (English).
        Supported languages: 'fr' (French), 'de' (German), 'es' (Spanish),
        'it' (Italian), 'pt' (Portuguese), 'nl' (Dutch), 'sv' (Swedish),
        'no' (Norwegian), 'da' (Danish), 'fi' (Finnish), 'hu' (Hungarian),
        'ro' (Romanian), 'ru' (Russian).
    memoize : bool, optional
        If True, stems will be memoized to speed up repeated stemming of the
        same word. Default is True.
    name : str, optional
        The name of the layer. If not given, the name will be derived from
        the class name.
    verbose : bool, optional
        If True, the layer will output verbose messages during execution.
        Default is True.

    """

    _lang_mapping: Dict[str, str] = {
        "fr": "french",
        "en": "english",
        "de": "german",
        "es": "spanish",
        "it": "italian",
        "pt": "portuguese",
        "nl": "dutch",
        "sv": "swedish",
        "no": "norwegian",
        "da": "danish",
        "fi": "finnish",
        "hu": "hungarian",
        "ro": "romanian",
        "ru": "russian",
    }

    def __init__(
        self,
        lang: str = "en",
        memoize: bool = True,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)

        self._lang: str = self._lang_mapping.get(lang, lang)
        self._memoize: bool = memoize
        self._reload()

    def _unload(self) -> None:
        del self._stemmer
        self._word2stem: Dict[str, str] = {}

    def _reload(self, **_: Any) -> None:
        from nltk.stem.snowball import SnowballStemmer

        self._stemmer = SnowballStemmer(self._lang)
        self._word2stem: Dict[str, str] = {}

    def process_document(self, text: List[str]) -> List[str]:
        """
        Process a document by applying stemming to it.

        Parameters
        ----------
        text : list
            The document to process, as a list of words.

        Returns
        -------
        stemmed : list
            The stemmed document, as a list of words.
        """
        if self._memoize:
            words: List[str] = []
            for w in text:
                stem: Optional[str] = self._word2stem.get(w, None)
                if stem is None:
                    stem = self._stemmer.stem(w)
                    self._word2stem[w] = stem
                    words.append(stem)
                else:
                    words.append(stem)
            return words
        else:
            return [self._stemmer.stem(w) for w in text]


class TokenMonster(Layer):
    def __init__(
        self,
        model="english-32000-balanced-v1",
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        import tokenmonster

        super().__init__(name, verbose)
        self.vocab = tokenmonster.load(model)
        self.n_features = len(self.vocab)

    def process_document(self, text: str) -> np.ndarray:
        return self.vocab.tokenize(text)

    def decode(self, tokens: np.ndarray) -> str:
        return self.vocab.decode(tokens)
