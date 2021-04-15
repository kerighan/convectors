from .. import Layer
from functools import partial
import re


# =============================================================================
# Layers
# =============================================================================

class Tokenize(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        stopwords=None,
        strip_accents=False,
        strip_punctuation=True,
        lower=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Tokenize, self).__init__(input, output, name, verbose, parallel)
        if stopwords is not None:
            if isinstance(stopwords, set):
                self.stopwords = stopwords
            else:
                from .stopwords import stopwords as sw
                self.stopwords = set()
                for item in stopwords:
                    self.stopwords.update(sw[item])
        else:
            self.stopwords = stopwords
        self.strip_accents = strip_accents
        self.strip_punctuation = strip_punctuation
        self.lower = lower

        # create document processing partial function
        self.process_doc = partial(
            tokenize,
            stopwords=self.stopwords,
            strip_accents=self.strip_accents,
            strip_punctuation=self.strip_punctuation,
            lower=self.lower)


class Snowball(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        lang="french",
        memoize=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Snowball, self).__init__(input, output, name, verbose, parallel)
        from nltk.stem.snowball import SnowballStemmer
        self.stemmer = SnowballStemmer(lang)
        self.memoize = memoize
        self.word2stem = {}

    def process_doc(self, text):
        if self.memoize:
            words = []
            for w in text:
                stem = self.word2stem.get(w, None)
                if stem is None:
                    stem = self.stemmer.stem(w)
                    self.word2stem[w] = stem
                    words.append(stem)
                else:
                    words.append(stem)
            return words
        else:
            return [self.stemmer.stem(w) for w in text]


class Phrase(Layer):
    parallel = True
    trainable = True

    def __init__(
        self,
        input=None,
        output=None,
        min_count=3,
        threshold=3,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Phrase, self).__init__(input, output, name, verbose, parallel)
        self.min_count = min_count
        self.threshold = threshold

    def fit(self, series):
        self.pmi = set(pmi(
            series, window_size=1,
            minimum=self.threshold,
            threshold=self.min_count).keys())
        self.trained = True

    def process_doc(self, text):
        if len(text) == 0:
            return []

        last_word = text[0]
        chain = [last_word]
        words = []
        for i in range(1, len(text)):
            word = text[i]
            if (last_word, word) in self.pmi:
                chain.append(word)
            else:
                words.append("_".join(chain))
                chain = [word]
            last_word = word
        if len(chain) != 0:
            words.append("_".join(chain))
        return words


# =============================================================================
# Functions
# =============================================================================

def tokenize(
    text,
    stopwords=None,
    strip_accents=False,
    strip_punctuation=True,
    lower=True,
):
    text = str(text)
    # turn lowercase
    if lower:
        text = text.lower()
    # remove accents
    if strip_accents:
        from unidecode import unidecode
        text = unidecode(text)
    if strip_punctuation:
        # tokenize by removing punctuation
        words = re.findall(
            r"([\w]+|['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|"
            r"'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF'])",
            text, re.UNICODE)
    else:
        # tokenize by keeping punctuation
        words = re.findall(
            r"([\w]+|['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|"
            r"'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']|"
            r"[,?;.:\/!()\[\]'\"’\\><+-=])",
            text, re.UNICODE)
    # remove stopwords
    if stopwords is not None:
        words = [w for w in words if w not in stopwords]
    return words


def pmi(series, window_size=3, threshold=2, minimum=0.6):
    from collections import Counter, defaultdict
    import itertools
    import math

    # compute freq
    freq = Counter(itertools.chain(*series))
    N = len(series)
    M = N * window_size

    cooc = defaultdict(int)
    for words in series:
        windows = zip(*[words[i:] for i in range(window_size + 1)])
        for window in windows:
            source = window[window_size]
            if source is None:
                continue
            for pos, target in enumerate(window):
                if pos == window_size or target is None or target == source:
                    continue

                couple = (target, source)
                cooc[couple] += 1

    npmi_ = {}
    for couple, count in cooc.items():
        if count < threshold:
            continue
        x, y = couple
        p_x = freq[x]
        p_y = freq[y]
        p_xy = count / M
        prod = p_x * p_y / (N*N)
        npmi_value = math.log(p_xy / prod)
        if npmi_value > minimum:
            npmi_[couple] = npmi_value
    return npmi_