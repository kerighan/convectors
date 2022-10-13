import itertools
import re
from functools import partial

from .. import Layer

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
        sentence_tokenize=False,
        word_tokenize=True,
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
        self.sentence_tokenize = sentence_tokenize
        self.word_tokenize = word_tokenize
        self.lower = lower

        # create document processing partial function
        self.process_doc = partial(
            tokenize,
            stopwords=self.stopwords,
            strip_accents=self.strip_accents,
            strip_punctuation=self.strip_punctuation,
            sentence_tokenize=self.sentence_tokenize,
            word_tokenize=self.word_tokenize,
            lower=self.lower)


class BPETokenize(Layer):
    parallel = False
    trainable = True
    document_wise = True

    def __init__(self,
                 input=None,
                 output=None,
                 max_features=10000,
                 name=None,
                 verbose=True,
                 as_int=False,
                 pct_bpe=.88,
                 parallel=False):
        super().__init__(input, output, name, verbose, parallel)
        self.max_features = max_features
        self.as_int = as_int
        self.pct_bpe = pct_bpe

    def fit(self, series, *args, y=None):
        from bpe import Encoder
        self.encoder = Encoder(self.max_features, pct_bpe=self.pct_bpe)
        self.encoder.fit(list(series))
        self.n_features = self.encoder.vocab_size

    def process_doc(self, doc):
        if self.as_int:
            return next(self.encoder.transform([doc]))
        return self.encoder.tokenize(doc)


class Snowball(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        lang="fr",
        memoize=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Snowball, self).__init__(input, output, name, verbose, parallel)

        self.translate = {
            "fr": "french",
            "en": "english",
            "de": "german"
        }
        self.lang = self.translate.get(lang, lang)
        self.reload()
        self.memoize = memoize
        self.word2stem = {}

    def unload(self):
        del self.stemmer
        self.word2stem = {}

    def reload(self, **_):
        from nltk.stem.snowball import SnowballStemmer
        self.stemmer = SnowballStemmer(self.lang)

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


class Lemmatize(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        lang="fr",
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Lemmatize, self).__init__(
            input, output, name, verbose, parallel)

        self.lang = lang
        self.word2stem = {}
        self.reload()

    def unload(self):
        self.word2stem = {}

    def reload(self, **_):
        import os
        import pickle

        db_fn = os.path.join(
            os.path.dirname(__file__),
            f"../ressources/lemma/{self.lang}_lemma.p")
        with open(db_fn, "rb") as f:
            self.word2stem = pickle.load(f)

    def stem(self, w):
        lemma = self.word2stem.get(w, w)
        return lemma

    def process_doc(self, text):
        if len(text) == 0:
            return []
        if isinstance(text[0], str):
            return [self.stem(w) for w in text]
        return [[self.stem(w) for w in s] for s in text]


class NER(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        lang="en",
        include_tag=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(NER, self).__init__(
            input, output, name, verbose, parallel)

        self.lang = lang
        self.include_tag = include_tag
        self.reload()

    def unload(self):
        if hasattr(self, "word2ner"):
            del self.word2ner
        if hasattr(self, "kp"):
            del self.kp

    def reload(self, **_):
        import os
        import pickle

        from flashtext import KeywordProcessor

        db_fn = os.path.join(
            os.path.dirname(__file__),
            f"../ressources/ner/{self.lang}_ner.p")
        with open(db_fn, "rb") as f:
            self.word2ner = pickle.load(f)

        self.kp = KeywordProcessor(case_sensitive=True)
        for key in self.word2ner:
            self.kp.add_keyword(key)

    def ner(self, text):
        found = self.kp.extract_keywords(text)
        if self.include_tag:
            return [(x, self.word2ner[x]) for x in found]
        return found

    def process_doc(self, text):
        if len(text) == 0:
            return []
        if isinstance(text[0], str):
            return self.ner(text)
        return [[self.ner(w) for w in s] for s in text]


Lemmatizer = Lemmatize


class StanzaStemmer(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        lang="fr",
        name=None,
        verbose=True,
        parallel=False
    ):
        super(StanzaStemmer, self).__init__(
            input, output, name, verbose, parallel)

        self.lang = lang
        self.reload()

    def unload(self):
        del self.stemmer

    def reload(self, **_):
        import stanza
        try:
            self.stemmer = stanza.Pipeline(
                self.lang, processors='tokenize,mwt,pos,lemma')
        except stanza.pipeline.core.ResourcesFileNotFoundError:
            stanza.download(self.lang)
            self.stemmer = stanza.Pipeline(
                self.lang, processors='tokenize,mwt,pos,lemma')

    def process_doc(self, text):
        res = self.stemmer(text)
        return [word.lemma for sent in res.sentences for word in sent.words]


class Phrase(Layer):
    parallel = True
    trainable = True

    def __init__(
        self,
        input=None,
        output=None,
        min_cooc=10,
        threshold=.7,
        max_len=float("inf"),
        join_char=" ",
        bigram=False,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Phrase, self).__init__(input, output, name, verbose, parallel)
        self.min_cooc = min_cooc
        self.threshold = threshold
        self.bigram = bigram
        self.join_char = join_char
        self.max_len = max_len
        if bigram:
            self.process_doc = self.process_bigram
        else:
            self.process_doc = self.process_chain

    def fit(self, series, *args, y=None):
        self.pmi = set(pmi(
            series, window_size=2,
            threshold=self.threshold,
            min_cooc=self.min_cooc, undirected=False,
            normalize=True).keys())

    def process_bigram(self, text):
        skip = False
        res = []
        for word_a, word_b in zip(text[:-1], text[1:]):
            if skip:
                skip = False
                continue
            elif (word_a, word_b) in self.pmi:
                res.append(f"{word_a}{self.join_char}{word_b}")
                skip = True
            else:
                res.append(word_a)
        return res

    def process_chain(self, text):
        if len(text) == 0:
            return []

        last_word = text[0]
        chain = [last_word]
        words = []
        for i in range(1, len(text)):
            word = text[i]
            if (last_word, word) in self.pmi:
                chain.append(word)
            elif (word, last_word) in self.pmi:
                chain.append(word)
            else:
                if len(chain) >= self.max_len:
                    continue
                words.append(self.join_char.join(chain))
                chain = [word]
            last_word = word
        if 0 < len(chain) < self.max_len:
            words.append(self.join_char.join(chain))
        return words


class Sub(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        regex="url",
        replacement=" ",
        name=None,
        verbose=True,
        parallel=False
    ):
        from .utils import name2regex
        super(Sub, self).__init__(input, output, name, verbose, parallel)
        self.regex = re.compile(name2regex.get(regex, regex))
        self.replacement = replacement

    def process_doc(self, doc):
        return re.sub(self.regex, self.replacement, doc)


class FindAll(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        regex="emoji",
        name=None,
        verbose=True,
        parallel=False
    ):
        from .utils import name2regex
        super(FindAll, self).__init__(input, output, name, verbose, parallel)
        self.regex = re.compile(name2regex.get(regex, regex))

    def process_doc(self, doc):
        return re.findall(self.regex, doc)


class NGram(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        ngram=2,
        join=True,
        lower=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(NGram, self).__init__(input, output, name, verbose, parallel)
        self.ngram = ngram
        self.join = join
        self.lower = lower

    def process_doc(self, doc):
        if self.lower:
            doc = doc.lower()
        if self.join:
            return ngram(doc, n=self.ngram)
        return list(zip(*(doc[i:] for i in range(self.ngram))))


class Contract(Layer):
    parallel = True
    trainable = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Contract, self).__init__(input, output, name, verbose, parallel)

    def process_doc(self, doc):
        if not isinstance(doc, str):
            doc = str(doc)
        return re.sub(r"(.)\1{2,}", r"\1", doc)


class LangDetect(Layer):
    parallel = True
    trainable = False
    document_wise = True

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(LangDetect, self).__init__(
            input, output, name, verbose, parallel)
        from langdetect import detect
        from langdetect.lang_detect_exception import LangDetectException
        self.detect = detect
        self.exception = LangDetectException

    def process_doc(self, doc):
        try:
            return self.detect(doc)
        except self.exception:
            return None


class NearDuplicates(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        threshold=.85,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.threshold = threshold

    def process_series(self, documents):
        from collections import Counter

        from scipy.sparse.csgraph import connected_components

        from ..embedding import TfIdf
        nlp = NGram(ngram=3)
        nlp += TfIdf(max_features=50000)
        nlp.verbose = False
        X = nlp(documents.str.lower())
        A = X @ X.T
        A.setdiag(0)
        A = A >= self.threshold
        _, labels = connected_components(
            csgraph=A, directed=False, return_labels=True)
        counter = Counter(labels)
        label2bucket = {}
        for i, (label, occ) in enumerate(counter.most_common()):
            if occ == 1:
                break
            label2bucket[label] = i
        labels = [label2bucket.get(label, -1) for label in labels]
        return labels


class CountFilter(Layer):
    parallel = True
    trainable = True
    document_wise = True

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        min_df=None,
        max_df=None,
        min_tf=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.max_df = max_df
        self.min_df = min_df
        self.min_tf = min_tf

    def fit(self, series, y=None):
        from collections import Counter
        tf = Counter()
        df = Counter()
        n_tokens = 0
        n_docs = len(series)

        for doc in series:
            visited = set()
            n_tokens += len(doc)
            for token in doc:
                if token not in visited:
                    df[token] += 1
                tf[token] += 1

        self.tf = tf
        self.df = df
        if isinstance(self.max_df, float):
            self.max_df = int(round(self.max_df * n_docs))
        if isinstance(self.min_df, float):
            self.min_df = int(round(self.min_df * n_docs))

    def process_doc(self, doc):
        res = []
        for token in doc:
            token_df = self.df[token]
            if self.max_df is not None:
                if token_df > self.max_df:
                    continue
            if self.min_df is not None:
                if token_df < self.min_df:
                    continue
            if self.min_tf is not None:
                if self.tf[token] < self.min_tf:
                    continue
            res.append(token)
        return res


# =============================================================================
# Functions
# =============================================================================


def words_from(text):
    return re.findall(
        r"([\w]+|[\U0001F300-\U0001F5FF|\U0001F600-\U0001F64F|"
        r"\U0001F680-\U0001F6FF|\u2600-\u26FF\u2700-\u27BF])",
        text, re.UNICODE)


def words_and_punctuation_from(text):
    return re.findall(
        r"([\w]+|[\U0001F300-\U0001F5FF|\U0001F600-\U0001F64F|"
        r"\U0001F680-\U0001F6FF|\u2600-\u26FF\u2700-\u27BF]|"
        r"[,?;.:\/!()\[\]'\"’\\><+-=])",
        text, re.UNICODE)


def sentences_from(text):
    return re.split(
        r'(?<!\w[\t\r!:.?|•]\w.)(?<![A-Z][a-z][.])'
        r'(?<=[\t\r!:.?|•…])\s', text)


def tokenize(
    text,
    stopwords=None,
    strip_accents=False,
    strip_punctuation=True,
    sentence_tokenize=False,
    word_tokenize=True,
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
    if not sentence_tokenize:
        if strip_punctuation:
            # tokenize by removing punctuation
            words = words_from(text)
        else:
            # tokenize by keeping punctuation
            words = words_and_punctuation_from(text)
    elif word_tokenize:
        words = []
        sentences = sentences_from(text)
        if strip_punctuation:
            words = [words_from(s) for s in sentences]
        else:
            words = [words_and_punctuation_from(s) for s in sentences]
    else:
        return sentences_from(text)

    # remove stopwords
    if stopwords is not None:
        if not sentence_tokenize:
            words = [w.strip() for w in words if w.strip() not in stopwords]
        else:
            words = [[w.strip() for w in sentence
                      if w.strip() not in stopwords]
                     for sentence in words]
    return words


def pmi(
    series, window_size=3, undirected=False, min_cooc=2, k=1,
    threshold=None, normalize=True
):
    from ..collections import PairCounter
    pair_counts = PairCounter(undirected=undirected)
    pair_counts.count(series)
    for doc in series:
        for window in ngram(doc, window_size, func=lambda x: x):
            src = window[0]
            for i in range(1, len(window)):
                tgt = window[i]
                pair_counts.increment(src, tgt)
    edges = pair_counts.get_pmi(
        normalize=normalize,
        threshold=threshold,
        min_cooc=min_cooc,
        k=k)
    return edges


def cooc(series, window_size=3, undirected=False):
    from collections import defaultdict

    cooc_ = defaultdict(int)
    for words in series.tolist():
        for i in range(len(words)):
            source = words[i]
            length = min(len(words) - i, window_size)
            for j in range(i+1, i + length):
                target = words[j]

                if undirected:
                    if source < target:
                        couple = (source, target)
                    else:
                        couple = (target, source)
                else:
                    couple = (source, target)
                cooc_[couple] += 1
    return cooc_


def ngram(xs, n=3, func=lambda x: "".join(x)):
    ts = itertools.tee(xs, n)
    for i, t in enumerate(ts[1:]):
        for _ in range(i + 1):
            next(t, None)
    return [func(it) for it in zip(*ts)]


def pmi_graph(
    series, window_size=5, min_count=2, threshold=.1, remove_isolates=True
):
    from collections import Counter

    import networkx as nx
    triplets = pmi(series, window_size=window_size, min_count=min_count,
                   undirected=True, normalize=True, minimum=threshold)

    counts = Counter(itertools.chain(*series))

    edges = []
    for (a, b), w in triplets.items():
        edges.append((a, b, w))
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    if not remove_isolates:
        G.add_nodes_from(counts.keys())

    sizes = {}
    for node in G.nodes:
        sizes[node] = counts[node]
    nx.set_node_attributes(G, sizes, name="size")
    return G


def cooc_graph(
    series, window_size=5, min_count=2, remove_isolates=True
):
    from collections import Counter

    import networkx as nx
    triplets = cooc(series, window_size=window_size, undirected=True)

    counts = Counter(itertools.chain(*series))

    edges = []
    for (a, b), w in triplets.items():
        if w > min_count:
            edges.append((a, b, w))
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    if not remove_isolates:
        G.add_nodes_from(counts.keys())

    sizes = {}
    for node in G.nodes:
        sizes[node] = counts[node]
    nx.set_node_attributes(G, sizes, name="size")
    return G
