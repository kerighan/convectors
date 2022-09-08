from collections import Counter, defaultdict

from numpy import exp, sqrt


class PairCounter:
    def __init__(self, undirected=True):
        from collections import defaultdict
        self.undir = undirected
        self.pair_counts = defaultdict(int)
        self.pair_total = 0
        self.counts = defaultdict(int)
        self.counts_total = 0

    def count(self, series):
        import itertools
        from collections import Counter
        self.counts = Counter(itertools.chain(*series))
        self.counts_total = sum(self.counts.values())

    def get_pair(self, a, b):
        if self.undir:
            pair = (a, b) if a < b else (b, a)
        else:
            pair = (a, b)
        return pair

    def increment(self, a, b):
        pair = self.get_pair(a, b)
        self.pair_counts[pair] += 1
        self.pair_total += 1

    def get(self, a, b=None):
        if b is not None:
            pair = self.get_pair(a, b)
            return self.pair_counts[pair]
        return self.counts[a]

    def get_pmi(
        self,
        normalize=False,
        threshold=0,
        min_cooc=2,
        k=1,
        ignore_self=True
    ):
        import numpy as np

        edges = {}
        for (a, b), f_ab in self.pair_counts.items():
            if f_ab == 0 or f_ab < min_cooc or (ignore_self and (a == b)):
                continue
            f_a = self.counts[a] / self.counts_total
            f_b = self.counts[b] / self.counts_total
            f_ab /= self.pair_total
            _pmi = np.log(f_ab**k / (f_a * f_b))
            if normalize:
                _pmi /= -np.log(f_ab)
                _pmi = min(1, _pmi)

            if threshold is None:
                edges[a, b] = _pmi
            elif _pmi > threshold:
                edges[a, b] = _pmi
        return edges

    def __repr__(self):
        return str(self.pair_counts)


class OddsCounter:
    def __init__(self, ci=.95):
        from scipy.stats import norm as scipy_norm
        self.counts = {}
        self.counts_total = defaultdict(int)
        self.global_counter = Counter()
        self.global_total = 0
        self._current_key = None
        self.ci_factor = -scipy_norm.isf((1 - ci) / 2.)

    def __getitem__(self, key):
        assert not isinstance(key, tuple)

        self._current_key = key

        if key not in self.counts:
            self.counts[key] = Counter()
        return self

    def __setitem__(self, key, _):
        assert not isinstance(key, tuple)
        self._current_key = key

    def __iadd__(self, tokens):
        key = self._current_key

        if key not in self.counts:
            self.counts[key] = Counter()
        counter = self.counts[key]

        for token in tokens:
            counter[token] += 1
            self.counts_total[key] += 1
            self.global_counter[token] += 1
            self.global_total += 1

    def compute_odds(self, a, b, c, d):
        eps = 1
        b = max(b - a, eps)
        d = max(d - c, eps)
        c = max(c, eps)
        odds_ratio = (a / b) / (c / d)

        uncertainty = sqrt(1/a + 1/b + 1/c + 1/d)
        uncertainty = exp(self.ci_factor*uncertainty)
        odds_ratio *= uncertainty
        return odds_ratio

    def get_keywords(
        self, topic, exclude=True, return_score=False, min_tf=1
    ):
        count_total = self.counts_total[topic]
        res = []
        for token, tf in self.counts[topic].items():
            if tf < min_tf:
                continue
            if exclude:
                odds = self.compute_odds(tf, count_total,
                                         self.global_counter[token] - tf,
                                         self.global_total - count_total)
            else:
                odds = self.compute_odds(tf, count_total,
                                         self.global_counter[token],
                                         self.global_total)
            if odds > 1:
                res.append((token, odds))
        res = sorted(res, key=lambda x: x[1], reverse=True)
        if not return_score:
            res = [a for a, _ in res]
        return res
