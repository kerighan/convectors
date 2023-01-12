import random

from .. import Layer


class Mask(Layer):
    parallel = True
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        p=.1,
        use_frequency=True,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Mask, self).__init__(input, output, name, verbose, parallel)
        self.p = p
        # if use_frequency:
        #     self.document_wise = False

    def process_doc(self, doc):
        doc = [item if random.random() > self.p else "<MASK>" for item in doc]
        return doc

    def process_series(self, series):
        import itertools
        from collections import Counter

        import numpy as np
        count = Counter(itertools.chain(*series))

        docs, masks = [], []
        for doc in series:
            if len(doc) == 0:
                continue
            scores = [1/np.log(count[token]+1) for token in doc]
            scores = np.array(scores) / np.sum(scores)
            mask_index = np.random.choice(range(len(doc)), p=scores)
            new_doc = doc.copy()
            new_doc[mask_index] = "<MASK>"
            docs.append(new_doc)
            masks.append(doc[mask_index])
        return docs, masks


class TeacherForce(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        multiple_tokens=False,
        name=None,
        verbose=True,
        parallel=False
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.multiple_tokens = multiple_tokens

    def process_series(self, series):
        X, y = [], []

        for doc in series:
            for i in range(1, len(doc)):
                X.append(doc[:i])
                if self.multiple_tokens:
                    y.append(doc[i:])
                else:
                    y.append(doc[i])
        return X, y
