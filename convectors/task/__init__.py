import random

from .. import Layer


class Mask(Layer):
    parallel = True
    trainable = False
    document_wise = True

    def __init__(
        self,
        input=None,
        output=None,
        p=.1,
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Mask, self).__init__(input, output, name, verbose, parallel)
        self.p = p

    def process_doc(self, doc):
        doc = [item if random.random() > self.p else "<MASK>" for item in doc]
        return doc
