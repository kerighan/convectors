import re

from . import Layer

# =============================================================================
# Layers
# =============================================================================


class SplitHashtag(Layer):
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
        super(SplitHashtag, self).__init__(
            input, output, name, verbose, parallel)

        # create document processing partial function
    def process_doc(self, doc):
        return " ".join([h for h in re.split('([A-Z][a-z]+)', doc) if h])
