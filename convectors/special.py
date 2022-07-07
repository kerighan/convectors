import re

from . import Layer, to_matrix

# =============================================================================
# Layers
# =============================================================================


class SplitHashtag(Layer):
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
        super(SplitHashtag, self).__init__(
            input, output, name, verbose, parallel)

        # create document processing partial function
    def process_doc(self, doc):
        return re.sub(
            r'#[a-z]\S*',
            lambda m: ' '.join(re.findall(
                '[A-Z][^A-Z]*|[a-z][^A-Z]*', m.group().lstrip('#'))),
            doc,
        )


class Lambda(Layer):
    parallel = True
    trainable = False
    document_wise = True

    def __init__(
        self,
        func,
        input=None,
        output=None,
        name=None,
        verbose=True,
        parallel=False,
        axis=1
    ):
        super(Lambda, self).__init__(
            input, output, name, verbose, parallel)
        if axis == 1:
            self.process_doc = func
        elif axis == 0:
            self.document_wise = False
            self.process_series = func
        else:
            raise ValueError("axis can only be 0 or 1")


class DomainName(Layer):
    parallel = True
    trainable = False
    document_wise = True

    def __init__(self, input=None, output=None, verbose=False, name=None,
                 parallel=False):
        """
        Returns URL base source in urls
        """
        super(DomainName, self).__init__(
            input, output, name, verbose, parallel)

    def process_doc(self, u):
        if isinstance(u, list):
            a = [re.findall(
                r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)', url)
                for url in u
            ]
            return [item for liste in a for item in liste]
        else:
            res = re.findall(
                r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)', u)
            if len(res) == 1:
                return res[0]
            elif len(res) == 0:
                return
            return res


class Argmax(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        parallel=False,
        axis=-1
    ):
        super().__init__(input, output, name, verbose, parallel)
        self.axis = axis

    def process_series(self, series):
        return to_matrix(series).argmax(axis=self.axis)
