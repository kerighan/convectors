from numpy.lib.arraysetops import isin

from .. import Layer


class HuggingFaceLayer(Layer):
    parallel = False
    trainable = False
    document_wise = True

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=True
    ):
        super(HuggingFaceLayer, self).__init__(
            input, output, name, verbose, False)
        self.document_wise = document_wise

    def process_doc(self, doc):
        return self.nlp(doc)

    def process_series(self, series):
        if not isinstance(series, list):
            series = list(series)
        return self.nlp(series)


class NER(HuggingFaceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        simplified=True,
        name=None,
        verbose=True,
        document_wise=True
    ):
        from transformers import (AutoModelForTokenClassification,
                                  AutoTokenizer, pipeline)
        super(NER, self).__init__(
            input, output, name, verbose, document_wise)

        tokenizer = AutoTokenizer.from_pretrained(
            "Jean-Baptiste/camembert-ner-with-dates")
        model = AutoModelForTokenClassification.from_pretrained(
            "Jean-Baptiste/camembert-ner-with-dates")
        self.nlp = pipeline("ner", model=model,
                            tokenizer=tokenizer, aggregation_strategy="simple")
        self.simplified = simplified

    def process_doc(self, doc):
        res = self.nlp(doc)
        if not self.simplified:
            return res
        return [(it["word"], it["entity_group"]) for it in res]

    def process_series(self, series):
        if not isinstance(series, list):
            series = list(series)
        res = self.nlp(series)
        if not self.simplified:
            return res
        return [
            [(it["word"], it["entity_group"]) for it in doc]
            for doc in res
        ]
