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
        self.reload()

    def process_doc(self, doc):
        return self.nlp(doc)

    def process_series(self, series):
        if not isinstance(series, list):
            series = list(series)
        return self.nlp(series)

    def unload(self):
        del self.nlp


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
        super(NER, self).__init__(
            input, output, name, verbose, document_wise)
        self.simplified = simplified

    def reload(self):
        from transformers import (AutoModelForTokenClassification,
                                  AutoTokenizer, pipeline)
        tokenizer = AutoTokenizer.from_pretrained(
            "Jean-Baptiste/camembert-ner-with-dates")
        model = AutoModelForTokenClassification.from_pretrained(
            "Jean-Baptiste/camembert-ner-with-dates")
        self.nlp = pipeline("ner", model=model,
                            tokenizer=tokenizer,
                            aggregation_strategy="simple")

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


class Summarize(HuggingFaceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=True
    ):
        super().__init__(input, output, name, verbose, document_wise)

    def reload(self):
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                  SummarizationPipeline)
        mdl = "lincoln/mbart-mlsum-automatic-summarization"
        tokenizer = AutoTokenizer.from_pretrained(mdl)
        model = AutoModelForSeq2SeqLM.from_pretrained(mdl)
        self.nlp = SummarizationPipeline(
            model=model, tokenizer=tokenizer)

    def process_doc(self, doc):
        return self.nlp(doc)

    def process_series(self, series):
        if not isinstance(series, list):
            series = list(series)
        res = self.nlp(series)
        return res
