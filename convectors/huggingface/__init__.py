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


class Summarize(HuggingFaceLayer):
    parallel = False
    trainable = False
    document_wise = True

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
    ):
        super().__init__(
            input, output, name, verbose)

    def reload(self):
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                  SummarizationPipeline)
        model_name = 'lincoln/mbart-mlsum-automatic-summarization'
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
        loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.nlp = SummarizationPipeline(
            model=loaded_model, tokenizer=loaded_tokenizer)


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


class NER2(HuggingFaceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        simplified=True,
        name=None,
        verbose=True,
        document_wise=True
    ):
        super(NER2, self).__init__(
            input, output, name, verbose, document_wise)
        self.simplified = simplified

    def reload(self):
        from flair.data import Sentence
        from flair.models import SequenceTagger

        self.nlp = SequenceTagger.load("flair/ner-french")
        self.sentence = Sentence

    def process_doc(self, doc):
        res = self.nlp.predict(self.sentence(doc))
        if not self.simplified:
            return res
        return [it for it in res.get_spans("ner")]

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


class SentenceTransformer(HuggingFaceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        lang=None,
        verbose=True,
        document_wise=False
    ):
        self.lang = lang
        super().__init__(
            input, output, name, verbose, document_wise)

    def reload(self):
        from sentence_transformers import SentenceTransformer as ST
        if self.lang == "fr":
            # self.nlp = ST('lincoln/flaubert-mlsum-topic-classification')
            self.nlp = ST('flaubert/flaubert_base_cased')
        else:
            self.nlp = ST(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def process_series(self, series):
        if not isinstance(series, list):
            series = list(series)
        return self.nlp.encode(series)


class Sentiment(HuggingFaceLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=False
    ):
        super().__init__(
            input, output, name, verbose, document_wise)

    def reload(self):
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, pipeline)

        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment")

        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment")
        self.nlp = pipeline("sentiment-analysis", model=model,
                            tokenizer=tokenizer)

    def process_series(self, series):
        if not isinstance(series, list):
            series = list(series)
        res = self.nlp(series)
        stars = []
        for item in res:
            label = item["label"]
            stars.append(int(label[:1]))
        return stars
