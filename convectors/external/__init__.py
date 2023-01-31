from .. import Layer


class NLPLayer(Layer):
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
        super(NLPLayer, self).__init__(
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


class Summarize(NLPLayer):
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


class NER(NLPLayer):
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


class SentenceTransformer(NLPLayer):
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


class Sentiment(NLPLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=True
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
        self.nlp = pipeline("sentiment-analysis",
                            model=model, tokenizer=tokenizer)

    def process_doc(self, doc):
        try:
            item = self.nlp(doc)[0]
            label = item["label"]
            return int(label[:1])
        except RuntimeError:
            return None


class FlairNER(NLPLayer):
    def __init__(
        self,
        input=None,
        output=None,
        simplified=True,
        name=None,
        lang="fr",
        verbose=True,
        document_wise=True
    ):
        self.lang = lang
        super(FlairNER, self).__init__(
            input, output, name, verbose, document_wise)
        self.simplified = simplified

    def reload(self):
        from flair.data import Sentence
        from flair.models import SequenceTagger
        if self.lang == "fr":
            self.nlp = SequenceTagger.load("flair/ner-french")
        self.sentence = Sentence

    def process_doc(self, doc):
        sent = self.sentence(doc)
        self.nlp.predict(sent)
        return [(it.text, it.tag) for it in sent.get_spans("ner")]

    def process_series(self, series):
        if not isinstance(series, list):
            series = list(series)
        res = self.nlp(series)
        if not self.simplified:
            return res
        return [
            [(it["word"], it["entity_group"]) for it in doc]
            for doc in res]


class T5Translate(NLPLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=True,
        target="French"
    ):
        super().__init__(
            input, output, name, verbose, document_wise)
        self.tgt = target

    def reload(self):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-large", return_dict=True)

    def process_doc(self, doc):
        src = "French"
        # tgt = "German"
        inp = f"translate {src} to {self.tgt}: " + doc
        print(inp)
        input_ids = self.tokenizer(inp, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        decoded = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        print(decoded)
        print()
        return decoded


class OpusTranslate(NLPLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=True,
        src="en",
        tgt="fr"
    ):
        self.src = src
        self.tgt = tgt
        super().__init__(
            input, output, name, verbose, document_wise)

    def reload(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model_name = f"Helsinki-NLP/opus-mt-{self.src}-{self.tgt}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)

    def process_doc(self, doc):
        try:
            inputs = self.tokenizer(doc, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            decoded = self.tokenizer.decode(outputs[0],
                                            skip_special_tokens=True)
            return decoded
        except Exception:
            return doc


class OpusTranslateBatch(NLPLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=False,
        src="en",
        tgt="fr",
        batch_size=50
    ):
        self.src = src
        self.tgt = tgt
        self.batch_size = batch_size
        super().__init__(
            input, output, name, verbose, document_wise)

    def reload(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        model_name = f"Helsinki-NLP/opus-mt-{self.src}-{self.tgt}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # self.model.to(self.device)
        self.tokenizer.max_length = 512
        self.model.max_length = 512

    def process_series(self, series):
        from more_itertools import chunked
        from tqdm import tqdm

        res = []
        total = len(series)//self.batch_size + int(
            (len(series) % self.batch_size) > 0)
        for docs in tqdm(chunked(series, n=self.batch_size), total=total):
            try:
                inputs = self.tokenizer(
                    docs, return_tensors="pt", padding=True)
                outputs = self.model.generate(**inputs)
                decoded = [self.tokenizer.decode(out, skip_special_tokens=True)
                           for out in outputs]
            except Exception as e:
                print(e)
                decoded = []*len(docs)
            res += decoded
        return res


class Tiktokenize(Layer):
    parallel = False
    trainable = False
    document_wise = True

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        document_wise=True,
        encoding="gpt2"
    ):
        super().__init__(
            input, output, name, verbose, False)
        self.document_wise = document_wise
        self.encoding = encoding
        self.reload()

    def reload(self):
        import tiktoken
        self.enc = tiktoken.get_encoding(self.encoding)
        self.n_features = self.enc.n_vocab

    def process_doc(self, doc):
        res = self.enc.encode(doc, allowed_special={'<|endoftext|>'})
        res = [it + 1 for it in res]
        return res

    def decode(self, doc):
        return self.enc.decode([it - 1 for it in doc])

    def unload(self):
        del self.enc
