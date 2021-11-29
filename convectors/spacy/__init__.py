from .. import Layer


class Stemmer(Layer):
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
        super(Stemmer, self).__init__(
            input, output, name, verbose, parallel)

        self.lang = lang
        self.reload()

    def unload(self):
        del self.stemmer

    def reload(self, **_):
        import spacy
        self.nlp = spacy.load(f"{self.lang}_core_news_sm")

    def process_doc(self, text):
        doc = self.nlp(text)
        return [word.lemma_ for word in doc]


class LangDetect(Layer):
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
        super(LangDetect, self).__init__(
            input, output, name, verbose, parallel)

        self.reload()

    def unload(self):
        del self.stemmer

    def reload(self, **_):
        from spacy_langdetect import LanguageDetector

        import spacy
        from spacy.language import Language

        @Language.factory("language_detector")
        def get_lang_detector(nlp, name):
            return LanguageDetector()

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe("language_detector", last=True)
        self.nlp = nlp

    def process_doc(self, text):
        doc = self.nlp(text)
        return doc._.language["language"]
