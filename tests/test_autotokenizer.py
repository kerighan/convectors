from convectors.tokenizers.autotokenizer import AutoTokenize


nlp = AutoTokenize()
print(nlp.n_features)
print(nlp("C'est une phrase."))
print(nlp.decode(nlp("C'est une phrase.")))
