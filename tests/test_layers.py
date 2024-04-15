def test_concatenate():
    from convectors.layers import Input, Concatenate
    from convectors.models import Model

    input_layer1 = Input(name="input_layer1")
    input_layer2 = Input(name="input_layer2")

    concat_layer = Concatenate(name="concat_layer")

    # input the inputs to the concatenate layer
    concat_layer.input([input_layer1, input_layer2])

    # Pass some data through the pipeline
    data1 = ["This is a sample", "This is another "]
    data2 = [" sentence.", "longer sentence."]

    model = Model(inputs=[input_layer1, input_layer2], outputs=concat_layer)
    processed_data = model(data1, data2)
    assert processed_data == [
        "This is a sample sentence.",
        "This is another longer sentence.",
    ]


def test_tokenize():
    from convectors.layers import Tokenize

    tokenize = Tokenize()
    assert tokenize("Hello world!") == ["hello", "world"]

    tokenize_2 = Tokenize(stopwords=["en"], strip_accents=True, name="tokenize_2")

    # Pass some data through the pipeline
    data = ["This is a sample sentence.", "This is another sample sentence."]
    processed_data = tokenize_2(data)
    assert processed_data == [["sample", "sentence"], ["another", "sample", "sentence"]]


def test_snowball():
    from convectors.layers import Tokenize, Input, SnowballStem
    from convectors.models import Model

    input_layer = Input()
    tokenize = Tokenize().input(input_layer)
    stem = SnowballStem(lang="en").input(tokenize)

    model = Model(inputs=input_layer, outputs=stem)
    processed_data = model("This is a sample sentence.")
    assert processed_data == ["this", "is", "a", "sampl", "sentenc"]

    processed_data = model(["This is a sample sentence."])
    assert processed_data == [["this", "is", "a", "sampl", "sentenc"]]


def test_vectorizers():
    from convectors.layers import Tokenize, Input, TfIdf, HashingVectorizer, SVD
    from convectors.models import Model

    input_layer = Input()
    tokenize = Tokenize().input(input_layer)

    hashing_vectorizer = HashingVectorizer().input(tokenize)
    svd = SVD(n_components=3).input(hashing_vectorizer)
    tfidf = TfIdf(sparse=False).input(tokenize)

    model = Model(inputs=input_layer, outputs=[tfidf, svd])
    # fit
    model.fit(
        [
            "This is a sample sentence.",
            "This is another sample sentence.",
            "This is a third sample sentence.",
        ]
    )
    # transform
    vec_1, vec_2 = model("This is a fourth sentence.")


def test_batch():
    from convectors.layers import (
        Input,
        Tiktokenize,
        Lambda,
        Pad,
        DocumentSplitter,
        Suffix,
        Prefix,
    )
    from convectors.models import Model

    input_layer = Input()

    suffix = Suffix(suffix="<|endoftext|>").input(input_layer)
    prefix = Prefix(prefix="<|startoftext|>").input(suffix)

    tiktoken = Tiktokenize().input(prefix)
    doc_splitter = DocumentSplitter(maxlen=4, overlap=1).input(tiktoken)

    first_layer = Lambda(lambda x: x[:-1]).input(doc_splitter)
    first_layer = Pad(maxlen=3).input(first_layer)

    second_layer = Lambda(lambda x: x[1:]).input(doc_splitter)
    second_layer = Pad(maxlen=3).input(second_layer)

    model = Model(inputs=input_layer, outputs=[first_layer, second_layer])

    texts = [f"This is the {i}th sentence" for i in range(9)]
    for X, y in model(texts, batch_size=2):
        pass


if __name__ == "__main__":
    test_concatenate()
    test_tokenize()
    test_snowball()
    test_vectorizers()
    test_batch()
