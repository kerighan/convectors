def test_functional_api():
    from convectors.layers import (
        Tokenize,
        Input,
        Tiktokenize,
        SnowballStem,
        Concatenate,
    )
    from convectors.models import Model

    inp_1 = Input()
    tokenize_1 = Tokenize().input(inp_1)
    tokenize_1 = SnowballStem().input(tokenize_1)

    inp_2 = Input()
    tokenize_2 = Tokenize().input(inp_2)
    tokenize_2 = SnowballStem().input(tokenize_2)
    concat = Concatenate().input((tokenize_1, tokenize_2))

    tiktokenize = Tiktokenize().input(inp_2)

    model = Model(inputs=[inp_1, inp_2], outputs=[concat, tiktokenize])
    X_1, X_2 = model("Hello worlds!", "How are you?")
    assert X_1 == ["hello", "world", "how", "are", "you"]
    assert X_2 == [2438, 390, 346, 31]

    X_1, X_2 = model(["Hello worlds!"], ["How are you?"])
    assert X_1 == [["hello", "world", "how", "are", "you"]]
    assert X_2 == [[2438, 390, 346, 31]]

    model = Model(inputs=[inp_1, inp_2], outputs=concat)
    X_1 = model("Hello worlds!", "How are you?")
    assert X_1 == ["hello", "world", "how", "are", "you"]


if __name__ == "__main__":
    test_functional_api()
