from convectors.layers import Tiktokenize, Input, TfIdf, SVD
from convectors.models import Model


def generator():
    a = ["how are you ?", "where are you ?", "Here I am"]
    for it in a:
        yield it


inp = Input()
tokenize = Tiktokenize().input(inp)
tfidf = TfIdf(sparse=False).input(tokenize)
svd = SVD(n_components=3).input(tfidf)

model = Model(inputs=inp, outputs=[tokenize, svd])
X = model(generator(), batch_size=1)
print(X)
for item in X:
    print(item)
# print(X)
