from convectors.layers import (
    Tokenize, Embedding, OneHot, Phrase, Snowball)
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# load data
newsgroups_train = fetch_20newsgroups(subset='train')

# turn data into a DataFrame
df = pd.DataFrame()
df["text"] = newsgroups_train.data
df["target"] = list(newsgroups_train.target)
print(df)

# create encoder model
encoder = Tokenize(stopwords=["en"])
encoder += Snowball(lang="english")
encoder += Phrase()
encoder += Embedding(pad=True, maxlen=200)
print(encoder(df.text))
# X = encoder(df.text)

# # one hot model
# one_hot = OneHot(to_categorical=False)
# y = one_hot(df.target)

# print(X)
# print(y)
