from convectors.layers import Tokenize, Snowball, Phrase, TfIdf, SVD
import pandas as pd

df = pd.read_pickle("/home/maixent/Dropbox (Reputation Squad)/DATAPROJECTS/"
                    "projets/TousAntiCovid/recurring/datasets/"
                    "0108_0115_statuses.p")
df = df[~df.isRT]

model = Tokenize("content", "tokens", stopwords=["fr"])
model += Snowball("tokens")
model += Phrase("tokens")
model += TfIdf("tokens", "vectors", sparse=True)
model += SVD("vectors", n_components=2)

res = model(df.content)

res = model(df.content)

# print(type(res))
# model(df)
# print(df[["content", "vectors"]])
# print(df[["content", "tokens", "vectors"]])
