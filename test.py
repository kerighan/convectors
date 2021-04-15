from convectors.layers import Tokenize, Embedding
import pandas as pd

df = pd.read_pickle("D:\\research\\veolia\\datasets\\statuses.p")
df = df[~df.isRT]
print("loaded")

model = Tokenize(stopwords=["fr"])
model += Embedding(max_features=1000, maxlen=10, pad=True)
print(model)
print(model(df.content))
