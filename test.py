from convectors import load_model
from convectors.layers import (
    Tokenize, Snowball, Phrase, TfIdf, UMAP)
from convectors import load_model
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_pickle("/home/maixent/Dropbox (Reputation Squad)/DATAPROJECTS/"
                    "projets/TousAntiCovid/recurring/datasets/"
                    "0108_0115_statuses.p")
# df = df[~df.isRT]
print("here")

model = Tokenize(stopwords=["fr"])
model += Snowball()
model += Phrase()
model += TfIdf()
model += UMAP()
print(model(df.content))
