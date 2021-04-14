from convectors.layers import Tokenize, Snowball, Phrase, TfIdf
import pandas as pd

df = pd.read_pickle("/home/maixent/Dropbox (Reputation Squad)/DATAPROJECTS/"
                    "projets/TousAntiCovid/recurring/datasets/"
                    "0108_0115_statuses.p")
df = df[~df.isRT]

model = Tokenize(stopwords=["fr"])
model += Snowball()
model += Phrase()
model += TfIdf(sparse=False)
series = model(df.content)
print(series)
# process_series