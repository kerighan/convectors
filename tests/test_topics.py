from sklearn.datasets import fetch_20newsgroups
from convectors.layers import Input, Tokenize, SnowballStem, TfIdf
from convectors.algorithms.pmi import text_graph_topics
import networkx as nx
from telekinesis import query
import pandas as pd


# df = query(
#     """
# DATA { Press->Articles }
# QUERY { ENGIE }
# TIMELINE { 30d }
# INTENT { list }
# """
# )
# df.to_pickle("content.p")
df = pd.read_pickle("content.p")
data = df.content.tolist()
# print(df)
topic_info = text_graph_topics(data)
print(topic_info)
# newsgroups_train = fetch_20newsgroups(subset="train")
# data = newsgroups_train.data

# break
