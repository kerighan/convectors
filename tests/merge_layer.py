from convectors.layers import SVD, Merge, TfIdf, Tokenize
from sklearn.datasets import fetch_20newsgroups

# load data
training_set = fetch_20newsgroups(subset="train").data

left_model = Tokenize()
left_model += TfIdf(max_features=400)
left_model += SVD(n_components=20)

right_model = Tokenize()
right_model = TfIdf(max_features=200)
right_model += SVD(n_components=30)

model = Merge(left_model, right_model)
model += SVD(n_components=10)

X = model(training_set, training_set)

print(X.shape)
