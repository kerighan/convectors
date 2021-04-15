from convectors.layers import Tokenize, Snowball, TfIdf, SVD, SVM
from sklearn.datasets import fetch_20newsgroups

# load data
training_set = fetch_20newsgroups(subset='train')
testing_set = fetch_20newsgroups(subset='test')

# create encoder model
nlp = Tokenize(stopwords=["en"])
nlp += Snowball(lang="english")
nlp += TfIdf(max_features=20000, max_df=.3)
nlp += SVD(n_components=400)
nlp += SVM()

# fit and train model with just one call
nlp(training_set.data,
    y=training_set.target)

# use trained model for inference
y_pred = nlp(testing_set.data)
y_true = testing_set.target
# compute accuracy
print((y_true == y_pred).mean())
