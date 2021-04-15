from convectors.layers import Tokenize, Snowball, TfIdf, SVD, Keras, OneHot
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# load data
training_set = fetch_20newsgroups(subset='train')
testing_set = fetch_20newsgroups(subset='test')

one_hot = OneHot(to_categorical=True)
# fit and return data
y = one_hot(training_set.target[:100])

# create encoder model
nlp = Tokenize(stopwords=["en"])
nlp += Snowball(lang="english")
nlp += TfIdf(max_features=10000, max_df=.3)
nlp += SVD(n_components=50)
# fit and return data
X = nlp(training_set.data[:100])

# create keras model
model = Sequential()
model.add(Dense(100, activation="tanh"))
model.add(Dense(one_hot.n_classes, activation="softmax"))
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=2, batch_size=200)

# add Keras model to NLP model
nlp += Keras(model=model)
nlp += one_hot.get_decoder()
print(nlp(testing_set.data))
