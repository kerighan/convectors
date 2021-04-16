from sklearn.datasets import fetch_20newsgroups
from convectors.layers import Tokenize, TfIdf, SVD, Keras, OneHot
from convectors import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
training_set = fetch_20newsgroups(subset='train')
testing_set = fetch_20newsgroups(subset='test')

# create one hot encoder for categorical data
one_hot = OneHot(to_categorical=True)
# fit and return data
y = one_hot(training_set.target)

# create encoder model
nlp = Tokenize(stopwords=["en"])
nlp += TfIdf(max_features=20000, max_df=.3)
nlp += SVD(n_components=100)
# fit and return data
X = nlp(training_set.data)

# create keras model and fit
model = Sequential()
model.add(Dense(100, activation="tanh"))
model.add(Dense(100, activation="tanh"))
model.add(Dense(one_hot.n_classes, activation="softmax"))
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=6, batch_size=200, validation_split=.1)

# add Keras model to NLP model
nlp += Keras(model=model)
# add one_hot decoder to NLP model
nlp += one_hot.get_decoder()
# save model
nlp.save("model.p")

# retrieve model as well as the Keras neural network in it
nlp = load_model("model.p")
y_pred = nlp(testing_set.data)
y_true = testing_set.target
print((y_pred == y_true).mean())
