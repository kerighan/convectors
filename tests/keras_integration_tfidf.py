from sklearn.datasets import fetch_20newsgroups
from convectors.layers import Tokenize, Snowball, OneHot, TfIdf, SVD, Keras
from convectors import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
training_set = fetch_20newsgroups(subset="train")
testing_set = fetch_20newsgroups(subset="test")

# create encoder model
encoder = Tokenize(stopwords=["en"])
encoder += Snowball(lang="english")
encoder += TfIdf(max_features=20000, max_df=.3)
encoder += SVD(n_components=200)

# create one hot encoder for categorical data
one_hot = OneHot(to_categorical=True)

# get training data
X = encoder(training_set.data)
Y = one_hot(training_set.target)

# infer number of features and classes
N_CLASSES = one_hot.n_features

# create keras model and fit
model = Sequential()
model.add(Dense(100, activation="tanh"))
model.add(Dense(100, activation="tanh"))
model.add(Dense(N_CLASSES, activation="softmax"))

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X, Y, epochs=6, batch_size=200, validation_split=.1)

# once learned, add Keras model and one_hot decoder to NLP model
encoder += Keras(model=model)
encoder += one_hot.get_decoder()

# save model
encoder.save("model.p")
del encoder

# retrieve model as well as the Keras neural network in it
encoder = load_model("model.p")
y_pred = encoder(testing_set.data)
y_true = testing_set.target
print((y_pred == y_true).mean())
