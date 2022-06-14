from convectors import load_model
from convectors.layers import SVD, Keras, OneHot, Snowball, TfIdf, Tokenize
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

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
X_train = encoder(training_set.data)
y_train = training_set.target

# infer number of features and classes
N_CLASSES = y_train.max().n_features + 1

# create keras model and fit
model = Sequential()
model.add(Dense(100, activation="tanh"))
model.add(Dense(100, activation="tanh"))
model.add(Dense(N_CLASSES, activation="softmax"))
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=6, batch_size=200, validation_split=.1)

# once learned, add Keras model and one_hot decoder to NLP model
encoder += Keras(model=model, trained=True)

# save and load model
encoder.save("encoder.p")
encoder = load_model("encoder.p")

# predict on new data
y_pred = encoder(testing_set.data).argmax(axis=1)
y_true = testing_set.target
print((y_pred == y_true).mean())
