from convectors import load_model
from convectors.layers import SVD, Argmax, Keras, Snowball, TfIdf, Tokenize
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

# get training data
X_train = encoder(training_set.data)
y_train = training_set.target

# infer number of features and classes
N_CLASSES = y_train.max() + 1

# create keras model and fit
model = Sequential()
model.add(Dense(100, activation="tanh"))
model.add(Dense(100, activation="tanh"))
model.add(Dense(N_CLASSES, activation="softmax"))
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=6, batch_size=200, validation_split=.1)

# once learned, add Keras model to processing pipeline
encoder += Keras(model=model, trained=True)
# add a simple Argmax layer
encoder += Argmax()
# turn verbosity off
encoder.verbose = False

# save and load model
encoder.save("encoder.p")
encoder = load_model("encoder.p")

# predict on new data
y_pred = encoder(testing_set.data)
y_true = testing_set.target
print((y_pred == y_true).mean())
