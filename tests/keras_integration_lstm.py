from convectors import load_model
from convectors.layers import Argmax, Keras, Sequence, Tokenize
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential

# load data
training_set = fetch_20newsgroups(subset="train")
testing_set = fetch_20newsgroups(subset="test")

# create encoder model
encoder = Tokenize(stopwords=["en"])
encoder += Sequence(max_features=20000, pad=True, maxlen=200)
X_train = encoder(training_set.data)  # fit and transform
# get training data
y_train = training_set.target

# infer number of features and classes
N_FEATURES = encoder["Sequence"].n_features + 1
N_CLASSES = y_train.max() + 1
EMBEDDING_DIM = 32

# create keras model and fit
model = Sequential()
model.add(Embedding(N_FEATURES, EMBEDDING_DIM, mask_zero=True))
model.add(LSTM(32, activation="tanh", return_sequences=False))
model.add(Dense(32, activation="tanh"))
model.add(Dense(N_CLASSES, activation="softmax"))

model.compile("nadam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=1, batch_size=800)

# once learned, add Keras model
encoder += Keras(model=model, trained=True)
encoder += Argmax()
# turn verbosity off
encoder.verbose = False

# for model persistence:
encoder.save("model.p")
encoder = load_model("model.p")

y_pred = encoder(testing_set.data)
y_true = testing_set.target
# print accuracy
print((y_pred == y_true).mean())
