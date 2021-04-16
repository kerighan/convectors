Convectors: build end-to-end NLP pipelines
==========

Inspired by the Keras syntax, Convectors allows you to build NLP pipelines by adding different processing Layers.
Fully compatible with pandas and Keras, it can either process list or pandas series on the fly, or apply processing to a whole DataFrame by using columns as inputs and outputs. Tensorflow's Keras models can be added as a layer, embedded and saved within a larger end-to-end NLP model.


```
pip install convectors
```

Simple classification example
=====

In this basic example, we create an NLP pipeline consisting of: tokenizer, Snowball stemmer, TfIdf vectorizer, SVD and a linear SVM classifier.

```python
from convectors.layers import Tokenize, Snowball, TfIdf, SVD, SVM
from sklearn.datasets import fetch_20newsgroups

# load data
training_set = fetch_20newsgroups(subset="train")
testing_set = fetch_20newsgroups(subset="test")

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

# persist model for future usage
nlp.save("model.p")
```

Seamless Keras integration
=====

```python
from sklearn.datasets import fetch_20newsgroups
from convectors.layers import Tokenize, TfIdf, SVD, Keras, OneHot
from convectors import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
training_set = fetch_20newsgroups(subset="train")
testing_set = fetch_20newsgroups(subset="test")

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
model.add(Dense(one_hot.n_features, activation="softmax"))
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
```