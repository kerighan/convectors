Convectors: build end-to-end NLP pipelines
==========

Inspired by the Keras syntax, Convectors allows you to build NLP pipelines by adding different processing Layers.
Fully compatible with pandas, it can either process list or pandas series on the fly, or apply processing to a whole DataFrame by using columns as inputs and outputs.


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

# persist model for future usage
nlp.save("model.p")
```