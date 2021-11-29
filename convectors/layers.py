from .classifier import (MLP, RNN, SVM, TFMLP, AdaBoost, Keras, RandomForest,
                         Transformer, XGBoost)
from .embedding import CountVectorizer, Doc2Vec, OneHot, Sequence, TfIdf
from .linguistics import (Contract, FindAll, LangDetect, Lemmatize, NGram,
                          Phrase, Snowball, Sub, Tokenize)
from .multi import Merge
from .preprocessing import Normalize
from .reduction import NMF, PCA, SVD, UMAP, RandomizedSVD
from .regressor import KerasRegressor, MLPRegressor
from .special import DomainName, Lambda, SplitHashtag

# retro compatibility
Lemmatizer = Lemmatize


__all__ = [
    "Tokenize", "Snowball", "Phrase", "Sub", "Contract", "FindAll",
    "LangDetect", "TfIdf", "CountVectorizer", "Sequence", "OneHot",
    "Doc2Vec", "SVD", "PCA", "NMF", "UMAP", "Lemmatize", "NGram",
    "RandomForest", "AdaBoost", "SVM", "MLP", "Keras", "Transformer",
    "XGBoost", "Merge", "Normalize", "SplitHashtag", "TFMLP",
    "MLPRegressor", "KerasRegressor", "Lambda", "RNN", "DomainName",
    "RandomizedSVD"
]
