import numpy as np

from .. import Layer, to_matrix
from .utils import tensorflow_shutup

tensorflow_shutup()


class ClassifierLayer(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose, False)

    def fit(self, series, y=None):
        assert y is not None
        X = to_matrix(series)
        self.clf.fit(X, y)

    def process_series(self, series):
        X = to_matrix(series)
        return self.clf.predict(X)


class RandomForest(ClassifierLayer):
    def __init__(
        self,
        input=None,
        output=None,
        n_estimators=100,
        max_depth=None,
        name=None,
        verbose=True,
        **kwargs
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose)

        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, **kwargs)


class AdaBoost(ClassifierLayer):
    def __init__(
        self,
        input=None,
        output=None,
        n_estimators=50,
        name=None,
        verbose=True,
        **kwargs
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose)

        from sklearn.ensemble import AdaBoostClassifier
        self.clf = AdaBoostClassifier(n_estimators=n_estimators, **kwargs)


class SVM(ClassifierLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        **kwargs
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose)

        from sklearn.svm import LinearSVC
        self.clf = LinearSVC(**kwargs)


class MLP(ClassifierLayer):
    def __init__(
        self,
        input=None,
        output=None,
        hidden_layer_sizes=100,
        activation='relu',
        name=None,
        verbose=True,
        **kwargs
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose)

        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, **kwargs)


class GradientBoosting(ClassifierLayer):
    def __init__(
        self,
        input=None,
        output=None,
        n_estimators=100,
        name=None,
        verbose=True,
        **kwargs
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose)

        from sklearn.ensemble import GradientBoostingClassifier as GBC
        self.clf = GBC(n_estimators=n_estimators, **kwargs)


class XGBoost(ClassifierLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        **kwargs
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose)

        from xgboost import XGBClassifier
        self.clf = XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss", **kwargs)


class Voting(ClassifierLayer):
    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        estimators=["gb", "rf", "mlp"],
        verbose=True,
        **kwargs
    ):
        super(ClassifierLayer, self).__init__(
            input, output, name, verbose)

        e = []
        for estimator in estimators:
            if estimator == "gb":
                from sklearn.ensemble import GradientBoostingClassifier as GBC
                e.append(("gb", GBC()))
            elif estimator == "rf":
                from sklearn.ensemble import RandomForestClassifier
                e.append(("rf", RandomForestClassifier()))
            elif estimator == "mlp":
                from sklearn.neural_network import MLPClassifier
                e.append(("mlp", MLPClassifier()))
            elif estimator == "svm":
                from sklearn.svm import LinearSVC
                e.append(("svm", LinearSVC()))
            elif estimator == "xgboost":
                from xgboost import XGBClassifier
                e.append(("xgboost", XGBClassifier(
                    use_label_encoder=False, eval_metric="mlogloss")))

        from sklearn.ensemble import VotingClassifier
        self.clf = VotingClassifier(estimators=e, **kwargs)


class Keras(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        model=None,
        name=None,
        verbose=True
    ):
        super(Keras, self).__init__(
            input, output, name, verbose, False)
        assert model is not None
        self.model = model

    def unload(self):
        self.weights = self.model.get_weights()
        self.config = self.model.get_config()
        del self.model

    def reload(self, custom_objects=None, **_):
        from tensorflow.keras.models import Model as KModel
        from tensorflow.keras.models import Sequential
        try:
            model = KModel.from_config(
                self.config, custom_objects=custom_objects)
        except KeyError:
            model = Sequential.from_config(
                self.config, custom_objects=custom_objects)
        model.set_weights(self.weights)
        del self.weights
        del self.config
        self.model = model

    def process_series(self, series):
        from scipy.sparse import issparse
        X = to_matrix(series)
        if issparse(X):
            X = np.array(X.todense())
        return self.model.predict(X)


class Transformer(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        verbose=True,
        embedding_dim=32,
        encoder_dim=16,
        weighted_dim=16,
        n_heads=4,
        n_encoders=1,
        n_weighted=1,
        encoder_activation=None,
        weighted_activation="tanh",
        l1=1e-5,
        weights=None,
        train_embedding=True,
        **options
    ):
        super(Transformer, self).__init__(input, output, name, verbose, False)
        self.options = options
        if "epochs" not in self.options:
            options["epochs"] = 1
        if "batch_size" not in self.options:
            options["batch_size"] = 200

        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.weighted_dim = weighted_dim
        self.n_heads = n_heads
        self.n_encoders = n_encoders
        self.n_weighted = n_weighted
        self.l1 = l1
        self.weights = weights
        self.train_embedding = train_embedding
        self.encoder_activation = encoder_activation
        self.weighted_activation = weighted_activation

    def unload(self):
        self.weights = self.model.get_weights()
        self.config = self.model.get_config()
        del self.model

    def reload(self, **_):
        from tensorflow.keras.models import Model as KModel
        from tensorflow.keras.models import Sequential

        from .layers import SelfAttention, WeightedAttention

        custom_objects = {
            "SelfAttention": SelfAttention,
            "WeightedAttention": WeightedAttention
        }
        try:
            model = KModel.from_config(
                self.config, custom_objects=custom_objects)
        except KeyError:
            model = Sequential.from_config(
                self.config, custom_objects=custom_objects)
        model.set_weights(self.weights)
        del self.weights
        del self.config
        self.model = model

    def fit(self, series, y=None):
        from tensorflow.keras.layers import (Activation, BatchNormalization,
                                             Dense, Embedding, InputLayer)
        from tensorflow.keras.models import Sequential

        from .layers import SelfAttention, WeightedAttention
        assert y is not None

        # get data
        from scipy.sparse import issparse
        X = to_matrix(series)
        if issparse(X):
            X = np.array(X.todense())

        # shape variables
        input_len = X.shape[1]
        embedding_dim = self.embedding_dim

        if len(y.shape) == 1:
            n_classes = y.max() + 1
            loss = "sparse_categorical_crossentropy"
        else:
            n_classes = y.shape[1]
            loss = "categorical_crossentropy"

        # create model
        model = Sequential()
        model.add(InputLayer(input_shape=(input_len,)))

        # embedding layer
        if self.weights is None:
            n_features = int(X.max()) + 1
            model.add(Embedding(n_features, embedding_dim, mask_zero=True))
        else:
            n_features = self.weights.shape[0]
            embedding_dim = self.weights.shape[1]
            model.add(Embedding(n_features, embedding_dim,
                                mask_zero=True, weights=[self.weights],
                                trainable=self.train_embedding))

        model.add(Activation("tanh"))

        # self attention layers
        for _ in range(self.n_encoders):
            model.add(SelfAttention(self.encoder_dim,
                      self.n_heads, self.l1,
                      activation=self.encoder_activation))
            model.add(BatchNormalization())

        # weighted attention layer
        model.add(WeightedAttention(
            self.weighted_dim, self.n_weighted, self.l1,
            activation=self.weighted_activation))
        model.add(Dense(n_classes, activation="softmax"))
        model.compile("nadam", loss, metrics=["accuracy"])
        if self.verbose:
            model.summary()

        # fit model
        model.fit(X, y, **self.options)
        self.model = model

    def process_series(self, series):
        from scipy.sparse import issparse
        X = to_matrix(series)
        if issparse(X):
            X = np.array(X.todense())
        if not hasattr(self, "model"):
            self.reload()
        return self.model.predict(X)
