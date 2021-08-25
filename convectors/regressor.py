import numpy as np

from . import Layer, to_matrix
from .classifier.utils import tensorflow_shutup

tensorflow_shutup()


class RegressorLayer(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        validation_split=.1,
        name=None,
        verbose=True,
    ):
        super(RegressorLayer, self).__init__(
            input, output, name, verbose, False)
        self.validation_split = validation_split

    def fit(self, series, y=None):
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split

        assert y is not None
        X = to_matrix(series)

        # train test split
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=self.validation_split, random_state=0)

        self.regr.fit(X, y)

        mae = 100 * mean_absolute_error(self.regr.predict(X_test), y_test)
        print(f"val_mae={mae:.2f}")

    def process_series(self, series):
        X = to_matrix(series)
        return self.regr.predict(X)


class MLPRegressor(RegressorLayer):
    def __init__(
        self,
        input=None,
        output=None,
        hidden_layer_sizes=100,
        activation='relu',
        validation_split=.1,
        name=None,
        verbose=True,
        **kwargs
    ):
        super(MLPRegressor, self).__init__(
            input, output, validation_split, name, verbose)

        from sklearn.neural_network import MLPRegressor as MLPR
        self.regr = MLPR(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, **kwargs)


class KerasRegressor(Layer):
    parallel = False
    trainable = True
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        model=None,
        validation_split=.1,
        name=None,
        verbose=True,
        **options
    ):
        super(KerasRegressor, self).__init__(
            input, output, name, verbose, False)
        assert model is not None
        self.options = options
        if "epochs" not in self.options:
            options["epochs"] = 1
        if "batch_size" not in self.options:
            options["batch_size"] = 200
        self.model = model
        self.validation_split = validation_split

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

    def fit(self, series, y=None):
        from sklearn.model_selection import train_test_split

        from .classifier.layers import SaveBestModel

        assert y is not None

        # get data
        X = self.get_numpy_matrix(series)

        # train test split
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=self.validation_split, random_state=0)

        if self.verbose:
            self.model.summary()

        # fit model
        save_best_model = SaveBestModel()
        self.model.fit(X, y, validation_data=(X_test, y_test),
                       callbacks=[save_best_model], **self.options)
        self.model.set_weights(save_best_model.best_weights)


class TransformerRegressor(Layer):
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
        n_hidden=0,
        encoder_activation=None,
        weighted_activation="tanh",
        embedding_activation="tanh",
        out_activation="sigmoid",
        l1=1e-6,
        optimizer="nadam",
        loss="mse",
        weights=None,
        train_embedding=True,
        validation_split=.1,
        **options
    ):
        super(TransformerRegressor, self).__init__(
            input, output, name, verbose, False)
        self.options = options
        if "epochs" not in self.options:
            options["epochs"] = 1
        if "batch_size" not in self.options:
            options["batch_size"] = 200

        self.optimizer = optimizer
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.weighted_dim = weighted_dim
        self.n_heads = n_heads
        self.n_encoders = n_encoders
        self.n_weighted = n_weighted
        self.n_hidden = n_hidden
        self.l1 = l1
        self.weights = weights
        self.train_embedding = train_embedding
        self.encoder_activation = encoder_activation
        self.weighted_activation = weighted_activation
        self.embedding_activation = embedding_activation
        self.out_activation = out_activation
        self.validation_split = validation_split
        self.loss = loss

    def unload(self):
        self.weights = self.model.get_weights()
        self.config = self.model.get_config()
        del self.model

    def reload(self, **_):
        from tensorflow.keras.models import Model as KModel
        from tensorflow.keras.models import Sequential

        from .classifier.layers import SelfAttention, WeightedAttention

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
        from sklearn.model_selection import train_test_split

        from .classifier.layers import SaveBestModel

        assert y is not None

        # get data
        X = self.get_numpy_matrix(series)
        n_features = int(X.max()) + 1 + 2  # one for mask and one for unk

        # train test split
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=self.validation_split, random_state=0)

        # shape variables
        input_len = X.shape[1]
        embedding_dim = self.embedding_dim

        if len(y.shape) == 1:
            n_classes = 1
        else:
            n_classes = y.shape[1]

        # create model
        if not hasattr(self, "model"):
            self.model = self.create_model(input_len,
                                           embedding_dim,
                                           n_features,
                                           n_classes)

        # compile model
        self.model.compile(self.optimizer, self.loss, metrics=["mae"])
        if self.verbose:
            self.model.summary()

        # fit model
        save_best_model = SaveBestModel()
        self.model.fit(X, y, validation_data=(X_test, y_test),
                       callbacks=[save_best_model], **self.options)
        self.model.set_weights(save_best_model.best_weights)

    def create_model(
        self,
        input_len, embedding_dim, n_features, n_classes
    ):
        from tensorflow.keras.layers import (Activation, BatchNormalization,
                                             Dense, Embedding, InputLayer)
        from tensorflow.keras.models import Sequential

        from .classifier.layers import SelfAttention, WeightedAttention

        model = Sequential()
        model.add(InputLayer(input_shape=(input_len,)))

        # embedding layer
        if self.weights is None:
            model.add(Embedding(n_features, embedding_dim, mask_zero=True))
        else:
            n_features = self.weights.shape[0]
            embedding_dim = self.weights.shape[1]
            model.add(Embedding(n_features, embedding_dim,
                                mask_zero=True, weights=[self.weights],
                                trainable=self.train_embedding))
        model.add(BatchNormalization())

        if self.embedding_activation is not None:
            model.add(Activation(self.embedding_activation))

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

        for _ in range(self.n_hidden):
            model.add(Dense(self.weighted_dim,
                      activation=self.weighted_activation))

        model.add(Dense(n_classes, activation=self.out_activation))
        return model

    def process_series(self, series):
        from scipy.sparse import issparse
        X = to_matrix(series)
        if issparse(X):
            X = np.array(X.todense())
        if not hasattr(self, "model"):
            self.reload()
        return self.model.predict(X)

    def truncate_model(self, until=-1, freeze=False):
        from tensorflow.keras.models import Sequential
        model = Sequential()

        if until is None:
            layers = self.model.layers
        else:
            layers = self.model.layers[:until]
        for layer in layers:
            if freeze:
                layer.trainable = False
            model.add(layer)
        return model
