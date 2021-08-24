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
