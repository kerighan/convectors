from .. import Layer, to_matrix


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
