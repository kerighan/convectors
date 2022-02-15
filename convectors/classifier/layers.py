import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l1 as l1reg


class SelfAttention(Layer):
    def __init__(
            self, hidden_dim=20, n_heads=4, l1=1e-5,
            activation=None, **_):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = int(hidden_dim**.5)
        self.n_heads = n_heads
        self.l1 = l1

        self.activation = activation
        if activation is None:
            self.act = lambda x: x
        elif activation == "sigmoid":
            self.act = tf.sigmoid
        elif activation == "tanh":
            self.act = tf.tanh
        elif activation == "relu":
            self.act = tf.nn.relu
        elif activation == "selu":
            self.act = tf.nn.selu
        elif activation == "softmax":
            self.act = tf.nn.softmax

    def build(self, input_shape):
        input_length = int(input_shape[-2])
        embedding_dim = int(input_shape[-1])

        self.positional = self.add_weight(
            "positional", shape=[input_length, embedding_dim])

        self.query = self.add_weight(
            "query", shape=[self.n_heads, embedding_dim, self.hidden_dim],
            regularizer=l1reg(self.l1))
        self.key = self.add_weight(
            "key", shape=[self.n_heads, embedding_dim, self.hidden_dim],
            regularizer=l1reg(self.l1))
        self.value = self.add_weight(
            "value", shape=[self.n_heads, embedding_dim, self.hidden_dim],
            regularizer=l1reg(self.l1))
        self.weight = self.add_weight(
            "weight", shape=[self.n_heads * self.hidden_dim, embedding_dim],
            regularizer=l1reg(self.l1))

    def call(self, input, mask=None):
        input += self.positional

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            input *= mask[:, :, None]

        results = []
        for i in range(self.n_heads):
            # query vector
            q = self.act(tf.matmul(input, self.query[i]))

            # key vector
            k = self.act(tf.matmul(input, self.key[i]))
            k = tf.transpose(k, [0, 2, 1])

            # value vector
            v = self.act(tf.matmul(input, self.value[i]))

            # score vector
            score = tf.matmul(q, k) / self.scale
            score = tf.nn.softmax(score, axis=-1)

            result = tf.matmul(score, v)
            results.append(result)
        results = tf.concat(results, axis=-1)

        transformed = tf.matmul(results, self.weight)
        return transformed + self.act(input)

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "l1": self.l1,
            "activation": self.activation})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WeightedAttention(tf.keras.layers.Layer):
    def __init__(
            self, hidden_dim=32, n_layers=1, l1=1e-5,
            activation="sigmoid", **kwargs):
        super(WeightedAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.l1 = l1
        self.scale = hidden_dim**.5

        self.activation = activation
        if activation is None:
            self.act = lambda x: x
        elif activation == "sigmoid":
            self.act = tf.sigmoid
        elif activation == "tanh":
            self.act = tf.tanh
        elif activation == "relu":
            self.act = tf.nn.relu
        elif activation == "selu":
            self.act = tf.nn.selu
        elif activation == "softmax":
            self.act = tf.nn.softmax

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.projector = self.add_weight(
            "projector", shape=(dim, self.hidden_dim),
            trainable=True, regularizer=l1reg(self.l1))

        if self.n_layers > 1:
            self.hidden = self.add_weight(
                "hidden",
                shape=(self.n_layers - 1, self.hidden_dim, self.hidden_dim),
                trainable=True, regularizer=l1reg(self.l1))

        self.evaluator = self.add_weight(
            "evaluator", shape=(self.hidden_dim, 1), trainable=True,
            regularizer=l1reg(self.l1))

    def call(self, inp, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            inp *= mask[:, :, None]

        # project and evaluate incoming inputs
        weights = self.act(tf.matmul(inp, self.projector) / self.scale)
        # feed forward
        for i in range(self.n_layers - 1):
            weights = self.act(
                tf.matmul(weights, self.hidden[i]) / self.scale)

        weights = self.act(tf.matmul(weights, self.evaluator) / self.scale)
        weights = tf.nn.softmax(weights, axis=-2)

        result = weights * inp
        result = tf.math.reduce_sum(result, axis=1, keepdims=False)
        return result

    def get_config(self):
        config = super(WeightedAttention, self).get_config()
        config.update({"hidden_dim": self.hidden_dim,
                       "n_layers": self.n_layers,
                       "l1": self.l1,
                       "activation": self.activation})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


class ACTS(Layer):
    def __init__(self,
                 encoder_dim=16,
                 n_sample_points=20,
                 minval=1e-3,
                 maxval=100,
                 l1=1e-4,
                 activation=None,
                 train_theta=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_sample_points = n_sample_points
        self.minval = minval
        self.maxval = maxval
        self.encoder_dim = encoder_dim
        self.l1 = l1
        self.activation = tf.keras.activations.get(activation)
        self.train_theta = train_theta

    def build(self, input_shape):
        import numpy as np
        input_length = input_shape[1]
        input_dim = input_shape[2]
        total_dim = 2 * self.n_sample_points * input_dim

        # self.theta = self.add_weight(
        #     "theta", shape=[1, self.n_sample_points],
        #     initializer=tf.random_uniform_initializer(
        #         minval=self.minval, maxval=self.maxval),
        #     regularizer=l1(1e-4))
        self.theta = tf.Variable(np.linspace(
            self.minval, self.maxval, self.n_sample_points
        ).astype(np.float32)[None, :], trainable=self.train_theta)

        # self.theta = self.add_weight(
        #     "theta", shape=[input_dim, 1, self.n_sample_points],
        #     initializer=tf.random_uniform_initializer(
        #         minval=self.minval, maxval=self.maxval),
        #     regularizer=l1(self.l1))

        self.attention = self.add_weight(
            "attention", shape=[self.encoder_dim, 1])

        self.query = self.add_weight(
            "query",
            shape=[input_dim, self.encoder_dim],
            regularizer=l1reg(self.l1))

        self.value = self.add_weight(
            "value",
            shape=[input_dim, self.encoder_dim],
            regularizer=l1reg(self.l1))

        self.input_length = input_length
        self.input_dim = input_dim

    def call(self, input, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            input *= mask[:, :, None]

        # value = self.activation(tf.matmul(input, self.value))
        # value = tf.transpose(input, perm=[0, 2, 1])

        # query = tf.nn.sigmoid(tf.matmul(input, self.query))

        # attention score
        # score = tf.matmul(query, self.attention)
        # score = tf.nn.softmax(score / (self.input_dim**.5),
        #                       axis=-2)[:, None]

        # compute characteristic function
        phi = tf.matmul(input[:, :, :, None], self.theta)
        # real = tf.reduce_sum(tf.cos(phi) * score, axis=-2)
        # imag = tf.reduce_sum(tf.sin(phi) * score, axis=-2)
        real = tf.reduce_mean(tf.cos(phi), axis=-2)
        imag = tf.reduce_mean(tf.sin(phi), axis=-2)
        vec = tf.concat([real, imag], axis=-1)

        vec = tf.reshape(vec, (-1, vec.shape[-1] * vec.shape[-2]))
        return vec


class MultiACTS(Layer):
    def __init__(self,
                 encoder_dim=16,
                 n_sample_points=20,
                 minval=1e-3,
                 maxval=100,
                 l1=1e-4,
                 activation=None,
                 train_theta=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_sample_points = n_sample_points
        self.minval = minval
        self.maxval = maxval
        self.encoder_dim = encoder_dim
        self.l1 = l1
        self.train_theta = train_theta
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        import numpy as np
        input_length = input_shape[1]
        input_dim = input_shape[2]
        total_dim = 2 * self.n_sample_points * input_dim

        # self.theta = self.add_weight(
        #     "theta", shape=[input_dim, self.n_sample_points],
        #     initializer=tf.random_uniform_initializer(
        #         minval=self.minval, maxval=self.maxval),
        #     regularizer=l1(1e-4))

        # self.theta = self.add_weight(
        #     "theta", shape=[input_dim, 1, self.n_sample_points],
        #     initializer=tf.random_uniform_initializer(
        #         minval=self.minval, maxval=self.maxval),
        #     regularizer=l1(self.l1))
        theta_init = np.tile(np.linspace(
            self.minval, self.maxval, self.n_sample_points
        ), input_dim).reshape((input_dim, 1, self.n_sample_points))
        self.theta = tf.Variable(theta_init.astype(
            np.float32), trainable=self.train_theta)

        self.input_length = input_length
        self.input_dim = input_dim

    def call(self, input, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            input *= mask[:, :, None]

        vec = []
        for i in range(self.encoder_dim):
            phi = tf.matmul(input[:, i, :, None], self.theta[i])
            real = tf.reduce_mean(tf.cos(phi), axis=-2)
            imag = tf.reduce_mean(tf.sin(phi), axis=-2)
            vec.append(real)
            vec.append(imag)

        vec = tf.concat(vec, axis=-1)

        return vec
