from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer


class SelfAttention(Layer):
    def __init__(
            self, hidden_dim=20, n_heads=4, l1=1e-5,
            activation=None, **_):
        import tensorflow as tf
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
        from tensorflow.keras.regularizers import l1 as l1reg

        if input_shape[-2] is not None:
            input_length = int(input_shape[-2])
        else:
            input_length = None
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

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, input, mask=None):
        import tensorflow as tf
        input += self.positional
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            input *= mask

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

            if mask is not None:
                score *= mask
                norm = tf.reduce_sum(score, keepdims=True, axis=1) + 1e-6
                score /= norm

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


class WeightedAttention(Layer):
    def __init__(
            self, hidden_dim=32, n_layers=1, l1=1e-5,
            activation="sigmoid", **kwargs):
        import tensorflow as tf
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
        from tensorflow.keras.regularizers import l1 as l1reg
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

    def compute_mask(self, _, mask=None):
        return None

    def call(self, inp, mask=None):
        import tensorflow as tf

        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            inp *= mask

        # project and evaluate incoming inputs
        weights = self.act(tf.matmul(inp, self.projector) / self.scale)
        # feed forward
        for i in range(self.n_layers - 1):
            weights = self.act(
                tf.matmul(weights, self.hidden[i]) / self.scale)

        weights = self.act(tf.matmul(weights, self.evaluator) / self.scale)
        weights = tf.nn.softmax(weights, axis=-2)

        if mask is not None:
            weights *= mask
            norm = tf.reduce_sum(weights, keepdims=True, axis=1) + 1e-12
            weights /= norm

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


class SaveBestModel(Callback):
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
                 residual=True,
                 learn_positional=True,
                 **kwargs):
        import tensorflow as tf
        super().__init__(**kwargs)
        self.n_sample_points = n_sample_points
        self.minval = minval
        self.maxval = maxval
        self.encoder_dim = encoder_dim
        self.l1 = l1
        self.activation = tf.keras.activations.get(activation)
        self.train_theta = train_theta
        self.scale = encoder_dim**.5
        self.residual = residual
        self.learn_positional = learn_positional

    def build(self, input_shape):
        import tensorflow as tf
        from tensorflow.keras.regularizers import l1 as l1reg
        input_length = input_shape[1]
        input_dim = input_shape[2]

        self.theta = self.add_weight(
            "theta", shape=[1, self.n_sample_points],
            initializer=tf.random_uniform_initializer(
                minval=self.minval, maxval=self.maxval),
            regularizer=l1reg(self.l1))

        # self.theta = tf.Variable(np.linspace(
        #     self.minval, self.maxval, self.n_sample_points
        # ).astype(np.float32)[None, :], trainable=self.train_theta)

        # self.value = self.add_weight(
        #     "value", shape=[input_dim, self.encoder_dim],
        #     regularizer=l1reg(self.l1))

        # self.attention = self.add_weight(
        #     "attention", shape=[input_dim, 1])

        self.attention_1 = self.add_weight(
            "attention_1",
            shape=[input_dim, self.encoder_dim],
            regularizer=l1reg(self.l1))
        self.attention_2 = self.add_weight(
            "attention_2",
            shape=[self.encoder_dim, 1],
            regularizer=l1reg(self.l1))
        self.alpha = self.add_weight(
            "alpha",
            shape=[1],
            regularizer=l1reg(self.l1))

        if self.learn_positional:
            self.positional = self.add_weight(
                "positional", shape=[input_length, input_dim])

        self.input_length = input_length
        self.input_dim = input_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, input, mask=None):
        if self.learn_positional:
            input += self.positional

        import tensorflow as tf
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            input *= mask

        # attention score
        # score = tf.matmul(input, self.attention)
        score = tf.nn.tanh(tf.matmul(input, self.attention_1) / self.scale)
        score = tf.matmul(score, self.attention_2)

        score = tf.nn.softmax((1 + self.alpha**2) * score, axis=-2)
        if mask is not None:
            score *= mask
            norm = tf.reduce_sum(score, keepdims=True, axis=1) + 1e-12
            score /= norm
        score = score[:, :, :, None]

        # compute characteristic function
        phi = tf.matmul(input[:, :, :, None], self.theta)
        real = tf.reduce_sum(tf.cos(phi) * score, axis=-2)
        imag = tf.reduce_sum(tf.sin(phi) * score, axis=-2)
        vec = tf.concat([real, imag], axis=-1)

        vec = tf.reshape(vec, (-1, vec.shape[-1] * vec.shape[-2]))

        if self.residual:
            res = tf.nn.tanh(tf.reduce_sum(
                input * score[:, :, 0], axis=-2, keepdims=False))
            # print(input.shape, res.shape, vec.shape, score.shape)
            vec = tf.concat([vec, res], axis=-1)
            return vec
        return vec

    def get_config(self):
        config = super(ACTS, self).get_config()
        config.update({"encoder_dim": self.encoder_dim,
                       "n_sample_points": self.n_sample_points,
                       "minval": self.minval,
                       "maxval": self.maxval,
                       "l1": self.l1,
                       "train_theta": self.train_theta,
                       "activation": self.activation,
                       "residual": self.residual})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiACTS(Layer):
    def __init__(self,
                 encoder_dim=16,
                 n_sample_points=10,
                 minval=1e-3,
                 maxval=100,
                 l1=1e-4,
                 activation=None,
                 train_theta=True,
                 residual=True,
                 learn_positional=True,
                 **kwargs):
        import tensorflow as tf
        super().__init__(**kwargs)
        self.n_sample_points = n_sample_points
        self.minval = minval
        self.maxval = maxval
        self.encoder_dim = encoder_dim
        self.l1 = l1
        self.train_theta = train_theta
        self.activation = tf.keras.activations.get(activation)
        self.scale = encoder_dim**.5
        self.residual = residual
        self.learn_positional = learn_positional

    def build(self, input_shape):
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.regularizers import l1 as l1reg
        input_length = input_shape[1]
        input_dim = input_shape[2]

        # self.theta = self.add_weight(
        #     "theta", shape=[input_dim, self.n_sample_points],
        #     initializer=tf.random_uniform_initializer(
        #         minval=self.minval, maxval=self.maxval),
        #     regularizer=l1(1e-4))

        self.theta = self.add_weight(
            "theta", shape=[input_dim, 1, self.n_sample_points],
            initializer=tf.random_uniform_initializer(
                minval=self.minval, maxval=self.maxval),
            regularizer=l1reg(self.l1))

        self.attention_1 = self.add_weight(
            "attention_1",
            shape=[input_dim, self.encoder_dim],
            regularizer=l1reg(self.l1))
        self.attention_2 = self.add_weight(
            "attention_2",
            shape=[self.encoder_dim, 1],
            regularizer=l1reg(self.l1))
        self.alpha = self.add_weight(
            "alpha",
            shape=[1],
            regularizer=l1reg(self.l1))

        if self.learn_positional:
            self.positional = self.add_weight(
                "positional", shape=[input_length, input_dim])

        # theta_init = np.tile(np.linspace(
        #     self.minval, self.maxval, self.n_sample_points
        # ), input_dim).reshape((input_dim, 1, self.n_sample_points))

        # self.theta = tf.Variable(theta_init.astype(np.float32),
        #                          trainable=self.train_theta)

        self.input_length = input_length
        self.input_dim = input_dim

    def call(self, input, mask=None):
        if self.learn_positional:
            input += self.positional

        import tensorflow as tf
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            input *= mask

        # attention score
        # score = tf.matmul(input, self.attention)
        # score = tf.nn.softmax(score / (self.input_dim**.5), axis=-2)
        # if mask is not None:
        #     score *= mask
        #     norm = tf.reduce_sum(score, keepdims=True, axis=1) + 1e-6
        #     score /= norm
        score = tf.nn.tanh(tf.matmul(input, self.attention_1) / self.scale)
        score = tf.matmul(score, self.attention_2)

        score = tf.nn.softmax((1 + self.alpha**2) * score, axis=-2)
        if mask is not None:
            score *= mask
            norm = tf.reduce_sum(score, keepdims=True, axis=1) + 1e-12
            score /= norm
        score = score[:, :, :]

        # concat
        vec = []
        for i in range(self.input_dim):
            phi = tf.matmul(input[:, :, i, None], self.theta[i])
            # dim_score = score[:, :, i, None]
            real = tf.reduce_sum(tf.cos(phi) * score, axis=-2)
            imag = tf.reduce_sum(tf.sin(phi) * score, axis=-2)
            vec.append(real)
            vec.append(imag)

        vec = tf.concat(vec, axis=-1)

        if self.residual:
            res = tf.nn.tanh(tf.reduce_sum(input * score, axis=-2))
            vec = tf.concat([vec, res], axis=-1)
            return vec
        return vec

    def get_config(self):
        config = super(MultiACTS, self).get_config()
        config.update({"encoder_dim": self.encoder_dim,
                       "n_sample_points": self.n_sample_points,
                       "minval": self.minval,
                       "maxval": self.maxval,
                       "l1": self.l1,
                       "train_theta": self.train_theta,
                       "activation": self.activation,
                       "residual": self.residual,
                       "learn_positional": self.learn_positional})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
