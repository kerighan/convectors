# !/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np

from ... import Layer


class Vectorize(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        image_size=150,
        batch_size=100,
        model='InceptionV3',
        verbose=True,
        parallel=True
    ):
        super(Vectorize, self).__init__(input, output, name, verbose, parallel)

        self.image_size = image_size
        self.batch_size = batch_size
        self.model_name = model

    def process_series(self, series):
        from math import ceil

        from ..utils import get_db, load_picture, resize

        preprocess_input, model = get_vectorizer(
            self.model_name, self.image_size)

        db_type = series[0].db_type
        db_filename = series[0].db_filename

        db = get_db(db_type, db_filename)

        n_step = ceil(series.shape[0] / self.batch_size)
        res = []
        for i in range(n_step):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            imgs = np.array([resize(load_picture(
                db_type, db, img), self.image_size)
                for img in series[start:end]])
            features = get_features(imgs, model, preprocess_input)
            res.append(features)
        res = np.vstack(res)
        return res

# =============================================================================
# Functions
# =============================================================================


def get_vectorizer(model_name, image_size):
    from tensorflow.keras import applications

    module_name = model_name.lower()
    if "v3" in module_name:
        module_name = module_name.replace("v3", "_v3")

    vectorizer = applications.__getattribute__(
        module_name).__getattribute__(model_name)
    preprocess_input = applications.__getattribute__(
        module_name.lower()).__getattribute__("preprocess_input")

    model = vectorizer(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(image_size, image_size, 3))
    return [preprocess_input, model]


def get_features(x, model, preprocess_input):
    x = preprocess_input(x)
    features = model.predict(x)

    return features
