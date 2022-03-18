# !/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np


def get_db(db_type, db_filename):
    if db_type == "sqlitedict":
        from sqlitedict import SqliteDict
        db = SqliteDict(db_filename)
    elif db_type == "lmdb":
        import lmdb
        env = lmdb.open(db_filename)
        db = env.begin()

    return db


def load_picture(db_type, db, x):
    import pickle

    if db_type == "sqlitedict":
        try:
            res = pickle.loads(db[x.key])
            return res
        except AttributeError:
            return np.zeros(shape=(1, 1, 1))

    elif db_type == "lmdb":
        try:
            res = pickle.loads(db.get(x.key.encode("ascii")))
            return res
        except AttributeError:
            print('error')
            return np.zeros(shape=(1, 1, 1))


def resize(x, image_size):
    import cv2
    if x.size == 1:
        res = np.empty(shape=(image_size, image_size, 3))
        res.fill(np.nan)
        return res

    x = cv2.resize(x, dsize=(
        image_size, image_size), interpolation=cv2.INTER_CUBIC)

    return x
