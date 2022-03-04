# !/usr/bin/env python
# -*- coding: utf8 -*-
import io
import os
import pickle
from dataclasses import dataclass
from functools import wraps

import numpy as np
from PIL import Image

from ... import Layer


@dataclass
class Picture:
    key: str
    db_type: str
    db_filename: str


class Fetch(Layer):
    parallel = False
    trainable = True

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        db_dir="tmp",
        db_type="lmdb",
        keys=None,
        map_size=5*int(1e9),
        verbose=True,
        parallel=True
    ):
        super(Fetch, self).__init__(input, output, name, verbose, parallel)

        self.db_type = db_type
        self.db_dir = f"{db_dir}_{db_type}"
        self.map_size = map_size
        self.keys = keys

    def apply_sqlitedict(self, series):
        from sqlitedict import SqliteDict
        filename = f"{self.db_dir}.sqlite"
        self.env = SqliteDict(filename, autocommit=True)

        # INSERT NOT FOUND
        to_download = []
        for url in series:
            if url not in self.env:
                to_download.append(url)

        def insert(url, img):
            self.env[url] = img

        thread_pool_downloader(to_download, insert)

        # GET ALL IMAGES
        results = []
        for url in series:
            if url not in self.env:
                results.append(np.nan)
            else:
                results.append(Picture(url, self.db_type, filename))
        return results

    def apply(self, series, y=None):
        if self.db_type == "sqlitedict":
            return self.apply_sqlitedict(series)


class ImageDB():
    def __init__(self, image):
        # Dimensions of image for reconstruction
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


# =============================================================================
# Functions
# =============================================================================

def img_download(callback):
    import requests
    from PIL import Image

    def wrapper(url):
        try:
            res = requests.get(url, stream=True)
            img = np.array(Image.open(io.BytesIO(res.content)).convert('RGB'))
            img = pickle.dumps(img)
            callback(url, img)
        except Exception:
            return
    return wrapper


def thread_pool_downloader(series, callback):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        res = [executor.submit(img_download(callback), url) for url in series]
        concurrent.futures.wait(res)
