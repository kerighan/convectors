# !/usr/bin/env python
# -*- coding: utf8 -*-
from dataclasses import dataclass

import numpy as np

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
        verbose=True,
        parallel=True
    ):
        super(Fetch, self).__init__(input, output, name, verbose, parallel)

        self.db_type = db_type
        self.db_dir = f"{db_dir}_{db_type}"

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

    def apply_lmdb(self, series):
        import lmdb

        filename = f"{self.db_dir}"
        map_size = 6 * int(1e9)  # max possible size of lmdb
        self.env = lmdb.open(filename, map_size=map_size)

        to_download = []
        with self.env.begin() as txn:
            keys_in = list(txn.cursor().iternext(values=False))

        for url in series:
            if url.encode("ascii") not in keys_in:
                to_download.append(url)

        def insert(url, img):
            with self.env.begin(write=True) as txn:
                txn.put(url.encode("ascii"), img)

        thread_pool_downloader(to_download, insert)

        # GET ALL IMAGES
        with self.env.begin() as txn:
            keys_in = list(txn.cursor().iternext(values=False))

        results = []
        for url in series:
            if url.encode("ascii") not in keys_in:
                results.append(np.nan)
            else:
                results.append(Picture(url, self.db_type, filename))
        return results

    # def apply_disk(self, series):
    #     from glob import glob
    #     filename = f"{self.db_dir}"

    #     self.env = [file for file in glob(f"{filename}/*")]

    def apply(self, series, y=None):
        if self.db_type == "sqlitedict":
            return self.apply_sqlitedict(series)

        if self.db_type == "lmdb":
            return self.apply_lmdb(series)

        if self.db_type == "disk":
            return self.apply_disk(series)


# =============================================================================
# Functions
# =============================================================================

def img_download(callback):
    import io
    import pickle

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
