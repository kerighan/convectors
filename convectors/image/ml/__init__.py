# !/usr/bin/env python
# -*- coding: utf8 -*-
from ... import Layer


class FaceAnalyzor(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        backend='mtcnn',
        batch_size=500,
        image_size=224,
        actions=['age', 'gender'],
        verbose=True,
        parallel=True
    ):
        super(FaceAnalyzor, self).__init__(
            input, output, name, verbose, parallel)
        # backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

        self.backend = backend
        self.actions = actions
        self.batch_size = batch_size
        self.image_size = image_size

    def process_series(self, series):
        from math import ceil

        from deepface.DeepFace import analyze

        from ..utils import get_db, load_picture

        db_type = series[0].db_type
        db_filename = series[0].db_filename

        db = get_db(db_type, db_filename)

        n_step = ceil(series.shape[0] / self.batch_size)
        res = {}
        for i in range(n_step):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            imgs = [load_picture(db_type, db, img)
                    for img in series[start:end]]
            features = analyze(
                imgs, actions=self.actions,
                detector_backend=self.backend, enforce_detection=False)

            for a in self.actions:
                if a not in res:
                    res[a] = [v[a] for k, v in features.items()]
                else:
                    res[a].append([v[a] for k, v in features.items()])

        return res


class FaceFinder(Layer):
    parallel = False
    trainable = False
    document_wise = False

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        image=None,
        batch_size=200,
        model='VGG-Face',
        metric="cosine",
        verbose=True,
        parallel=True
    ):
        super(FaceFinder, self).__init__(
            input, output, name, verbose, parallel)
        # models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
        # metrics = ["cosine", "euclidean", "euclidean_l2"]

        self.image = image
        self.batch_size = batch_size
        self.model = model
        self.metric = metric

    def process_series(self, series):
        from math import ceil

        from deepface.DeepFace import find

        from ..utils import get_db, load_picture

        db_type = series[0].db_type
        db_filename = series[0].db_filename

        db = get_db(db_type, db_filename)

        n_step = ceil(series.shape[0] / self.batch_size)
        res = []
        for i in range(n_step):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            imgs = [load_picture(db_type, db, img)
                    for img in series[start:end]]

            features = find(
                self.image, db_path=imgs, model_name=self.model,
                enforce_detection=False, distance_metric=self.metric)

            res.append(features)

        return res


class OCR(Layer):
    parallel = False
    trainable = False
    document_wise = True

    def __init__(
        self,
        input=None,
        output=None,
        name=None,
        batch_size=200,
        verbose=True,
        parallel=True
    ):
        super(OCR, self).__init__(
            input, output, name, verbose, parallel)

        self.batch_size = batch_size

    def process_doc(self, series):
        import easyocr

        from ..utils import get_db, load_picture

        reader = easyocr.Reader(['fr', 'en'], gpu=True)

        db_type = series.db_type
        db_filename = series.db_filename

        db = get_db(db_type, db_filename)
        img = load_picture(db_type, db, series)

        ocr = reader.readtext(img)

        return ' '.join([e[1] for e in ocr])
