from ..base_layer import Layer
import numpy as np
from typing import Any, Optional
from tqdm import tqdm


class EfficientNet(Layer):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        include_top: bool = False,
        weights: Optional[str] = "imagenet",
        pooling: Optional[str] = "avg",
        batch_size: int = 64,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        if model_name == "efficientnet-b0":
            from tensorflow.keras.applications import EfficientNetB0 as EffNet

            size = 224
        elif model_name == "efficientnet-b1":
            from tensorflow.keras.applications import EfficientNetB1 as EffNet

            size = 240
        elif model_name == "efficientnet-b2":
            from tensorflow.keras.applications import EfficientNetB2 as EffNet

            size = 260
        elif model_name == "efficientnet-b3":
            from tensorflow.keras.applications import EfficientNetB3 as EffNet

            size = 300

        self._image_size = size
        self._batch_size = batch_size
        self._model = EffNet(
            input_shape=(self._image_size, self._image_size, 3),
            include_top=include_top,
            weights=weights,
            pooling=pooling,
        )

    def _load_image(self, image: Any) -> Any:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        try:
            img = load_img(image, target_size=(self._image_size, self._image_size))
            return img_to_array(img)
        except Exception as e:
            return np.zeros((self._image_size, self._image_size, 3))

    def process_documents(self, series: Any) -> np.ndarray:
        X = []
        iterator = range(0, len(series), self._batch_size)
        if self.verbose:
            iterator = tqdm(iterator, desc="Processing documents")
        for i in iterator:
            chunk = series[i : i + self._batch_size]
            # load images
            images = np.array([self._load_image(image) for image in chunk])
            vectors = self._model.predict(
                images, verbose=0, batch_size=self._batch_size
            )
            X.extend(list(vectors))
        return np.array(X)

    def process_images(self, images: Any) -> Any:
        return self._model.predict(images)

    @property
    def n_features(self) -> int:
        return self._model.output_shape[-1]
