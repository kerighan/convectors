from ..base_layer import Layer
from ..utils import to_matrix
from typing import Any, Optional
import numpy as np


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


class Keras(Layer):
    def __init__(
        self,
        model,
        batch_size=128,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name, verbose)
        self._model = model
        self._batch_size = batch_size
        self.kwargs = kwargs

        # Check if model is a tflite model
        self._is_tflite_model = isinstance(model, bytes)
        if self._is_tflite_model:
            import tensorflow as tf

            self._interpreter = tf.lite.Interpreter(model_content=model)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()[0]["index"]
            self._inp_dtype = self._interpreter.get_input_details()[0]["dtype"]
            self._output_details = self._interpreter.get_output_details()[0]["index"]

    def _unload(self):
        if not self._is_tflite_model:
            self.weights = self._model.get_weights()
            self.config = self._model.get_config()
            del self._model
        else:
            del self._interpreter

    def _reload(self, custom_objects=None, **_):
        if not self._is_tflite_model:
            from tensorflow.keras import models

            try:
                model = models.Model.from_config(
                    self.config, custom_objects=custom_objects
                )
            except KeyError:
                model = models.Sequential.from_config(
                    self.config, custom_objects=custom_objects
                )
            model.set_weights(self.weights)
            del self.weights
            del self.config
            self._model = model
        else:
            import tensorflow as tf

            self._interpreter = tf.lite.Interpreter(model_content=self._model)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()[0]["index"]
            self._inp_dtype = self._interpreter.get_input_details()[0]["dtype"]
            self._output_details = self._interpreter.get_output_details()[0]["index"]

    def _predict_tflite(self, X):
        output_tensors = []
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X_shape = list(X.shape)
        last_batch_size = -1
        for chunk in chunked(X, self._batch_size):
            # Resize the input tensor if necessary
            if last_batch_size != len(chunk):
                X_shape[0] = len(chunk)
                self._interpreter.resize_tensor_input(self._input_details, (X_shape))
                last_batch_size = len(chunk)
                self._interpreter.allocate_tensors()

            # Prepare the input tensor for prediction
            input_tensor = np.array(chunk, dtype=self._inp_dtype)
            self._interpreter.set_tensor(0, input_tensor)

            # Run the prediction and get the output tensor
            self._interpreter.invoke()
            out = self._interpreter.get_tensor(self._output_details)
            output_tensors.extend(list(out))
        return np.array(output_tensors)

    def process_documents(self, series: Any) -> Any:
        # convert to matrix
        if not isinstance(series, np.ndarray):
            from scipy.sparse import issparse

            X = to_matrix(series)
            if issparse(X):
                X = np.array(X.todense())
        else:
            X = series

        # predict using either tflite of standard keras model
        if not self._is_tflite_model:
            return self._model.predict(
                X, batch_size=self._batch_size, verbose=False, **self.kwargs
            )
        else:
            return self._predict_tflite(X)
