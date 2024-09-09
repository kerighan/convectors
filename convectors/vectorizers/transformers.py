from typing import Any, Optional
from tqdm import tqdm
from ..base_layer import Layer
import numpy as np


class HuggingFaceVectorizer(Layer):
    def __init__(
        self,
        model_name: str = "multilingual-e5-small",
        batch_size: int = 64,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self._batch_size = batch_size
        if model_name == "multilingual-e5-small":
            import torch.nn.functional as F

            from torch import Tensor
            from transformers import AutoTokenizer, AutoModel

            tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
            model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

            def average_pool(
                last_hidden_states: Tensor, attention_mask: Tensor
            ) -> Tensor:
                last_hidden = last_hidden_states.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0
                )
                return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            def _vectorize(input_texts):
                batch_dict = tokenizer(
                    input_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                outputs = model(**batch_dict)
                embeddings = average_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                return embeddings

            self._vectorize = _vectorize

    def process_documents(self, series: Any) -> np.ndarray:
        X = []
        iterator = range(0, len(series), self._batch_size)
        if self.verbose:
            iterator = tqdm(iterator, desc="Processing documents")
        for i in iterator:
            chunk = series[i : i + self._batch_size]
            X.extend(list(self._vectorize(chunk).detach().numpy()))
        #     # load images
        #     images = np.array([self._load_image(image) for image in chunk])
        #     vectors = self._model.predict(
        #         images, verbose=0, batch_size=self._batch_size
        #     )
        #     X.extend(list(vectors))
        return np.array(X)
