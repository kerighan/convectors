from ..base_layer import Layer
from typing import Optional


class Lambda(Layer):
    def __init__(
        self,
        func,
        axis=1,
        name: Optional[str] = None,
        verbose: bool = False
    ):
        super().__init__(name, verbose)
        if axis == 1:
            self.process_document = func
        elif axis == 0:
            self.process_documents = func
