from ..base_layer import Layer
from typing import Any, List, Optional


class Concatenate(Layer):
    """
    This class represents a layer that concatenates the inputs from multiple
    parent layers.
    It inherits from the Layer class and overrides the `_process_document`
    method to handle concatenation.

    Parameters
    ----------
    name : str, optional
        The name of the layer. If not given, the name will be derived from the
        class name.
    verbose : bool, optional
        If True, the layer will output verbose messages during execution.
        Default is True.

    """

    def __init__(
        self, name: Optional[str] = None, verbose: bool = False
    ) -> None:
        super().__init__(name, verbose)

    def process_document(self, *args: List[Any]) -> Any:
        x = args[0]
        for arg in args[1:]:
            x += arg
        return x
