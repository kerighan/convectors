from .utils import camel_to_snake
from typing import Any, List, Optional, Union, Tuple


class Layer:
    """
    This is the base class for all layers in the pipeline. It defines the main
    methods and attributes that every layer should have.

    Parameters
    ----------
    name : str, optional
        The name of the layer. If not given, the name will be derived from
        the class name.
    verbose : bool, optional
        If True, the layer will output verbose messages during execution.
        Default is False.

    Attributes
    ----------
    _children : list
        List of child layers. These are the layers that take the output of this
        layer as their input.
    _parents : list
        List of parent layers. These are the layers whose output is taken as
        input for this layer.

    """

    def __init__(self, name: Optional[str] = None, verbose: bool = False) -> None:
        self.name: str = name or camel_to_snake(self.__class__.__name__)
        self.verbose: bool = verbose
        self._children: List[Layer] = []
        self._parents: List[Layer] = []
        self._trained: bool = False

    # ---------------------------------------
    # Properties
    # ---------------------------------------

    @property
    def _trainable(self) -> bool:
        return hasattr(self, "fit")

    # ---------------------------------------
    # Methods to be implemented by subclasses
    # ---------------------------------------

    def _unload(self) -> None:
        pass

    def _reload(self, **_: Any) -> None:
        pass

    # ---------------------------------------
    # Private methods
    # ---------------------------------------

    def fit_predict(self, *x: Any, **kwargs) -> Any:
        # if length of x is 1, fit with x[0]
        if len(self._parents) <= 1:
            self.fit(x[0])
            self._trained = True
            res = self.predict(x[0])
        else:  # else, fit with all x
            self.fit(*x)
            res = self.predict(*x)
        return res

    def predict(self, *x: Any, **kwargs) -> Any:
        if hasattr(self, "process_document"):
            if len(self._parents) <= 1:
                res = [self.process_document(it) for it in x[0]]
            else:
                res = [self.process_document(*it) for it in zip(*x)]
        elif hasattr(self, "process_documents"):
            if len(self._parents) <= 1:
                res = self.process_documents(x[0])
            else:
                res = self.process_documents(*x)
        return res

    # ---------------------------------------
    # Public methods
    # ---------------------------------------

    def input(
        self, layer: Union["Layer", List["Layer"], Tuple["Layer", ...]]
    ) -> "Layer":
        """
        Binds the given layer or layers to the current layer. The output of the
        current layer will be used as input for the bound layer or layers.
        Also, the bound layer or layers are added to the parents of the current
        layer, and the current layer is added to the children of the bound
        layer or layers.

        Parameters
        ----------
        layer : Layer or list of Layers or tuple of Layers
            The layer or layers to be bound to the current layer.

        Returns
        -------
        Layer
            The current layer with the bound layer or layers.

        """
        self._parents = []
        if isinstance(layer, (list, tuple)):
            for _layer in layer:
                self._parents.append(_layer)
                _layer._children.append(self)
        else:
            self._parents.append(layer)
            layer._children.append(self)
        return self

    def save(self, filename: str) -> None:
        """
        Save the state of the layer to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the layer to.
        """
        import dill

        parents_copy = self._parents[:]
        children_copy = self._children[:]
        self._parents = []
        self._children = []
        self._unload()
        with open(filename, "wb") as f:
            dill.dump(self, f)
        self._parents = parents_copy
        self._children = children_copy

    # ---------------------------------------
    # Overloaded methods
    # ---------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Makes the class instance callable and applies the `fit_predict` method
        on the arguments. If the input arguments are strings, it wraps them
        into lists before processing and returns a single element. Otherwise,
        it returns a list of processed elements.

        Parameters
        ----------
        *args
            Variable length argument list. It is expected to contain data to be
            processed by the `fit_predict` method.
        **kwargs
            Arbitrary keyword arguments. These will be passed directly to the
            `fit_predict` method.

        Returns
        -------
        processed : list or single element
            The processed data. If input was a single string, a single
            processed element is returned. Otherwise, a list of processed
            elements is returned.
        """
        # if the arguments are strings, we assume that the user wants to
        # predict a single element
        _return_single_element = False
        for arg in args:
            if isinstance(arg, str):
                _return_single_element = True
        if _return_single_element:
            args = ([arg] for arg in args)

        # fit predict input
        if self._trainable and not self._trained:
            res = self.fit_predict(*args, **kwargs)
        else:
            res = self.predict(*args, **kwargs)

        # return single element if necessary
        if _return_single_element:
            return res[0]
        return res

    def __repr__(self) -> str:
        return f"{self.name}"


class Input(Layer):
    """
    This class represents the input layer of a pipeline. It inherits from the
    Layer class and overrides some of its methods to handle input data.

    Parameters
    ----------
    max_length : int, optional
        The maximum length for each data item. If specified, items will be
        truncated to this length.
    dtype : type, optional
        The data type that the input data should be converted to.
        Default is str.
    force_dtype : bool, optional
        If True, the input data will always be converted to `dtype`.
    name : str, optional
        The name of the layer. If not given, the name will be derived from
        the class name.

    """

    _trainable = False

    def __init__(self, max_length=None, dtype=str, force_dtype=True, name=None):
        super().__init__(name)
        self._max_length = max_length
        self._dtype = dtype
        self._force_dtype = force_dtype

    def _process_document(self, x):
        """
        Process a single document by casting its type and truncating its
        length.

        Parameters
        ----------
        x : object
            The document to process.

        Returns
        -------
        processed : object
            The processed document.
        """
        if self._force_dtype and not isinstance(x, self._dtype):
            x = self._dtype(x)
        if self._max_length is not None:
            x = x[self._max_length]
        return x
