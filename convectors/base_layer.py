from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, cast

from .utils import camel_to_snake

# Type variable for Layer subclasses
T = TypeVar('T', bound='Layer')


class Layer:
    """
    Base class for all layers in the processing pipeline.
    
    This class defines the main methods and attributes that every layer should have.
    It provides functionality for connecting layers, saving/loading state, and
    processing data through the layer.

    Parameters
    ----------
    name : str, optional
        The name of the layer. If not given, the name will be derived from
        the class name using snake_case conversion.
    verbose : bool, default=False
        If True, the layer will output verbose messages during execution.

    Attributes
    ----------
    name : str
        The name of the layer.
    verbose : bool
        Whether the layer should output verbose messages.
    _children : List[Layer]
        List of child layers. These are the layers that take the output of this
        layer as their input.
    _parents : List[Layer]
        List of parent layers. These are the layers whose output is taken as
        input for this layer.
    _trained : bool
        Whether the layer has been trained.
    """

    def __init__(self, name: Optional[str] = None, verbose: bool = False) -> None:
        self.name: str = name or camel_to_snake(self.__class__.__name__)
        self.verbose: bool = verbose
        self._children: List['Layer'] = []
        self._parents: List['Layer'] = []
        self._trained: bool = False

    # ---------------------------------------
    # Properties
    # ---------------------------------------

    @property
    def _trainable(self) -> bool:
        """
        Check if the layer is trainable.
        
        A layer is considered trainable if it has a 'fit' method.
        
        Returns
        -------
        bool
            True if the layer is trainable, False otherwise.
        """
        return hasattr(self, "fit")

    # ---------------------------------------
    # Methods to be implemented by subclasses
    # ---------------------------------------

    def _unload(self) -> None:
        """
        Unload any resources used by the layer.
        
        This method is called before saving the layer to a file.
        Subclasses should override this method if they need to release
        resources before serialization.
        """
        pass

    def _reload(self, **kwargs: Any) -> None:
        """
        Reload any resources used by the layer.
        
        This method is called after loading the layer from a file.
        Subclasses should override this method if they need to reload
        resources after deserialization.
        
        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.
        """
        pass

    # ---------------------------------------
    # Private methods
    # ---------------------------------------

    def fit_predict(self, *x: Any, **kwargs: Any) -> Any:
        """
        Fit the layer to the data and then predict.
        
        Parameters
        ----------
        *x : Any
            The input data to fit and predict.
        **kwargs : Any
            Additional keyword arguments.
            
        Returns
        -------
        Any
            The predicted data.
        """
        # if length of x is 1, fit with x[0]
        if len(self._parents) <= 1:
            if hasattr(self, 'fit'):
                self.fit(x[0])  # type: ignore
            self._trained = True
            res = self.predict(x[0])
        else:  # else, fit with all x
            if hasattr(self, 'fit'):
                self.fit(*x)  # type: ignore
            res = self.predict(*x)
        return res

    def predict(self, *x: Any, **kwargs: Any) -> Any:
        """
        Predict using the layer.
        
        Parameters
        ----------
        *x : Any
            The input data to predict.
        **kwargs : Any
            Additional keyword arguments.
            
        Returns
        -------
        Any
            The predicted data.
        """
        if hasattr(self, "process_document"):
            process_doc = cast(Callable, self.process_document)
            if len(self._parents) <= 1:
                res = [process_doc(it) for it in x[0]]
            else:
                res = [process_doc(*it) for it in zip(*x)]
        elif hasattr(self, "process_documents"):
            process_docs = cast(Callable, self.process_documents)
            if len(self._parents) <= 1:
                res = process_docs(x[0])
            else:
                res = process_docs(*x)
        else:
            raise NotImplementedError(
                f"Layer {self.name} does not implement 'process_document' or 'process_documents'"
            )
        return res

    # ---------------------------------------
    # Public methods
    # ---------------------------------------

    def input(
        self: T, layer: Union["Layer", List["Layer"], Tuple["Layer", ...]]
    ) -> T:
        """
        Bind input layer(s) to this layer.
        
        This method connects the given layer(s) as input to the current layer.
        The output of the input layer(s) will be used as input for this layer.
        
        Parameters
        ----------
        layer : Layer or List[Layer] or Tuple[Layer, ...]
            The layer or layers to be bound as input to this layer.

        Returns
        -------
        Layer
            The current layer with the bound input layer(s).
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
        Save the layer to a file.
        
        This method serializes the layer to a file using dill.
        
        Parameters
        ----------
        filename : str
            The name of the file to save the layer to.
        """
        import dill

        # Save copies of parents and children
        parents_copy = self._parents[:]
        children_copy = self._children[:]
        
        # Clear parents and children for serialization
        self._parents = []
        self._children = []
        
        # Unload resources
        self._unload()
        
        # Save to file
        with open(filename, "wb") as f:
            dill.dump(self, f)
        
        # Restore parents and children
        self._parents = parents_copy
        self._children = children_copy

    # ---------------------------------------
    # Overloaded methods
    # ---------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Process data through the layer.
        
        If the input arguments are strings, they are wrapped in lists before
        processing and a single element is returned. Otherwise, a list of
        processed elements is returned.

        Parameters
        ----------
        *args : Any
            The input data to process.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The processed data. If input was a single string, a single
            processed element is returned. Otherwise, a list of processed
            elements is returned.
        """
        # Check if any argument is a string (for single element processing)
        _return_single_element = False
        for arg in args:
            if isinstance(arg, str):
                _return_single_element = True
                break
                
        # Wrap strings in lists
        if _return_single_element:
            args = tuple([arg] for arg in args)

        # Process the data
        if self._trainable and not self._trained:
            res = self.fit_predict(*args, **kwargs)
        else:
            res = self.predict(*args, **kwargs)

        # Return single element if necessary
        if _return_single_element and isinstance(res, list) and len(res) > 0:
            return res[0]
        return res

    def __repr__(self) -> str:
        """
        Get a string representation of the layer.
        
        Returns
        -------
        str
            The name of the layer.
        """
        return f"{self.name}"

    def __add__(self, layer: Union["Layer", "Sequential"]) -> "Sequential":
        """
        Create a Sequential model by adding this layer and another layer.
        
        Parameters
        ----------
        layer : Layer or Sequential
            The layer to add after this layer.

        Returns
        -------
        Sequential
            A new Sequential model containing this layer and the added layer.
        """
        from .models import Sequential

        if isinstance(layer, Sequential):
            model = Sequential([self] + layer._layers[1:])
        else:
            model = Sequential([self, layer])
        return model

    def __iadd__(self, layer: "Layer") -> "Sequential":
        """
        Create a Sequential model by adding this layer and another layer in-place.
        
        Parameters
        ----------
        layer : Layer
            The layer to add after this layer.

        Returns
        -------
        Sequential
            A new Sequential model containing this layer and the added layer.
        """
        from .models import Sequential

        model = Sequential([self, layer])
        return model


class Input(Layer):
    """
    Input layer for a processing pipeline.
    
    This class represents the input layer of a pipeline. It handles type conversion
    and length truncation of input data.

    Parameters
    ----------
    max_length : int, optional
        The maximum length for each data item. If specified, items will be
        truncated to this length.
    dtype : type, default=str
        The data type that the input data should be converted to.
    force_dtype : bool, default=True
        If True, the input data will always be converted to `dtype`.
    name : str, optional
        The name of the layer. If not given, the name will be derived from
        the class name.

    Attributes
    ----------
    _max_length : Optional[int]
        The maximum length for each data item.
    _dtype : type
        The data type that the input data should be converted to.
    _force_dtype : bool
        Whether to force conversion to the specified data type.
    """

    _trainable = False

    def __init__(
        self, 
        max_length: Optional[int] = None, 
        dtype: type = str, 
        force_dtype: bool = True, 
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self._max_length = max_length
        self._dtype = dtype
        self._force_dtype = force_dtype

    def process_document(self, x: Any) -> Any:
        """
        Process a single document by casting its type and truncating its length.

        Parameters
        ----------
        x : Any
            The document to process.

        Returns
        -------
        Any
            The processed document with type conversion and length truncation applied.
        """
        # Convert type if needed
        if self._force_dtype and not isinstance(x, self._dtype):
            x = self._dtype(x)
            
        # Truncate length if needed
        if self._max_length is not None:
            x = x[:self._max_length]  # Fixed slicing syntax
            
        return x
