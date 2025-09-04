import networkx as nx
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .base_layer import Layer


class Model:
    """
    A model that represents a directed acyclic graph (DAG) of layers.
    
    The Model class allows for the creation of complex processing pipelines
    by connecting multiple layers together. It handles the flow of data through
    the layers and ensures that the graph is a valid DAG.
    
    Parameters
    ----------
    inputs : Layer or List[Layer], optional
        The input layer(s) of the model.
    outputs : Layer or List[Layer], optional
        The output layer(s) of the model.
    name : str, optional
        The name of the model. If not provided, the class name is used.
    
    Attributes
    ----------
    name : str
        The name of the model.
    _inputs : List[Layer]
        The input layers of the model.
    _outputs : List[Layer]
        The output layers of the model.
    _topological_sort : List[Layer]
        The layers of the model in topological order.
    _input_layer_names : List[str]
        The names of the input layers.
    _output_layer_names : List[str]
        The names of the output layers.
    """
    
    def __init__(
        self,
        inputs: Optional[Union[Layer, List[Layer]]] = None,
        outputs: Optional[Union[Layer, List[Layer]]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        # generate graph from inputs and outputs
        self._graph = self._generate_graph(inputs, outputs)

    # ---------------------------------------
    # Private methods
    # ---------------------------------------

    def _generate_graph(self, inputs: Optional[Union[Layer, List[Layer]]], 
                        outputs: Optional[Union[Layer, List[Layer]]]) -> nx.DiGraph:
        """
        Generate a directed acyclic graph (DAG) from the input and output layers.
        
        Parameters
        ----------
        inputs : Layer or List[Layer], optional
            The input layer(s) of the model.
        outputs : Layer or List[Layer], optional
            The output layer(s) of the model.
            
        Returns
        -------
        nx.DiGraph
            The generated graph.
            
        Raises
        ------
        ValueError
            If the graph is not a DAG.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        # store inputs and outputs
        self._inputs: List[Layer] = inputs
        self._outputs: List[Layer] = outputs

        # layers have a _children attribute where we can find the next layers
        # in the graph. We can use a breadth-first search to find all layers
        # connected to the inputs.
        edges: Set[Tuple[Layer, Layer]] = set()
        queue: List[Layer] = []
        nodes: Set[Layer] = set()
        for input_layer in inputs:
            queue.append(input_layer)

        while queue:
            layer = queue.pop(0)
            if layer in nodes:
                continue
            # add node to graph
            nodes.add(layer)
            for child in layer._children:
                edges.add((layer, child))
                queue.append(child)
            for parent in layer._parents:
                edges.add((parent, layer))
                queue.append(parent)

        # create graph
        G = nx.DiGraph()
        G.add_edges_from(edges)
        # check that it is a DAG
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Graph is not a DAG")

        # use topsort to rename nodes that are of the same type
        # this is done by appending a number to the layer name
        # e.g. tokenize -> tokenize_2
        # this is done to avoid name collisions

        # get topological sort
        topological_sort: List[Layer] = list(nx.topological_sort(G))
        # get layer names
        layer_names: List[str] = [layer.name for layer in topological_sort]
        # change name of layers that are not unique
        occurrences: Dict[str, int] = {}
        for i, layer in enumerate(topological_sort):
            layer_name = layer.name
            if layer_names.count(layer_name) > 1:
                if layer_name not in occurrences:
                    occurrences[layer_name] = 1
                layer.name = f"{layer_name}_{occurrences[layer_name]}"
                occurrences[layer_name] += 1
        self._output_layer_names: List[str] = [layer.name for layer in outputs]
        self._input_layer_names: List[str] = [layer.name for layer in inputs]
        self._topological_sort: List[Layer] = topological_sort

        # call functions in topsort order
        remaining_calls: Dict[str, int] = {
            layer.name: len(layer._children) for layer in self._topological_sort
        }
        for layer_name in self._output_layer_names:
            remaining_calls[layer_name] += 1

        # check that all layers are connected. If not, remove from topological sort
        layers_to_remove: List[str] = []
        for layer_name, occ in remaining_calls.items():
            if occ == 0:
                layers_to_remove.append(layer_name)

        for layer_name in layers_to_remove:
            del remaining_calls[layer_name]
            for layer in self._topological_sort:
                if layer.name == layer_name:
                    self._topological_sort.remove(layer)
                    break
                    
        return G

    def _call(self, *args: Any, method: str = "auto", batch_size: Optional[int] = None) -> Any:
        """
        Process data in batches.
        
        Parameters
        ----------
        *args : Any
            The input data to process.
        method : str, default="auto"
            The method to use for processing. Can be "auto", "fit", "predict", or "fit_predict".
        batch_size : int, optional
            The size of each batch. If None, all data is processed at once.
            
        Yields
        ------
        Any
            The processed data for each batch.
        """
        # get the next batch for all arguments by looping
        # (in case the data comes from a generator)
        index = 0
        while True:
            batch = []
            for arg in args:
                if isinstance(arg, list):
                    arg_batch = arg[index : index + batch_size]
                elif isinstance(arg, pd.Series):
                    arg_batch = arg.iloc[index : index + batch_size]
                else:
                    arg_batch = []
                    for _ in range(batch_size):
                        try:
                            arg_batch.append(next(arg))
                        except StopIteration:
                            break
                if len(arg_batch) == 0:
                    return
                batch.append(arg_batch)
            index += batch_size

            # call on batch
            res = self._call_on_batch(*batch, method=method)
            # yield result
            yield res
            # check if we are done
            if len(batch[0]) == 0:
                break

    def _call_on_batch(self, *args: Any, method: str = "auto") -> Any:
        """
        Process a single batch of data.
        
        Parameters
        ----------
        *args : Any
            The input data to process.
        method : str, default="auto"
            The method to use for processing. Can be "auto", "fit", "predict", or "fit_predict".
            
        Returns
        -------
        Any
            The processed data.
        """
        # handle the case of single strings as inputs
        _return_single_element = False
        for arg in args:
            if isinstance(arg, str):
                _return_single_element = True
                break
        
        # in this case, encapsulate the string in a list
        if _return_single_element:
            data: Dict[str, Any] = {name: [arg] for name, arg in zip(self._input_layer_names, args)}
        else:
            data = {name: arg for name, arg in zip(self._input_layer_names, args)}

        # call functions in topsort order
        remaining_calls: Dict[str, int] = {
            layer.name: len(layer._children) for layer in self._topological_sort
        }
        for layer_name in self._output_layer_names:
            remaining_calls[layer_name] += 1
        for layer_name in self._input_layer_names:
            remaining_calls[layer_name] += 1

        # call functions in topsort order
        for layer in self._topological_sort:
            if len(layer._parents) == 0:
                antecedents = [layer]
            else:
                antecedents = layer._parents

            if layer.__class__.__name__ == "Input":
                remaining_calls[layer.name] -= 1
                continue

            if method == "auto":
                data[layer.name] = layer(*(data[parent.name] for parent in antecedents))
            elif method in {"fit_predict", "fit"} and layer._trainable:
                data[layer.name] = layer.fit_predict(
                    *(data[parent.name] for parent in antecedents)
                )
            else:
                data[layer.name] = layer.predict(
                    *(data[parent.name] for parent in antecedents)
                )

            # remove a call from the remaining calls of the parents
            for parent in antecedents:
                remaining_calls[parent.name] -= 1
            # garbage collect
            for layer_name in list(remaining_calls.keys()):
                if remaining_calls[layer_name] == 0:
                    del data[layer_name]
            remaining_calls = {
                layer_name: remaining
                for layer_name, remaining in remaining_calls.items()
                if remaining > 0
            }

        if method == "fit":
            return self
        if _return_single_element:
            res = [data[name][0] for name in self._output_layer_names]
        else:
            res = [data[name] for name in self._output_layer_names]
        if len(self._output_layer_names) == 1:
            return res[0]
        return res

    # ---------------------------------------
    # Public methods
    # ---------------------------------------

    def fit_predict(self, *args: Any, batch_size: Optional[int] = None) -> Any:
        """
        Fit the model to the data and then predict.
        
        Parameters
        ----------
        *args : Any
            The input data to fit and predict.
        batch_size : int, optional
            The size of each batch. If None, all data is processed at once.
            
        Returns
        -------
        Any
            The predicted data.
        """
        return self._call(*args, method="fit_predict", batch_size=batch_size)

    def predict(self, *args: Any, batch_size: Optional[int] = None) -> Any:
        """
        Predict using the model.
        
        Parameters
        ----------
        *args : Any
            The input data to predict.
        batch_size : int, optional
            The size of each batch. If None, all data is processed at once.
            
        Returns
        -------
        Any
            The predicted data.
        """
        return self._call(*args, method="predict", batch_size=batch_size)

    def fit(self, *args: Any, batch_size: Optional[int] = None) -> "Model":
        """
        Fit the model to the data.
        
        Parameters
        ----------
        *args : Any
            The input data to fit.
        batch_size : int, optional
            The size of each batch. If None, all data is processed at once.
            
        Returns
        -------
        Model
            The fitted model.
        """
        return self._call(*args, method="fit", batch_size=batch_size)

    def save(self, filename: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filename : str
            The name of the file to save the model to.
        """
        import dill

        for layer in self._topological_sort:
            layer._unload()

        with open(filename, "wb") as f:
            dill.dump(self, f)

        for layer in self._topological_sort:
            layer._reload()

    # ---------------------------------------
    # Overloaded methods
    # ---------------------------------------

    def __call__(self, *args: Any, batch_size: Optional[int] = None) -> Any:
        """
        Call the model on the input data.
        
        Parameters
        ----------
        *args : Any
            The input data to process.
        batch_size : int, optional
            The size of each batch. If None, all data is processed at once.
            
        Returns
        -------
        Any
            The processed data.
        """
        if batch_size is None:
            return self._call_on_batch(*args)
        else:
            return self._call(*args, batch_size=batch_size)

    def __getitem__(self, layer_name: str) -> Layer:
        """
        Get a layer by name.
        
        Parameters
        ----------
        layer_name : str
            The name of the layer to get.
            
        Returns
        -------
        Layer
            The layer with the given name.
            
        Raises
        ------
        KeyError
            If no layer with the given name exists.
        """
        for layer in self._topological_sort:
            if layer.name == layer_name:
                return layer
        raise KeyError(f"Layer {layer_name} not found")


class Sequential(Model):
    """
    A sequential model that represents a linear stack of layers.
    
    The Sequential model is a linear stack of layers, where each layer
    has exactly one input and one output. It is a simplified version of
    the Model class.
    
    Parameters
    ----------
    layers : List[Layer]
        The layers to add to the model.
        
    Attributes
    ----------
    _layers : List[Layer]
        The layers of the model.
    """
    
    def __init__(self, layers: List[Layer]) -> None:
        from .layers import Input

        layers = [Input()] + layers
        self._layers: List[Layer] = layers
        for parent, child in zip(layers[:-1], layers[1:]):
            child.input(parent)
        inputs = [layers[0]]
        outputs = [layers[-1]]
        super().__init__(inputs, outputs)

    def add(self, layer: Layer) -> None:
        """
        Add a layer to the model.
        
        Parameters
        ----------
        layer : Layer
            The layer to add to the model.
        """
        layer.input(self._layers[-1])
        self._layers.append(layer)
        super().__init__(self._layers[0], self._layers[-1])

    def __iadd__(self, layer: Layer) -> "Sequential":
        """
        Add a layer to the model in-place.
        
        Parameters
        ----------
        layer : Layer
            The layer to add to the model.
            
        Returns
        -------
        Sequential
            The model with the added layer.
        """
        self.add(layer)
        return self

    def __add__(self, layer: Layer) -> "Sequential":
        """
        Add a layer to the model and return a new model.
        
        Parameters
        ----------
        layer : Layer
            The layer to add to the model.
            
        Returns
        -------
        Sequential
            A new model with the added layer.
        """
        new_model = Sequential(self._layers[1:] + [layer])
        return new_model


def load_model(filename: str) -> Any:
    """
    Load a model from a file.
    
    Parameters
    ----------
    filename : str
        The name of the file to load the model from.
        
    Returns
    -------
    Any
        The loaded model.
    """
    import dill

    with open(filename, "rb") as f:
        model = dill.load(f)
    if not isinstance(model, Model):
        return model
    for layer in model._topological_sort:
        layer._reload()
    return model
