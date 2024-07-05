import networkx as nx
import pandas as pd


class Model:
    def __init__(
        self,
        inputs=None,
        outputs=None,
        name=None,
    ):
        self.name = name or self.__class__.__name__
        # generate graph from inputs and outputs
        self._graph = self._generate_graph(inputs, outputs)

    # ---------------------------------------
    # Private methods
    # ---------------------------------------

    def _generate_graph(self, inputs, outputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        # store inputs and outputs
        self._inputs = inputs
        self._outputs = outputs

        # layers have a _children attribute where we can find the next layers
        # in the graph. We can use a breadth-first search to find all layers
        # connected to the inputs.
        edges = set()
        queue = []
        nodes = set()
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
        topological_sort = list(nx.topological_sort(G))
        # get layer names
        layer_names = [layer.name for layer in topological_sort]
        # change name of layers that are not unique
        occurrences = {}
        for i, layer in enumerate(topological_sort):
            layer_name = layer.name
            if layer_names.count(layer_name) > 1:
                if layer_name not in occurrences:
                    occurrences[layer_name] = 1
                layer.name = f"{layer_name}_{occurrences[layer_name]}"
                occurrences[layer_name] += 1
        self._output_layer_names = [layer.name for layer in outputs]
        self._input_layer_names = [layer.name for layer in inputs]
        self._topological_sort = topological_sort

        # call functions in topsort order
        remaining_calls = {
            layer.name: len(layer._children) for layer in self._topological_sort
        }
        for layer_name in self._output_layer_names:
            remaining_calls[layer_name] += 1

        # check that all layers are connected. If not, remove from topological
        layers_to_remove = []
        for layer_name, occ in remaining_calls.items():
            if occ == 0:
                layers_to_remove.append(layer_name)

        for layer_name in layers_to_remove:
            del remaining_calls[layer_name]
            for layer in self._topological_sort:
                if layer.name == layer_name:
                    self._topological_sort.remove(layer)
                    break

    def _call(self, *args, method="auto", batch_size=None):
        # get the next batch for all arguments by looping
        # (in case the data cpùes from a generator)
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

    def _call_on_batch(self, *args, method="auto"):
        # handle the case of single strings as inputs
        for arg in args:
            if isinstance(arg, str):
                _return_single_element = True
            else:
                _return_single_element = False
        # in this case, encapsulate the string in a list
        if _return_single_element:
            data = {name: [arg] for name, arg in zip(self._input_layer_names, args)}
        else:
            data = {name: arg for name, arg in zip(self._input_layer_names, args)}

        # call functions in topsort order
        remaining_calls = {
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
            for layer_name in remaining_calls:
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

    def fit_predict(self, *args, batch_size=None):
        return self._call(*args, method="fit_predict", batch_size=batch_size)

    def predict(self, *args, batch_size=None):
        return self._call(*args, method="predict", batch_size=batch_size)

    def fit(self, *args, batch_size=None):
        return self._call(*args, method="fit", batch_size=batch_size)

    def save(self, filename):
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

    def __call__(self, *args, batch_size=None):
        if batch_size is None:
            return self._call_on_batch(*args)
        else:
            return self._call(*args, batch_size=batch_size)

    def __getitem__(self, layer_name):
        for layer in self._topological_sort:
            if layer.name == layer_name:
                return layer
        raise KeyError(f"Layer {layer_name} not found")


class Sequential(Model):
    def __init__(self, layers):
        from .layers import Input

        layers = [Input()] + layers
        self._layers = layers
        for parent, child in zip(layers[:-1], layers[1:]):
            child.input(parent)
        inputs = [layers[0]]
        outputs = [layers[-1]]
        super().__init__(inputs, outputs)

    def add(self, layer):
        layer.input(self._layers[-1])
        self._layers.append(layer)
        super().__init__(self._layers[0], self._layers[-1])

    def __iadd__(self, layer):
        self.add(layer)
        return self

    def __add__(self, layer):
        new_model = Sequential(self._layers + [layer])
        return new_model


def load_model(filename):
    import dill

    with open(filename, "rb") as f:
        model = dill.load(f)
    if not isinstance(model, Model):
        return model
    for layer in model._topological_sort:
        layer._reload()
    return model
