from .. import Layer, to_matrix

# =============================================================================
# Layers
# =============================================================================


class Merge(Layer):
    parallel = True
    trainable = False
    document_wise = False
    multi = True

    def __init__(
        self,
        *args,
        merge_mode="concat",
        name=None,
        verbose=True,
        parallel=False
    ):
        super(Merge, self).__init__(None, None, name, verbose, parallel)
        self.inputs = args
        self.merge_mode = merge_mode

    def process_series(self, series, *args, y=None):
        import numpy as np
        assert len(args) == len(self.inputs) - 1
        if self.merge_mode == "concat":
            X = to_matrix(self.inputs[0](series))
            for i, data in enumerate(args, 1):
                X = np.hstack([X, self.inputs[i](data)])
            return X
        elif self.merge_mode == "sum":
            X = to_matrix(self.inputs[0](series))
            for i, data in enumerate(args, 1):
                X += self.inputs[i](data)
            return X
        else:
            raise ValueError(f"unknown merge_mode={self.merge_mode}")
