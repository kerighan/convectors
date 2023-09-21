from ..base_layer import Layer
from typing import Any, List, Optional, Set


class Tiktokenize(Layer):
    """
    This class represents a layer that applies Tiktoken tokenization to the
    input. It inherits from the Layer class and overrides the constructor
    to handle Tiktoken tokenization parameters.

    Parameters
    ----------
    encoding : str, optional
        The encoding used by Tiktoken. Default is 'p50k_base'.
    offset : bool, optional
        If True, an offset of 1 is added to the tokenization. Default is True.
    special_tokens : list, optional
        A list of special tokens to be added to the encoding. Default is an
        empty list.
    name : str, optional
        The name of the layer. If not given, the name will be derived from the
        class name.
    verbose : bool, optional
        If True, the layer will output verbose messages during execution.
        Default is True.

    """

    def __init__(
        self,
        encoding: str = "p50k_base",
        offset: bool = True,
        special_tokens: Optional[List[str]] = [
            "<|startoftext|>",
            "<|delim|>",
            "<|sep|>",
            "<|mask|>",
            "<|unmask|>"
        ],
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self._encoding: str = encoding
        self._special_tokens: List[str] = special_tokens
        self._offset: bool = offset
        self._reload()

    def _reload(self, **_: Any) -> None:
        import tiktoken
        encoder = tiktoken.get_encoding(self._encoding)
        index = encoder.max_token_value

        # populate special tokens
        special_tokens = encoder._special_tokens
        self._allowed_special: Set[str] = {'<|endoftext|>'}
        for i, t in enumerate(self._special_tokens, 1):
            special_tokens[t] = index + i
            self._allowed_special.add(t)

        # create encoding
        self._enc = tiktoken.Encoding(
            name=f"{self._encoding}_upgrade",
            pat_str=encoder._pat_str,
            mergeable_ranks=encoder._mergeable_ranks,
            special_tokens=special_tokens)
        self.n_features = self._enc.n_vocab

    def _unload(self) -> None:
        del self._enc
        del self._allowed_special

    def process_document(self, doc: str) -> List[int]:
        if self._offset:
            return [
                it + 1
                for it in self._enc.encode(
                    doc, allowed_special=self._allowed_special)
            ]
        return self._enc.encode(doc)

    def decode(self, doc: List[int]) -> str:
        if self._offset:
            return self._enc.decode([it - 1 for it in doc])
        return self._enc.decode(doc)
