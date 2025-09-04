from ..base_layer import Layer
from typing import Any, List, Optional


class AutoTokenize(Layer):
    """
    This class represents a layer that applies AutoTokenizer tokenization to the
    input using Hugging Face transformers. It inherits from the Layer class and 
    overrides the constructor to handle AutoTokenizer parameters.

    Parameters
    ----------
    model_name : str, optional
        The model name or path for the tokenizer. Default is 'google/gemma-3-270m-it'.
    offset : bool, optional
        If True, an offset of 1 is added to the tokenization. Default is True.
    name : str, optional
        The name of the layer. If not given, the name will be derived from the
        class name.
    verbose : bool, optional
        If True, the layer will output verbose messages during execution.
        Default is True.

    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-270m-it",
        offset: bool = True,
        name: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name, verbose)
        self._model_name: str = model_name
        self._offset: bool = offset
        self._reload()

    def _reload(self, **_: Any) -> None:
        from transformers import AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        
        # Set n_features to vocab size
        self.n_features = self._tokenizer.vocab_size

    def _unload(self) -> None:
        del self._tokenizer

    def process_document(self, doc: str) -> List[int]:
        """
        Tokenize a document and return input_ids.
        
        Parameters
        ----------
        doc : str
            The document to tokenize.
            
        Returns
        -------
        List[int]
            The tokenized input_ids.
        """
        # Tokenize and get input_ids
        tokens = self._tokenizer(doc, return_tensors=None)["input_ids"]
        
        if self._offset:
            return [token + 1 for token in tokens]
        return tokens

    def decode(self, doc: List[int]) -> str:
        """
        Decode token ids back to text.
        
        Parameters
        ----------
        doc : List[int]
            The token ids to decode.
            
        Returns
        -------
        str
            The decoded text.
        """
        if self._offset:
            tokens = [token - 1 for token in doc]
        else:
            tokens = doc
            
        return self._tokenizer.decode(tokens, skip_special_tokens=True)