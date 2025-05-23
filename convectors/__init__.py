"""
Convectors: A flexible NLP processing pipeline library.

Convectors is a Python library for building flexible NLP processing pipelines.
It provides a set of layers that can be connected to form complex processing
graphs, with support for tokenization, vectorization, dimensionality reduction,
and more.

The library is designed to be modular and extensible, allowing users to create
custom layers and integrate them into the pipeline.

Main components:
- layers: Various processing layers (tokenization, vectorization, etc.)
- models: Model classes for connecting layers into processing pipelines
- algorithms: Implementations of various NLP algorithms
"""

from .layers import *
from .models import Model, Sequential, load_model

__version__ = "1.0.1"