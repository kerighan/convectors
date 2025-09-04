# Convectors

Convectors is a flexible NLP processing pipeline library for Python. It provides a set of layers that can be connected to form complex processing graphs, with support for tokenization, vectorization, dimensionality reduction, and more.

## Features

- **Modular Design**: Build complex NLP pipelines by connecting layers
- **Flexible API**: Support for both functional and sequential APIs
- **Rich Layer Collection**: Includes tokenization, vectorization, dimensionality reduction, and more
- **Extensible**: Easily create custom layers and integrate them into the pipeline
- **Serialization**: Save and load models for later use

## Installation

```bash
pip install convectors
```

## Quick Start

### Sequential API

```python
from convectors.layers import Tokenize, SnowballStem, TfIdf
from convectors.models import Sequential

# Create a sequential model
model = Sequential([
    Tokenize(),
    SnowballStem(),
    TfIdf()
])

# Process data
vectors = model(["Hello world!", "How are you?"])
```

### Functional API

```python
from convectors.layers import Input, Tokenize, SnowballStem, TfIdf
from convectors.models import Model

# Create input layer
inp = Input()

# Create processing layers
tokenize = Tokenize().input(inp)
stem = SnowballStem().input(tokenize)
tfidf = TfIdf().input(stem)

# Create model
model = Model(inputs=inp, outputs=tfidf)

# Process data
vectors = model(["Hello world!", "How are you?"])
```

### Multiple Inputs and Outputs

```python
from convectors.layers import Input, Tokenize, SnowballStem, Tiktokenize, Concatenate
from convectors.models import Model

# Create first input branch
inp_1 = Input()
tokenize_1 = Tokenize().input(inp_1)
stem_1 = SnowballStem().input(tokenize_1)

# Create second input branch
inp_2 = Input()
tokenize_2 = Tokenize().input(inp_2)
stem_2 = SnowballStem().input(tokenize_2)

# Merge branches
concat = Concatenate().input((stem_1, stem_2))

# Create another output
tiktoken = Tiktokenize().input(inp_2)

# Create model with multiple inputs and outputs
model = Model(inputs=[inp_1, inp_2], outputs=[concat, tiktoken])

# Process data
tokens, tiktoken_ids = model("Hello world!", "How are you?")
```

## Available Layers

### Tokenization
- `Tokenize`: Basic tokenization
- `SnowballStem`: Stemming using Snowball stemmer
- `Tiktokenize`: Tokenization using OpenAI's tiktoken
- `TokenMonster`: Tokenization using TokenMonster
- `NGrams`: Generate n-grams from tokens

### Vectorization
- `TfIdf`: TF-IDF vectorization
- `CountVectorizer`: Count vectorization
- `HashingVectorizer`: Hashing vectorization
- `BM25`: BM25 vectorization

### Dimensionality Reduction
- `SVD`: Singular Value Decomposition
- `PCA`: Principal Component Analysis
- `UMAP`: Uniform Manifold Approximation and Projection

### Nearest Neighbors
- `Annoy`: Approximate Nearest Neighbors
- `HNSW`: Hierarchical Navigable Small World
- `NNDescent`: Nearest Neighbor Descent
- `KDTree`: KD-Tree for nearest neighbor search

### Preprocessing
- `Prefix`: Add prefix to documents
- `Suffix`: Add suffix to documents
- `Pad`: Pad documents to a fixed length
- `DocumentSplitter`: Split documents into chunks
- `OneHot`: One-hot encoding
- `Sub`: Substitute text using regex
- `FindAll`: Find all occurrences of a pattern
- `Normalize`: Normalize text

### Other
- `Lambda`: Apply a custom function
- `Concatenate`: Concatenate outputs from multiple layers
- `Keras`: Use Keras models as layers

## Saving and Loading Models

```python
# Save model
model.save("model.pkl")

# Load model
from convectors.models import load_model
model = load_model("model.pkl")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.