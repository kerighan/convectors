"""
Basic usage examples for the convectors library.

This script demonstrates how to use the convectors library to build
NLP processing pipelines using both the sequential and functional APIs.
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from convectors.layers import (
    Input,
    Tokenize,
    SnowballStem,
    TfIdf,
    SVD,
    Concatenate,
    Tiktokenize,
)
from convectors.models import Model, Sequential


def sequential_api_example():
    """Example of using the Sequential API."""
    print("\n=== Sequential API Example ===")
    
    # Create a sequential model
    model = Sequential([
        Tokenize(),
        SnowballStem(),
        TfIdf(),
        SVD(n_components=10)
    ])
    
    # Sample data
    texts = [
        "Hello world! This is a test document.",
        "Another example document for processing.",
        "Natural language processing is fascinating."
    ]
    
    # Process the data
    vectors = model(texts)
    
    print(f"Input texts: {len(texts)} documents")
    print(f"Output vectors shape: {vectors.shape}")
    print(f"First vector: {vectors[0][:3]}...")  # Show first 3 components


def functional_api_example():
    """Example of using the Functional API."""
    print("\n=== Functional API Example ===")
    
    # Create input layer
    inp = Input()
    
    # Create processing layers
    tokenize = Tokenize().input(inp)
    stem = SnowballStem().input(tokenize)
    tfidf = TfIdf().input(stem)
    svd = SVD(n_components=10).input(tfidf)
    
    # Create model
    model = Model(inputs=inp, outputs=svd)
    
    # Sample data
    texts = [
        "Hello world! This is a test document.",
        "Another example document for processing.",
        "Natural language processing is fascinating."
    ]
    
    # Process the data
    vectors = model(texts)
    
    print(f"Input texts: {len(texts)} documents")
    print(f"Output vectors shape: {vectors.shape}")
    print(f"First vector: {vectors[0][:3]}...")  # Show first 3 components


def multiple_inputs_outputs_example():
    """Example of using multiple inputs and outputs."""
    print("\n=== Multiple Inputs and Outputs Example ===")
    
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
    tokens, tiktoken_ids = model(
        "Hello world! This is the first input.",
        "This is the second input for processing."
    )
    
    print(f"First output (concatenated tokens): {tokens[:5]}...")
    print(f"Second output (tiktoken IDs): {tiktoken_ids[:5]}...")


def real_world_example():
    """A more realistic example using the 20 newsgroups dataset."""
    print("\n=== Real-world Example with 20 Newsgroups Dataset ===")
    
    # Load a subset of the 20 newsgroups dataset
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=['alt.atheism', 'sci.space'],
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Create a pipeline for document vectorization
    inp = Input()
    tokenize = Tokenize().input(inp)
    stem = SnowballStem().input(tokenize)
    tfidf = TfIdf().input(stem)
    svd = SVD(n_components=20).input(tfidf)
    
    # Create model
    model = Model(inputs=inp, outputs=svd)
    
    # Process the data
    vectors = model(newsgroups.data[:100])  # Process first 100 documents
    
    print(f"Processed {len(vectors)} documents")
    print(f"Vector dimensionality: {vectors.shape[1]}")
    
    # Calculate average vector for each category
    category_0_indices = np.where(newsgroups.target[:100] == 0)[0]
    category_1_indices = np.where(newsgroups.target[:100] == 1)[0]
    
    avg_vector_0 = np.mean(vectors[category_0_indices], axis=0)
    avg_vector_1 = np.mean(vectors[category_1_indices], axis=0)
    
    print(f"Average vector for '{newsgroups.target_names[0]}': {avg_vector_0[:3]}...")
    print(f"Average vector for '{newsgroups.target_names[1]}': {avg_vector_1[:3]}...")


if __name__ == "__main__":
    sequential_api_example()
    functional_api_example()
    multiple_inputs_outputs_example()
    real_world_example()