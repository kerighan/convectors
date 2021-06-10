import os

import setuptools

# The directory containing this file
README_FILENAME = os.path.join(os.path.dirname(__file__), "README.md")
with open(README_FILENAME) as f:
    README = f.read()

setuptools.setup(
    name="convectors",
    version="0.0.4",
    author="Maixent Chenebaux",
    description=("End-to-end NLP package for seamless integration of "
                 "Pandas Series, DataFrame and Keras model"),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/kerighan/convectors",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy", "scikit-learn", "scipy",
        "tqdm", "pandas", "dill"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
