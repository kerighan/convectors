import setuptools
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="convectors",
    version="0.0.1",
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
