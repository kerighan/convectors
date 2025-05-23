import os
import setuptools

# Read the contents of the README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="convectors",
    version="1.0.1",
    author="Maixent Chenebaux",
    author_email="maixent.chenebaux@gmail.com",
    description="A flexible NLP processing pipeline library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kerighan/convectors",
    project_urls={
        "Bug Tracker": "https://github.com/kerighan/convectors/issues",
        "Documentation": "https://github.com/kerighan/convectors",
        "Source Code": "https://github.com/kerighan/convectors",
    },
    packages=setuptools.find_packages(),
    install_requires=[
        "nltk>=3.8.1",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
        "pandas",
        "dill",
        "emoji",
        "tiktoken",
        "unidecode",
        "networkx",
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    keywords="nlp, natural language processing, machine learning, text processing, vectorization",
)
