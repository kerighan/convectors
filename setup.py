import setuptools

setuptools.setup(
    name="convectors",
    version="1.0.1",
    author="Maixent Chenebaux",
    description="NLP package",
    long_description_content_type="text/markdown",
    url="https://github.com/kerighan/convectors",
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
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
