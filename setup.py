import setuptools


setuptools.setup(
    name="convectors",
    version="0.0.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy", "scikit-learn", "scipy",
        "tqdm", "pandas", "dill"
    ]
)
