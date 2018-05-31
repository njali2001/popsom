import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="popsom",
    version="0.0.1",
    author="Li Yuan",
    author_email="li_yuan@my.uri.edu",
    description="A Popluation-based Self-Organizing Maps Package",
    long_description="Self-Organizing Maps is a type of artificial neural network for the visualization of high-dimensional data.",
    long_description_content_type="text/markdown",
    url="https://github.com/njali2001/popsom",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)