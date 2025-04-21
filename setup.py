#!/usr/bin/env python
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="munl",
    version="1.0.0",
    author="Xavier Cadet",
    author_email="xfc17@ic.ac.uk",
    description="Deep Unlearn: Benchmarking Machine Unlearning for Image Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xcadet/deepunlearn",
    project_urls={
        "Bug Tracker": "https://github.com/xcadet/deepunlearn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["munl"],
    python_requires=">=3.10",
)
