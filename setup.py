from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="clearly_confused",
    version="0.0.1",
    description="A python confusion matrix plotter, displayed as a treemap",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArjanCodes/2023-package",
    author="shemla",
    author_email="ori.a.shemla@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas >= 2.2.3",
        "numpy >= 2.1.3"
        "matplotlib >= 3.9.3"
        ],
    extras_require={
        "dev": [],
    },
    python_requires=">=3.10",
)