import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sc-api-tools",
    version="0.0.1",
    author="Ludo Cornelissen",
    author_email="ludo.cornelissen@intel.com",
    description="A module for interacting with the Sonoma Creek REST API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["sc_api_tools"],
    python_requires=">=3.8",
)