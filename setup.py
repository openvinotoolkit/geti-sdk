import setuptools

with open("README.md", "r", encoding="utf-8") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

REQUIRED_PACKAGES: List[str] = []

with open("requirements.txt", 'r', encoding="utf-8") as requirements_file:
    for line in requirements_file:
        requirement = line.strip()
        if requirement and not requirement.startswith(('#', '-f')):
            REQUIRED_PACKAGES.append(requirement)

with open("sc_api_tools/__init__.py", 'r', encoding="utf-8") as init_file:
    for line in init_file:
        line = line.strip()
        if line.startswith("__version__")
            VERSION = line.split('='')[0].strip()

setuptools.setup(
    name="sc-api-tools",
    version=VERSION,
    author="Ludo Cornelissen",
    author_email="ludo.cornelissen@intel.com",
    description="A module for interacting with the Sonoma Creek REST API",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools",
    project_urls={
        "Bug Tracker": "https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["sc_api_tools"],
    python_requires=">=3.8",
    install_requires=REQUIRED_PACKAGES
)