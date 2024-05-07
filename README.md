<div align="center">

[![python](https://img.shields.io/badge/python-3.9%2B-green)]()
[![openvino](https://img.shields.io/badge/openvino-2023.2.0-purple)](https://github.com/openvinotoolkit/openvino)
![Intel Geti](https://img.shields.io/badge/Intel%C2%AE%20Geti%E2%84%A2-1.5%2B-blue?link=https%3A%2F%2Fgeti.intel.com%2F)

![Pre-merge Tests Status](https://img.shields.io/github/actions/workflow/status/openvinotoolkit/geti-sdk/pre-merge-tests.yml?label=pre-merge%20tests&link=https%3A%2F%2Fgithub.com%2Fopenvinotoolkit%2Fgeti-sdk%2Factions%2Fworkflows%2Fpre-merge-tests.yml)
![Nightly Tests [Geti latest] Status](https://img.shields.io/github/actions/workflow/status/openvinotoolkit/geti-sdk/nightly-tests-geti-latest.yaml?label=nightly%20tests%20%5BGeti%20latest%5D&link=https%3A%2F%2Fgithub.com%2Fopenvinotoolkit%2Fgeti-sdk%2Factions%2Fworkflows%2Fnightly-tests-geti-latest.yaml)
![Nightly Tests [Geti develop] Status](https://img.shields.io/github/actions/workflow/status/openvinotoolkit/geti-sdk/nightly-tests-geti-develop.yaml?label=nightly%20tests%20%5BGeti%20develop%5D&link=https%3A%2F%2Fgithub.com%2Fopenvinotoolkit%2Fgeti-sdk%2Factions%2Fworkflows%2Fnightly-tests-geti-develop.yaml)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8329/badge)](https://www.bestpractices.dev/projects/8329)

</div>

---

# Introduction

Welcome to the Intel® Geti™ SDK! The [Intel® Geti™ platform](https://geti.intel.com) enables
teams to rapidly develop AI models. The platform reduces the time needed to build
models by easing the complexities of model development and harnessing greater
collaboration between teams. Most importantly, the platform unlocks faster
time-to-value for digitization initiatives with AI.

The Intel® Geti™ SDK is a python package which contains tools to interact with an
Intel® Geti™ server via the REST API. It provides functionality for:

- Project creation from annotated datasets on disk
- Project downloading (images, videos, configuration, annotations, predictions and models)
- Project creation and upload from a previous download
- Deploying a project for local inference with OpenVINO
- Getting and setting project and model configuration
- Launching and monitoring training jobs
- Media upload and prediction

This repository also contains a set of (tutorial style) Jupyter
[notebooks](https://github.com/openvinotoolkit/geti-sdk/tree/main/notebooks)
that demonstrate how to use the SDK. We highly recommend checking them out to get a
feeling for use cases for the package.

# Getting started

## Installation
Using an environment manager such as
[Anaconda](https://www.anaconda.com/products/individual) or
[venv](https://docs.python.org/3/library/venv.html) to create a new
Python environment before installing the Intel® Geti™ SDK and its requirements is
highly recommended.

> **NOTE**: If you have installed multiple versions of Python,
> use `py -3.9 venv -m <env_name>` when creating your virtual environment to specify
> a supported version (in this case 3.9). Once you activate the
> virtual environment <venv_path>/Scripts/activate, make sure to upgrade pip
> to the latest version `python -m pip install --upgrade pip wheel setuptools`.

### Python version compatibility
Make sure to set up your environment using one of the supported Python versions for your
operating system, as indicated in the table below.

|             | Python <= 3.8 | Python 3.9         | Python 3.10        | Python 3.11        | Python 3.12 |
|:------------|:-------------:|:------------------:|:------------------:|:------------------:|:-----------:|
| **Linux**   | :x:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:         |
| **Windows** | :x:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:         |
| **MacOS**   | :x:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:         |

Once you have created and activated a new environment, follow the steps below to install
the package.

### Installing from PyPI
Use `pip install geti-sdk` to install the SDK from the Python Package Index (PyPI). To
install a specific version (for instance v1.5.0), use the command
`pip install geti-sdk==1.5.0`

### Installing from the Git repo
1. Download or clone the repository and navigate to the root directory of the repo in
   your terminal.

2. **Base installation** Within this directory, install the SDK using `pip install .` This command will install the
   package and its base dependencies in your environment.

3. **Notebooks installation (Optional)** If you want to be able to run the notebooks, make sure to
   install the extra requirements using `pip install .[notebooks]` This will install both the
   SDK and all other dependencies needed to run the notebooks in your environment

4. **Development installation (Optional)** If you plan on running the tests or want to build the
   documentation, you can install the package extra requirements by doing for example
   `pip install -e .[dev]`

   The valid options for the extra requirements are `[dev, docs, notebooks]`,
   corresponding to the following functionality:

   - `dev` Install requirements to run the test suite on your local machine
   - `notebooks` Install requirements to run the Juypter notebooks in the `notebooks`
     folder in this repository.
   - `docs` Install requirements to build the documentation for the SDK from source on
     your machine

## Using the SDK
The SDK contains example code in various forms to help you get familiar with the package.

- [Code examples](USAGE_EXAMPLES.md#code-examples) are short snippets that demonstrate
  how to perform several common tasks. This also shows how to configure the SDK to
  connect to your Intel® Geti™ server.

- [Jupyter notebooks](USAGE_EXAMPLES.md#jupyter-notebooks) are tutorial style notebooks that cover
  pretty much the full SDK functionality. **These are the recommended way to get started
  with the SDK.**

- [Example scripts](USAGE_EXAMPLES.md#example-scripts) are more extensive scripts that cover more
  advanced usage than the code examples, have a look at these if you don't like Jupyter.

## High level API reference
The `Geti` class provides the following methods:

- `download_project` -- Downloads a project by project name.


- `upload_project` -- Uploads project from a folder.


- `download_all_projects` -- Downloads all projects found on the server.


- `upload_all_projects` -- Uploads all projects found in a specified folder to the
  server.


- `upload_and_predict_image` -- Uploads a single image to an existing project on the
  server, and requests a prediction for that image. Optionally, the prediction can
  be visualized as an overlay on the image.


- `upload_and_predict_video` -- Uploads a single video to an existing project on the
  server, and requests predictions for the frames in the video. As with
  upload_and_predict_image, the predictions can be visualized on the frames. The
  parameter `frame_stride` can be used to control which frames are extracted for
  prediction.


- `upload_and_predict_media_folder` -- Uploads all media (images and videos) from a
  folder on local disk to an existing project on the server, and download
  predictions for all uploaded media.


- `deploy_project` -- Downloads the active model for all tasks in the project as an
  OpenVINO inference model. The resulting `Deployment` can be used to run inference
  for the project on a local machine. Pipeline inference is also supported.


- `create_project_single_task_from_dataset` -- Creates a single task project on the
  server, potentially using labels and uploading annotations from an external dataset.


- `create_task_chain_project_from_dataset` -- Creates a task chain project on the
  server, potentially using labels and uploading annotations from an external dataset.

For further details regarding these methods, please refer to the method documentation,
the [code snippets](#downloading-and-uploading-projects), and
[example scripts](https://github.com/openvinotoolkit/geti-sdk/tree/main/examples) provided in this repo.

Please visit the full documentation for a complete API reference.

## Using Docker

The Dockerfile can be used to run the package without having to install python on your
machine.

First build the docker image
``` sh
docker build -t geti-sdk .
```

then run it using,

``` sh
docker run --rm -ti -v $(pwd):/app geti-sdk:latest /bin/bash
```

# Supported features
## What is supported

- **Creating projects**. You can pass a variable `project_type` to control what kind of
  tasks will be created in the project pipeline. For example, if you want to create a
  single task segmentation project, you'd pass `project_type='segmentation'`. For a
  detection -> segmentation task chain, you can pass
  `project_type=detection_to_segmentation`. Please see the scripts in the `examples`
  folder for examples on how to do this.


- **Creating datasets** and retrieving dataset statistics.


- **Uploading** images, videos, annotations for images and video frames and configurations
  to a project.


- **Downloading** images, videos, annotations, models and predictions for all images and
  videos/video frames in a project. Also downloading the full project configuration
  is supported.


- **Setting configuration for a project**, like turning auto train on/off and
  setting number of iterations for all tasks.


- **Deploying a project** to load OpenVINO inference models for all tasks in the pipeline,
  and running the full pipeline inference on a local machine.


- **Creating and restoring a backup of an existing project**, using the code
  snippets provided [above](#downloading-and-uploading-projects). Only
  annotations, media and configurations are backed up, models are not.


- **Launching and monitoring training jobs** is straightforward with the `TrainingClient`.
  Please refer to the notebook `007_train_project` for instructions.


- **Authorization via Personal Access Token** is available for both On-Prem and SaaS users.


- **Fetching the active dataset**


- **Triggering (post-training) model optimization** for model quantization and
  changing models precision.


- **Running model tests**


- **Benchmarking models** to measure inference throughput on different hardware.
  It allows for quick and easy comparison of inference framerates for different
  model architectures and precision levels for the specified project.


## What is not supported

- Model upload
- Prediction upload
- Exporting datasets to COCO/YOLO/VOC format: For this, you can use the export
  functionality from the Intel® Geti™ user interface instead.
