## Introduction
Welcome to the GETi SDK! This python package contains tools to interact with a
GETi server via the REST API. It provides functionality for:

- Project creation from annotated datasets on disk
- Project downloading (images, videos, configuration, annotations, predictions and models)
- Project creation and upload from a previous download
- Deploying a project for local inference with OpenVINO
- Getting and setting project and model configuration
- Launching and monitoring training jobs
- Media upload and prediction

This repository also contains a set of (tutorial style) Jupyter
[notebooks](/notebooks/README.md) that demonstrate how to use the SDK. We highly
recommend checking them out to give you a flying start with the package.

## Getting started
### Installation
I recommend using an environment manager such as
[Anaconda](https://www.anaconda.com/products/individual) or
[venv](https://docs.python.org/3/library/venv.html) to create a new
Python environment before installing the package, and it's requirements. The SDK
requires Python version 3.8, so make sure to use that version in your environment.

Once you have created a new environment, follow these steps to install the package:

#### PyPI installation
To install the GETi SDK from PyPI, simply use `pip install geti_sdk`

If you plan on running the jupyter notebooks, install the requirements for them by
running `pip install geti_sdk[notebooks]`

#### Local installation
To install the SDK in editable mode, follow these steps:

1. Download or clone the repository and navigate to the root directory of the repo.

2. From there, install the SDK using `pip install -e .`

3. (Optional) If you plan on running the tests, the notebooks or want to build the
   documentation, you can install the package extra requirements by doing for example
   `pip install -e .[dev]`

   The valid options for the extra requirements are `[dev, docs, notebooks]`.

> **NOTE**: geti-sdk needs `python==3.8` to run. Python 3.9 will work on Linux
> systems, but unfortunately not on Windows yet since not all required packages are
> available for that version.

### Examples
The [examples](/examples/README.md) folder contains example scripts, showing various
use cases for the package. They can be run by navigating to the `examples` directory
in your terminal, and simply running the scripts like any other python script.

### Jupyter Notebooks
In addition, the [notebooks](/notebooks/README.md) folder contains jupyter notebooks
with example use cases for the `geti_sdk`. To run the notebooks,
make sure to first install the requirements for this using
`pip install -r requirements/requirements-notebooks.txt`

Once the notebook requirements are installed, navigate to the `notebooks` directory in
your terminal. Then, fire up JupyterLab by typing `jupyter lab`. This should open your
browser and take you to the JupyterLab landing page, with the SDK notebooks open.

> **NOTE**: Both the example scripts and the notebooks require access to a GETi
> instance.

## Example use cases
The package provides a main class `Geti` that can be used for the following use cases
### Downloading and uploading projects
- **Project download** The following python snippet is a minimal example of how to
  download a project using Geti:

    ```
    from geti_sdk import Geti

    geti = Geti(
      host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

    geti.download_project(project_name="dummy_project")
    ```
  Here, it is assumed that the project with name 'dummy_project' exists on the cluster.
  The Geti instance will create a folder named 'dummy_project' in your current working
  directory, and download the project parameters, images, videos, annotations,
  predictions and the active model for the project (including optimized models derived
  from it) to that folder.

  The method takes
  the following optional parameters:
    - `target_folder` -- Can be specified to change the directory to which the
      project data is saved.

    - `include_predictions` -- Set to True to download the predictions for all images
      and videos in the project. Set to False to not download any predictions.

    - `include_active_model` -- Set to True to download the active model for the
      project, and any optimized models derived from it. If set to False, no models
      are downloaded. True by default.


- **Project upload** The following python snippet is a minimal example of how to
  re-create a project on an GETi cluster using the data from a previously downloaded
  project:
    ```
    from geti_sdk import Geti

    geti = Geti(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

    geti.upload_project(target_folder="dummy_project")
    ```
  The parameter `target_folder` must be a valid path to the directory holding the
  project data. If you want to create the project using a different name than the
  original project, you can pass an additional parameter `project_name` to the upload
  method.

The Geti instance can be used to either back-up a project (by downloading it and later
uploading it again to the same cluster), or to migrate a project to a different cluster
(download it, and upload it to the target cluster).

#### Up/Downloading all projects
To up- or download all projects from a cluster, simply use the
`geti.download_all_projects` and `geti.upload_all_projects` methods instead of
the single project methods in the code snippets above.

### Deploying a project

The following code snippet shows how to create a deployment for local inference with
OpenVINO:
```
import cv2

from geti_sdk import Geti

geti = Geti(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

# Download the model data and create a `Deployment`
deployment = geti.deploy_project(project_name="dummy_project")

# Load the inference models for all tasks in the project, for CPU inference
deployment.load_inference_models(device='CPU')

# Run inference
dummy_image = cv2.imread('dummy_image.png')
prediction = deployment.infer(image=dummy_image)

# Save the deployment to disk
deployment.save(path_to_folder="dummy_project")
```
The `deployment.infer` method takes a numpy image as input.

The `deployment.save` method will save the deployment to the folder named
'dummy_project', on the local disk. The deployment can be reloaded again later using
`Deployment.from_folder('dummy_project')`.

## Supported features
What is supported:
- **Creating projects**. You can pass a variable `project_type` to control what kind of
  tasks will be created in the project pipeline. For example, if you want to create a
  single task segmentation project, you'd pass `project_type='segmentation'`. For a
  detection -> segmentation task chain, you can pass
  `project_type=detection_to_segmentation`. Please see the scripts in the `examples`
  folder for examples on how to do this.


- **Uploading** images, videos, annotations for images and video frames and configurations
  to a project


- **Downloading** images, videos, annotations, models and predictions for all images and
  videos/video frames in a project. Also downloading the full project configuration
  is supported.


- **Setting configuration for a project**, like turning auto train on/off and
  setting number of iterations for all tasks


- **Deploying a project** to load OpenVINO inference models for all tasks in the pipeline,
  and running the full pipeline inference on a local machine.


- **Creating and restoring a backup of an existing project**, using the code
  snippets provided [above](#downloading-and-uploading-projects). Only
  annotations, media and configurations are backed up, models are not.


- **Launching and monitoring training jobs**

What is not supported:
- Model upload
- Prediction upload
- Plenty of other things that are supported in GETi but not included in the SDK
  just yet. These will be added in due time.

## High level API reference
The `Geti` class provides the following methods:

- `download_project` -- Downloads a project by project name.


- `upload_project` -- Upload project from a folder.


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


- `upload_and_predict_media_folder` -- Upload all media (images and videos) from a
  folder on local disk to an existing project on the server, and download
  predictions for all uploaded media.


- `deploy_project` -- Downloads the active model for all tasks in the project as an
  OpenVINO inference model. The resulting `Deployment` can be used to run inference
  for the project on a local machine. Pipeline inference is also supported.


- `create_project_single_task_from_dataset` -- Creates a single task project on the
  server, potentially using labels and uploading annotations from an external dataset.


- `create_task_chain_project_from_dataset` -- Creates a task chain project on the
  server, potentially using labels and uploading annotations from an external dataset.

For further details regarding these methods, please refer to the method documentation
and the [code snippets](#downloading-and-uploading-projects) and
[example scripts](#examples) provided in this repo.

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
