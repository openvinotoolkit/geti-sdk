# SC REST API tools 
## Introduction
This package contains tools to interact with a Sonoma Creek cluster via 
the SC REST API. It provides functionality for:
- Project creation from datasets on disk
- Project downloading (images, videos, configuration, annotations, predictions and models)
- Project creation and upload from a previous download
- Deploying a project for local inference with OpenVINO

## Installation
I recommend using an environment manager such as 
[Anaconda](https://www.anaconda.com/products/individual) or 
[venv](https://docs.python.org/3/library/venv.html) to create a new 
Python environment before installing the package and it's requirements. The package 
requires Python version 3.8, so make sure to use that version in your environment. 

Once you have created a new environment, follow these steps to install the package:

1. Download or clone the repository and navigate to the package directory. 

2. From there, install the requirements using 
`pip install -r requirements.txt`. 
   
3. Then run `pip install .` to install the package. 
You can also install it in editable mode using `pip install -e .` This is handy if
you want to make changes to the package, or want to keep it up to date with the 
latest code changes in the repository. 

> **NOTE**: sc-api-tools needs `python==3.8` to run. Unfortunately python 3.9 won't 
> work yet since not all required packages are available for that version.

## Using the package
The package provides a main class `SCRESTClient` that can be used for creating, downloading and
uploading projects. 
### Downloading and uploading projects
- **Project download** The following python snippet is a minimal example of how to 
  download a project using the SCRESTClient:

    ```
    from sc_api_tools import SCRESTClient
    
    client = SCRESTClient(
      host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )
    
    client.download_project(project_name="dummy_project")
    ```
  Here, it is assumed that the project with name 'dummy_project' exists on the cluster. 
  The client will create a folder named 'dummy_project' in your current working 
  directory, and download the project parameters, images, videos, annotations, 
  predictions and the active model for the project (including optimized models derived 
  from it) to that folder.
  
  The method takes 
  the following optional parameters:
    - `target_folder` -- Can be specified to change the directory to which the 
      project data is saved.
      
    - `include_predictions` -- Set to True to download the predictions for all images 
      and videos in the project. Set to False to not download any predictions.
      
    - 'include_active_model' -- Set to True to download the active model for the 
      project, and any optimized models derived from it. If set to False, no models 
      are downloaded. True by default. 


- **Project upload** The following python snippet is a minimal example of how to 
  re-create a project on an SC cluster using the data from a previously downloaded 
  project:
    ```
    from sc_api_tools import SCRESTClient
    
    client = SCRESTClient(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )
    
    client.upload_project(target_folder="dummy_project")
    ```
  The parameter `target_folder` must be a valid path to the directory holding the 
  project data. If you want to create the project using a different name than the 
  original project, you can pass an additional parameter `project_name` to the upload 
  method.

The client can be used to either back-up a project (by downloading it and later 
uploading it again to the same cluster), or to migrate a project to a different cluster 
(download it, and upload it to the target cluster).

#### Up/Downloading all projects
To up- or download all projects from a cluster, simply use the 
`client.download_all_projects` and `client.upload_all_projects` methods instead of 
the single project methods in the code snippets above.

### Deploying a project
The following code snippet shows how to create a deployment for local inference with 
OpenVINO:
```
import cv2

from sc_api_tools import SCRESTClient

client = SCRESTClient(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

# Download the model data and create a `Deployment`    
deployment = client.deploy_project(project_name="dummy_project")

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

### Examples
The [examples](examples/README.md) folder contains example scripts, showing various 
usecases for the packages .

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
  
What is not (fully) supported:
- Model upload
- Label hierarchies *should* work, I have tested this but please use caution 
  and test extensively yourself
- Other stuff that I may have missed... Please please please test carefully before 
  relying on this tool to back up your projects!!!
  
## API reference
The high level `SCRESTClient` class provides the following methods:

- `download_project` -- Downloads a project by project name.
  

- `upload_project` -- Upload project from a folder.
  

- `download_all_projects` -- Downloads all projects found on the SC cluster.
  

- `upload_all_projects` -- Uploads all projects found in a specified folder to the SC 
  cluster.
  

- `upload_and_predict_image` -- Uploads a single image to an existing project on the 
  SC cluster, and requests a prediction for that image. Optionally, the prediction can 
  be visualized as an overlay on the image.


- `upload_and_predict_video` -- Uploads a single video to an existing project on the 
  SC cluster, and requests predictions for the frames in the video. As with 
  upload_and_predict_image, the predictions can be visualized on the frames. The 
  parameter `frame_stride` can be used to control which frames are extracted for 
  prediction.


- `upload_and_predict_media_folder` -- Upload all media (images and videos) from a 
  folder on local disk to an existing project on the SC cluster, and download 
  predictions for all uploaded media.
  

- `deploy_project` -- Downloads the active model for all tasks in the project as an 
  OpenVINO inference model. The resulting `Deployment` can be used to run inference 
  for the project on a local machine. Pipeline inference is also supported.


- `create_project_single_task_from_dataset` -- Creates a single task project on the SC 
  cluster, potentially using labels and uploading annotations from an external dataset.
  

- `create_task_chain_project_from_dataset` -- Creates a task chain project on the SC 
  cluster, potentially using labels and uploading annotations from an external dataset.
  
For further details regarding these methods, please refer to the method documentation 
and the [code snippets](#downloading-and-uploading-projects) and 
[example scripts](#project-creation-examples) provided in this repo.