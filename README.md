# SC REST API tools 
## Introduction
This package contains tools to interact with a Sonoma Creek cluster via 
the SC REST API. It provides functionality for:
- Project creation from datasets on disk
- Project downloading (images and annotations)
- Project creation and upload from a previous download

## Installation
Download or clone the repository and navigate to the package directory. From there, 
install the requirements using 
`pip install -r requirements.txt`. Then run `pip install .` to install the package. 
You can also install it in editable mode using `pip install -e .`. This is handy if
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
  directory, and download the project parameters, images and annotations to that folder. 
  Models are not downloaded, and also videos are not supported (yet). The method takes 
  an optional parameter `target_folder` that can be specified to change the 
  directory to which the project data is saved.


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

### Project creation examples
#### COCO/Datumaro examples
The `examples` folder contains sample scripts to create projects based on the 
COCO dataset in various configurations:

- `create_coco_project_single_task.py` -> Creates a single task project based using 
  the images from COCO dataset for the "dog" class. Can be set to create either a
  `Detection`, `Segmentation` or `Classification` project.
  

- `create_coco_project_task_chain.py` -> Creates a `Detection -> Segmentation` project that 
  contains bounding boxes for "dog" objects, and has the dogs segmented as a "dog shape"
  label.
  
> **NOTE**: To run these examples you'll need to have the COCO dataset (or a subset thereof) on
> your local disk. I recommend using the 2017 validation dataset, which contains 5000 
> images. It can be downloaded via
> [this link](http://images.cocodataset.org/zips/val2017.zip) (approx. 1 Gb download).

The above examples work with Datumaro for annotation loading, so in principle they 
should work with datasets in formats other than COCO too (as long as they're supported 
by Datumaro).

#### Vitens examples
The `create_vitens_aeromonas_project_single_task.py` script creates a detection project
using the bacteria colony dataset for the company Vitens (a former Cosmonio customer). 
This is one of the UserStory datasets. The dataset can be downloaded from 
[this link](https://intel.sharepoint.com/:u:/r/sites/user-story-dataset-sharing/Shared%20Documents/User%20Stories%20Datasets/Detection/Vitens%20Bacteria%20Counting/Vitens%20Aeromonas.zip?csf=1&web=1&e=wFXEle),
but the data is confidential, so please treat it as such.

## Supported features
What is supported:
- Creating projects. You can pass a variable `project_type` to control what kind of 
  tasks will be created in the project pipeline. For example, if you want to create a 
  single task segmentation project, you'd pass `project_type='segmentation'`. For a 
  detection -> segmentation task chain, you can pass 
  `project_type=detection_to_segmentation`.
- Uploading images, and annotations for images to a project
- Downloading images and annotations for images from a project
- Setting basic configuration for a project, like turning auto train on/off and 
  setting number of iterations for all tasks
- **Creating and restoring a backup of an existing project**, by first downloading its 
  data using the `download_project.py` script and later uploading it again using the 
  `upload_project.py` script. Only annotations and images are backed up, 
  models are not. 
>  **NOTE**: I have only tested backing up with `detection`, `segmentation`, 
>  `classification` and `detection_to_segmentation` projects. I see no reason why it 
>  wouldn't work for `anomaly_classification` and `detection_to_classification` 
>  projects, but please test carefully before relying on this.
  
What is not (fully) supported:
- Video and video frame annotations are not (yet) supported.
- Empty labels in a task chain project seem not always to be uploaded correctly. They 
  are handled correctly for the first task, but not if there is an object in the 
  first task but the label for the second task is empty
- Label hierarchies *should* work, I have tested this briefly but please use caution 
  and test extensively yourself
- Creating `detection to classification` projects *should* work, but is not tested
- Other stuff that I may have missed... Please please please test carefully before 
  relying on this tool to back up your projects!!!
