# SC REST API tools 
## Introduction
This package contains some helpful tools to interact with a Sonoma Creek cluster via 
the SC REST API. It provides functionality for uploading media, annotations and set 
some basic project configurable parameters. 

## Installation
Navigate to the package directory, and install the requirements using 
`pip install -r requirements.txt`. Then run `pip install .` to install the package. 
You can also install it in editable mode if you want to make changes to the package, 
using `pip install -e .`

## Using the package
The example scripts provide an impression of how the package can be used. Make sure 
to change the values under the section "Script configuration" to suit your SC server.

## Scripts
The `scripts` folder provides two utility scripts:
- `download_project.py` to download a project to a folder on your disk. Only images 
  and annotations are downloaded. Video's and annotations for video frames are not 
  supported yet. Also models are not downloaded.
- `create_project_from_backup.py` to re-create a project on an SC cluster using the 
  data from a previously downloaded project. This can be used to back-up projects, but
  beware that not all features are supported (see Supported features section below).
  

## Examples
The `examples` folder contains three example scripts to create projects based on the 
COCO dataset in various configurations:
- create_coco_project_detection.py -> Converts the coco annotations for the "dog" class 
  to bounding boxes, and creates a detection project with coco images containing dogs
    
- create_coco_project_segmentation.py -> Uses the coco polygon annotations for the "dog" 
  class, and creates a segmentation project with coco images containing dogs
  
- create_coco_project_pipeline.py -> Creates a Detection -> Segmentation project that 
  contains bounding boxes for "dog" objects, and has the dogs segmented as a "dog shape"
  label.
  
## Supported features
What is supported:
- Creating `detection`, `segmentation` and `detection to segmentation` projects
- Uploading images, and annotations for images to a project
- Downloading images and annotations for images from a project
- Setting basic configuration for a project, like turning auto train on/off and 
  setting number of iterations for all tasks-
- Creating and restoring a backup of an existing project, by first downloading its 
  data using the `download_project.py` script and later uploading it again using the 
  `create_project_from_backup.py` script. This only works for the aforementioned 
  project types, and empty labels are currently not included.
  
What is not supported:
- Empty labels (they are created, but not processed correctly when up/downloading)
- Label hierarchies
- Creating `classification`, `anomaly classification` and `detection to classification`
  projects
- Plenty of other complicated stuff... Please please please test carefully before 
  relying on this tool to back up your projects!!!