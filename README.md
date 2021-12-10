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
A utility script to download a project is provided in the `scripts` folder. Only 
images and annotations are downloaded. Video's and annotations for video frames are 
not supported yet.

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
  
What is not supported:
- Empty labels (they are created, but not processed correctly when up/downloading)
- Label hierarchies
- Creating `classification`, `anomaly classification` and `detection to classification` projects
- Plenty of other complicated stuff... Please test carefully before fully relying on this tool to back up your projects!!!