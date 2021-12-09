# SC REST API tools 
## Introduction
This package contains some helpful tools to interact with a Sonoma Creek cluster via 
the SC REST API. It provides functionality for uploading media, annotations and set 
some basic project configurable parameters. 

Three example scripts to create projects based on the COCO dataset in various 
configurations are provided:
- create_coco_project_detection.py -> Converts the coco annotations for the "dog" class 
  to bounding boxes, and creates a detection project with coco images containing dogs
    
- create_coco_project_segmentation.py -> Uses the coco polygon annotations for the "dog" 
  class, and creates a segmentation project with coco images containing dogs
  
- create_coco_project_pipeline.py -> Creates a Detection -> Segmentation project that 
  contains bounding boxes for "dog" objects, and has the dogs segmented as a "dog shape"
  label.
  
## Installation
Navigate to the package directory, and install the requirements using 
`pip install -r requirements.txt`. Then run `pip install .` to install the package. 
You can also install it in editable mode if you want to make changes to the package, 
using `pip install -e .`

## Using the package
The example scripts provide an impression of how the package can be used. Make sure 
to change the values under the section "Script configuration" to suit your SC server.