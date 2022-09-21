# Examples for the GETi SDK

## Getting started
The example scripts provided here show several common usecases for the GETi SDK. To run
the examples, simply:
1. Install the geti-sdk package into your python environment
2. Create a `.env` file containing the login details for you GETi instance,
   following the instructions in the [Credentials management](#credentials-management)
   box.
3. In your terminal, navigate to the `examples` folder
4. Activate your python environment
5. Run the example script you'd like to run using `python <name_of_script.py>`

> ### Credentials management
>
> The example scripts rely on a `.env` file to load the login details for the GETi
> instance which they run against. To provide the credentials for your GETi instance,
> create a file named `.env` directly in the `examples` directory. The file should have
> the following contents:
> ```shell
> # GETi instance details
> HOST=
> USERNAME=
> PASSWORD=
> ```
> Where you should of course fill the details appropriate for your instance.

## Creating a project from an existing dataset
#### COCO/Datumaro examples
This folder contains sample scripts to create projects based on the
COCO dataset in various configurations. Follow these steps to run any of the scripts:


1. Open the script you want to run, and modify the connection configuration for your
   GETi cluster. This means you have to specify the host (i.e. the web url or ip address
   at which the cluster can be reached) and the username and password (same as what
   you use to log in to GETi manually)

2. Open up a terminal and navigate to the 'examples' directory. From there, execute
   the script with `python <name_of_script.py>`.

The following example scripts are available:

- `create_coco_project_single_task.py` -> Creates a single task project based using
  the images from COCO dataset for the "dog" class. Can be set to create either a
  `Detection`, `Segmentation` or `Classification` project.


- `create_coco_project_task_chain.py` -> Creates a `Detection -> Segmentation` project that
  contains bounding boxes for "dog" objects, and has the dogs segmented as a "dog shape"
  label.


- `create_demo_projects.py` -> Populates your GETi cluster with 6 different projects,
  all based on the COCO dataset. Each project represents one of the supported task
  within GETi MVP. The projects created are:

  - **Segmentation demo** -- Segmentation of 'backpack' and 'suitcase' and objects
  - **Detection demo** -- Detection of 'person' and 'cell phone' objects
  - **Classification demo** -- Single class classification of 'horse' vs 'zebra'
    vs 'cat' vs 'bear'
  - **Anomaly classification demo** -- Anomaly classification of images of animals
    ('Normal') vs traffic lights and stop signs ('Anomalous')
  - **Animal detection to segmentation demo** -- Detection of 'animal', followed by
    segmentation into subcategories: 'horse', 'dog', 'cat', 'cow', 'sheep',
  - **Animal detection to classification demo** -- Detection of 'animal', followed by
    classification into two categories: 'wild' and 'domestic'

> **NOTE**: To run these examples you'll need to have the MS COCO dataset (or a subset thereof) on
> your local disk. If you run any of the example scripts, you can either:
>
>    - Specify a path to an existing folder containing at least one subset of the
>      COCO dataset
>
>    - Leave the path unspecified, in which case the script will attempt to download
>      the val2017 subset of the dataset from the internet, to a default path in the
>      package directory. This is an approximately 1 Gb download so it may take some
>      time. After downloading once, all demo scripts will automatically use the data.
>
>
> Of course, you can also download the val2017 COCO data manually via
> [this link](http://images.cocodataset.org/zips/val2017.zip).

The above examples work with Datumaro for annotation loading, so in principle they
should work with datasets in formats other than COCO too (as long as they're supported
by Datumaro).

## Uploading and getting predictions for media
The example scripts `upload_and_predict_from_numpy.py` and
`upload_and_predict_media_from_folder.py` show how to upload either a single media
item directly from memory, or upload an entire folder of media items and
get predictions for the media from the cluster.
