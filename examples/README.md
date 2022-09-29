# Examples for the Intel® Geti™ SDK

## Getting started
The example scripts provided here show several common usecases for the Intel® Geti™ SDK. To run
the examples, simply:
1. Install the `geti-sdk` package into your python environment
2. Create a `.env` file containing the authentication information for you Intel® Geti™
   server, following the instructions in the [Authentication](#authentication)
   box.
3. In your terminal, navigate to the `examples` folder
4. Activate your python environment
5. Run the example script you'd like to run using `python <name_of_script.py>`

> ### Authentication
>
> The example scripts rely on a `.env` file to load the server details for the Intel® Geti™
> instance which they run against. To provide the details for your Intel® Geti™ instance,
> create a file named `.env` directly in the `examples` directory. Two types of
> authentication are supported: Either via a Personal Access Token (the recommended
> approach) or via user credentials.
>
> #### Personal Access Token
> To use the personal access token for authenticating on your server, the `.env` file
> should have the following contents:
> ```shell
> # GETi instance details
> HOST=
> TOKEN=
> ```
> Where you should of course fill the details appropriate for your instance. For details
> on how to acquire a Personal Access Token, please refer to the section
> [Connecting to the Geti platform](../README.md#connecting-to-the-geti-platform) in the
> main readme.
>
> #### Credentials
> To use your user credentials for authenticating on your server, the `.env` file
> should have the following contents:
> ```shell
> # GETi instance details
> HOST=
> USERNAME=
> PASSWORD=
> ```
> Where you should of course fill the details appropriate for your instance.
>
> In case both a TOKEN and USERNAME/PASSWORD variables are provided, the SDK
> will default to using the TOKEN since this method is considered more secure.

## Creating a project from an existing dataset
#### COCO/Datumaro examples
This folder contains sample scripts to create projects based on the
[COCO](https://cocodataset.org/#home) [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) dataset in various configurations. Follow these steps to run any of the scripts:


1. Open the script you want to run, and modify the connection configuration for your
   Intel® Geti™ server. This means you have to specify the host (i.e. the web url or ip address
   at which the server can be reached) and the username and password (same as what
   you use to log in to the server manually)

2. Open up a terminal and navigate to the 'examples' directory. From there, execute
   the script with `python <name_of_script.py>`.

The following example scripts are available:

- `create_coco_project_single_task.py` -> Creates a single task project based using
  the images from COCO dataset for the "dog" class. Can be set to create either a
  `Detection`, `Segmentation` or `Classification` project.


- `create_coco_project_task_chain.py` -> Creates a `Detection -> Segmentation` project that
  contains bounding boxes for "dog" objects, and has the dogs segmented as a "dog shape"
  label.


- `create_demo_projects.py` -> Populates your Intel® Geti™ server with 6 different projects,
  all based on the [COCO](https://cocodataset.org/#home) [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) dataset. Each project represents a different task
  within the Intel® Geti™ platform. The projects created are:

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

> **NOTE**: To run these examples you'll need to have the COCO dataset (or a subset thereof) on
> your local disk. If you run any of the example scripts, you can either:
>
>    - Specify a path to an existing folder containing at least one subset of the
>      COCO dataset
>
>    - Leave the path unspecified, in which case the script will attempt to download
>      the val2017 subset of the dataset from the internet, to a default path in the
>      package directory. The download is approximately 1 Gb so it may take some
>      time. After downloading once, all demo scripts will automatically use the data.
>
>
> Of course, you can also download the val2017 COCO data manually via
> [this link](http://images.cocodataset.org/zips/val2017.zip).

The above examples work with [datumaro](https://github.com/openvinotoolkit/datumaro) for annotation loading, so in principle they
should work with datasets in formats other than COCO too (as long as they're supported
by datumaro).

## Uploading and getting predictions for media
The example scripts `upload_and_predict_from_numpy.py` and
`upload_and_predict_media_from_folder.py` show how to upload either a single media
item directly from memory, or upload an entire folder of media items and
get predictions for the media from the cluster.
