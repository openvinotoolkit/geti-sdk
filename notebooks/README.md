# Setting things up
To get started with the example notebooks provided in this folder, please make sure
that you have installed the `geti-sdk` package. The steps to install the package
can be found in the [main readme](../README.md) in this repository. Once the package
is installed, you can follow the steps below to set up the notebooks.

1. Navigate to this folder in a terminal

2. Activate the Python environment in which you installed the `geti-sdk` package

3. Run `pip install -r ../requirements/requirements-notebooks.txt` to install the packages required to
   run the notebooks

4. Create a `.env` file containing the server details for your Intel® Geti™ server,
   following the instructions in the [Authentication](#authentication)
   box.

5. In your terminal, navigate to the `notebooks` directory and execute the command
   `jupyter lab`. This will start the jupyter server and should take you straight to
   the jupyter web interface.

6. The notebooks should show up in the side menu of the jupyter web interface.

> ## Authentication
>
> The notebooks rely on a `.env` file to load the server details for the Intel® Geti™
> instance which they run against. To provide the details for your Intel® Geti™ instance,
> create a file named `.env` directly in the `notebooks` directory. Two types of
> authentication are supported: Either via a Personal Access Token (the recommended
> approach) or via user credentials.
>
> ### Personal Access Token
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
> ### Credentials
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
>
> ### SSL Certificate verification
> By default, the SDK verifies the SSL certificate of your server before establishing
> a connection over HTTPS. If the certificate can't be validated, this will results in
> an error and the SDK will not be able to connect to the server.
>
> However, this may not be appropriate or desirable in all cases, for instance if your
> Geti server does not have a certificate because you are running it in a private
> network environment. In that case, certificate validation can be disabled by adding
> the following variable to the `.env` file:
> ```shell
> VERIFY_CERT=0
> ```

# Available notebooks
The following notebooks are currently provided:

- [001 create_project](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/001_create_project.ipynb)
  -- This notebook shows how to create a project, and explains the parameters that
  can be used to control the project properties.


- [002 create_project_from_dataset](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/002_create_project_from_dataset.ipynb)
  -- This notebook shows how to create a project from an existing dataset, and
  upload images and annotations to it. The data used in this notebook is from the
  [COCO](https://cocodataset.org/#home) [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) dataset.


- [003 upload_and_predict_image](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/003_upload_and_predict_image.ipynb)
  -- This notebook shows how to upload an image to an existing project, and get
  a prediction for it.


- [004 create_pipeline_project_from_dataset](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/004_create_pipeline_project_from_dataset.ipynb)
  -- This notebook shows how to create a pipeline project (with two trainable tasks in
  it) from an existing dataset, and how to upload images and annotations to it.


- [005 modify_image](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/005_modify_image.ipynb)
  -- This notebook shows how to get an image from a project, convert it to grayscale, and
  then re-apply the annotation for the original image to it.


- [006 reconfigure_task](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/006_reconfigure_task.ipynb)
  -- This notebook shows how to view and change the configuration for a task.


- [007 train_project](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/007_train_project.ipynb)
  -- This notebook shows how to start a training job for a task in a project, monitor
  the job's progress and get the model that was trained in the job once the training
  completes.


- [008 deploy_project](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/008_deploy_project.ipynb)
  -- This notebook shows how to create a deployment for a project in order to run
  inference locally with OpenVINO.


- [009 download_and_upload_project](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/009_download_and_upload_project.ipynb)
  -- This notebook shows how to download a project to local disk, including all media,
  annotations as well as the project configuration. The notebook also demonstrates how
  to re-create the project from a previously downloaded project, and upload all
  downloaded data to the newly created project.

- [101 simulate_low-light_product_inspection_demo](https://github.com/openvinotoolkit/geti_sdk/blob/main/notebooks/use_cases/101_simulate_low_light_product_inspection.ipynb)
  -- This notebook shows how to systematically simulate a change
  in lighting conditions (i.e. a shift in data distribution),
  and the effect such a change has on model predictions. This notebook uses the 'transistor'
  category from the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
  [(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) dataset.
