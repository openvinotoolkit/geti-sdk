# Notebooks for sc-api-tools
## Getting started
To get started with the example notebooks provided in this folder, please make sure 
that you have installed the `sc-api-tools` package. The steps to install the package 
can be found in the [main readme](../README.md) in this repository. Once the package 
is installed, you can follow the steps below to set up the notebooks.
1. Navigate to this folder in a terminal 
   
2. Activate the python environment in which you installed the sc-api-tools package 
   
3. Run `pip install -r requirements-notebooks.txt` to install the packages required to 
   run the notebooks

4. Create a `.env` file containing the login details for your Sonoma Creek instance, 
   following the instructions in the [Credentials management](#credentials-management) 
   box.

5. In your terminal, navigate to the `notebooks` directory and execute the command 
   `jupyter lab`. This will fire up the jupyter server and should take you straight to 
   the jupyter web interface.
   
6. The notebooks should show up in the side menu of the jupyter web interface. 
   
> ### Credentials management
> The notebooks rely on a `.env` file to load the login details for the SonomaCreek 
> instance which they run against. To provide the credentials for your SC instance, 
> create a file named `.env` directly in the `notebooks` directory. The file should have 
> the following contents:
> ```shell
> # SonomaCreek instance details
> HOST=
> USERNAME=
> PASSWORD=
> ```
> Where you should of course fill the details appropriate for your instance. 

## Notebooks
The following notebooks are currently provided:
- [001 create_project](001_create_project.ipynb) -- This notebook shows how to create 
  a project, and explains the parameters that can be used to control the project 
  properties.
  

- [002 create_project_from_dataset](002_create_project_from_dataset.ipynb) -- This 
  notebook shows how to create a project from an existing dataset, and upload images 
  and annotations to it.
  

- [003 upload_and_predict_image](003_upload_and_predict_image.ipynb) -- This notebook 
  shows how to upload an image to an existing project, and get a prediction for it.
  

- [004 create_pipeline_project_from_dataset](004_create_pipeline_project_from_dataset.ipynb) 
  -- This notebook shows how to create a pipeline project (with two trainable tasks in 
  it) from an existing dataset, and how to upload images and annotations to it.
  

- [005 modify_image](005_modify_image.ipynb) 
  -- This notebook shows how to get an image from a project, convert it to grayscale, and 
  then re-apply the annotation for the original image to it.


- [006 reconfigure_task](006_reconfigure_task.ipynb) 
  -- This notebook shows how to view and change the configuration for a task.


- [007 train_project](007_train_project.ipynb) -- This notebook shows how to start a 
  training job for a task in a project, monitor the job's progress and get the model 
  that was trained in the job once the training completes. 


- [008 deploy_project](008_deploy_project.ipynb) -- This notebook shows how to create 
  a deployment for a project in order to run inference locally with OpenVINO.
  

- [009 download_and_upload_project](009_download_and_upload_project.ipynb) -- This 
notebook shows how to download a project to local disk, including all media, 
  annotations as well as the project configuration. The notebook also demonstrates how 
  to re-create the project from a previously downloaded project, and upload all 
  downloaded data to the newly created project.