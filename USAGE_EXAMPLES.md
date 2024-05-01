### Code examples
The package provides a main class `Geti` that can be used for the following use cases

#### Connecting to the Intel® Geti™ platform
To establish a connection between the SDK running on your local machine, and the
Intel® Geti™ platform running on a remote server, the `Geti` class needs to know the
hostname or IP address for the server and it needs to have some form of authentication.

Instantiating the `Geti` class will establish the connection and perform authentication.

- **Personal Access Token**

  The recommended authentication method is the 'Personal Access Token'. The token can be
  obtained by following the steps below:

    1. Open the Intel® Geti™ user interface in your browser
    2. Click on the `User` menu, in the top right corner of the page. The menu is
       accessible from any page inside the Intel® Geti™ interface.
    3. In the dropdown menu that follows, click on `Personal access token`, as shown in
       the image below.
    4. In the screen that follows, go through the steps to create a token.
    5. Make sure to copy the token value!

   ![Personal access token menu](docs/source/images/personal_access_token.png)

  Once you created a personal access token, it can be passed to the `Geti` class as follows:
  ```python
  from geti_sdk import Geti

  geti = Geti(
      host="https://your_server_hostname_or_ip_address",
      token="your_personal_access_token"
  )
  ```

- **User Credentials**
  > **NOTE**: For optimal security, using the token method outlined above is recommended.

  In addition to the token, your username and password can also be used to connect to
  the server. They can be passed as follows:

  ```python
  from geti_sdk import Geti

  geti = Geti(
      host="https://your_server_hostname_or_ip_address", username="dummy_user", password="dummy_password"
  )

  ```
  Here, `"dummy_user"` and `"dummy_password"` should be replaced by your username and
  password for the Geti server.


- **SSL certificate validation**

  By default, the SDK verifies the SSL certificate of your server before establishing
  a connection over HTTPS. If the certificate can't be validated, this will results in
  an error and the SDK will not be able to connect to the server.

  However, this may not be appropriate or desirable in all cases, for instance if your
  Geti server does not have a certificate because you are running it in a private
  network environment. In that case, certificate validation can be disabled by passing
  `verify_certificate=False` to the `Geti` constructor. Please only disable certificate
  validation in a secure environment!

#### Downloading and uploading projects

- **Project download** The following python snippet is a minimal example of how to
  download a project using `Geti`:

  ```python
  from geti_sdk import Geti

  geti = Geti(
    host="https://your_server_hostname_or_ip_address", token="your_personal_access_token"
  )

  geti.download_project(project_name="dummy_project")
  ```

  Here, it is assumed that the project with name 'dummy_project' exists on the cluster.
  The `Geti` instance will create a folder named 'dummy_project' in your current working
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
      are downloaded. False by default.

  > **NOTE**: During project downloading the Geti SDK stores data on local disk. If
  > necessary, please apply additional security control to protect downloaded files
  > (e.g., enforce access control, delete sensitive data securely).

- **Project upload** The following python snippet is a minimal example of how to
  re-create a project on an Intel® Geti™ server using the data from a previously
  downloaded project:

  ```python
  from geti_sdk import Geti

  geti = Geti(
    host="https://your_server_hostname_or_ip_address", token="your_personal_access_token"
  )

  geti.upload_project(target_folder="dummy_project")
  ```

  The parameter `target_folder` must be a valid path to the directory holding the
  project data. If you want to create the project using a different name than the
  original project, you can pass an additional parameter `project_name` to the upload
  method.

The `Geti` instance can be used to either back-up a project (by downloading it and later
uploading it again to the same cluster), or to migrate a project to a different cluster
(download it, and upload it to the target cluster).

#### Up- or downloading all projects
To up- or download all projects from a cluster, simply use the
`geti.download_all_projects` and `geti.upload_all_projects` methods instead of
the single project methods in the code snippets above.

#### Deploying a project

The following code snippet shows how to create a deployment for local inference with
OpenVINO:

```python
import cv2

from geti_sdk import Geti

geti = Geti(
host="https://your_server_hostname_or_ip_address", token="your_personal_access_token"
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

### Example scripts
The [examples](https://github.com/openvinotoolkit/geti-sdk/tree/main/examples)
folder contains example scripts, showing various use cases for the package. They can
be run by navigating to the `examples` directory in your terminal, and simply running
the scripts like any other python script.

### Jupyter Notebooks
In addition, the [notebooks](https://github.com/openvinotoolkit/geti-sdk/tree/main/notebooks)
folder contains Jupyter notebooks with example use cases for the `geti_sdk`. To run
the notebooks, make sure that the requirements for the notebooks are installed in your
Python environment. If you have not installed these when you were installing the SDK,
you can install them at any time using
`pip install -r requirements/requirements-notebooks.txt`

Once the notebook requirements are installed, navigate to the `notebooks` directory in
your terminal. Then, launch JupyterLab by typing `jupyter lab`. This should open your
browser and take you to the JupyterLab landing page, with the SDK notebooks open (see
the screenshot below).

> **NOTE**: Both the example scripts and the notebooks require access to a server
> running the Intel® Geti™ platform.

![Jupyter lab landing page](docs/source/images/jupyter_lab_landing_page.png)
