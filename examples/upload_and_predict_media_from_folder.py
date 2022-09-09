import os

from dotenv import dotenv_values
from utils import ensure_example_project

from sc_api_tools import SCRESTClient

if __name__ == "__main__":
    # Get credentials from .env file
    env_variables = dotenv_values(dotenv_path=".env")

    if not env_variables:
        raise ValueError(
            "Unable to load login details from .env file, please make sure the file "
            "exists at the root of the `examples` directory."
        )

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up REST client with server address and login details
    client = SCRESTClient(
        host=env_variables.get("HOST"),
        username=env_variables.get("USERNAME"),
        password=env_variables.get("PASSWORD"),
    )

    # `FOLDER_WITH_MEDIA` is the path to the directory with images and videos that
    # should be uploaded to the SC cluster
    FOLDER_WITH_MEDIA = os.path.join("..", "notebooks", "data")

    # `PROJECT_NAME` is the name of the project to which the media should be uploaded,
    # and from which predictions can be requested. A project with this name should
    # exist on the cluster. If the project exists but doesn't have any trained models,
    # the media will be uploaded but no predictions will be generated.
    PROJECT_NAME = "COCO dog detection"

    # `DELETE_AFTER_PREDICTION` can be set to True to delete the media from the
    # project once all predictions are downloaded. This can be useful to save disk
    # space on the cluster, or to avoid cluttering a project with a lot of
    # unannotated media
    DELETE_AFTER_PREDICTION = False

    # `OUTPUT_FOLDER` is the target folder to which the predictions will be saved
    OUTPUT_FOLDER = "media_folder_predictions"

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    # Make sure that the specified project exists on the server
    ensure_example_project(client=client, project_name=PROJECT_NAME)

    # Upload the media in the folder and generate predictions
    client.upload_and_predict_media_folder(
        project_name=PROJECT_NAME,
        media_folder=FOLDER_WITH_MEDIA,
        delete_after_prediction=DELETE_AFTER_PREDICTION,
        output_folder=OUTPUT_FOLDER,
    )
