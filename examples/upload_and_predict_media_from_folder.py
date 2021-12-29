import os

from sc_api_tools import SCRESTClient

if __name__ == '__main__':
    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up REST client with server address and login details
    client = SCRESTClient(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

    # `FOLDER_WITH_MEDIA` is the path to the directory with images and videos that
    # should be uploaded to the SC cluster
    FOLDER_WITH_MEDIA = os.path.join('..', 'dummy_folder')

    # `PROJECT_NAME` is the name of the project to which the media should be uploaded,
    # and from which predictions can be requested. A project with this name should
    # exist on the cluster. If the project exists but doesn't have any trained models,
    # the media will be uploaded but no predictions will be generated.
    PROJECT_NAME = 'dummy_project'

    # `DELETE_AFTER_PREDICTION` can be set to True to delete the media from the
    # project once all predictions are downloaded. This can be useful to save disk
    # space on the cluster, or to avoid cluttering a project with a lot of
    # unannotated media
    DELETE_AFTER_PREDICTION = False

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------
    client.upload_and_predict_media_folder(
        project_name=PROJECT_NAME,
        media_folder=FOLDER_WITH_MEDIA,
        delete_after_prediction=DELETE_AFTER_PREDICTION
    )
