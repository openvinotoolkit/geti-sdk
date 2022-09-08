import os

import cv2
import numpy as np
from dotenv import dotenv_values

from examples.utils import ensure_example_project
from sc_api_tools import SCRESTClient


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an input array representing an image by a certain `angle` (in degrees)

    :param image: Numpy array holding the pixel data for the image to rotate
    :param angle: Angle to rotate the image by, in degrees
    :return: Rotated image
    """
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    return cv2.warpAffine(
        src=image, M=rotation_matrix, dsize=image.shape[1::-1], flags=cv2.INTER_LINEAR
    )


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

    # `PROJECT_NAME` is the name of the project to which the media should be uploaded,
    # and from which predictions can be requested. A project with this name should
    # exist on the cluster. If the project exists but doesn't have any trained models,
    # the media will be uploaded but no predictions will be generated.
    PROJECT_NAME = "COCO dog detection"

    # `PATH_TO_IMAGE` is the path to the image that should be uploaded
    PATH_TO_IMAGE = os.path.join("..", "notebooks", "data", "dogs.png")

    # `DELETE_AFTER_PREDICTION` can be set to True to delete the media from the
    # project once all predictions are downloaded. This can be useful to save disk
    # space on the cluster, or to avoid cluttering a project with a lot of
    # unannotated media
    DELETE_AFTER_PREDICTION = False

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    # First, we load the image as a numpy array using opencv
    numpy_image = cv2.imread(PATH_TO_IMAGE)

    # Rotate the image by 20 degrees
    rotated_image = rotate_image(image=numpy_image, angle=20)

    # Make sure that the project exists
    ensure_example_project(client=client, project_name=PROJECT_NAME)

    print(
        "Uploading and predicting example image now, an image window containing the "
        "image and prediction will pop up."
    )

    # We can upload and predict the resulting array directly:
    sc_image, image_prediction = client.upload_and_predict_image(
        project_name=PROJECT_NAME,
        image=rotated_image,
        visualise_output=True,
        delete_after_prediction=DELETE_AFTER_PREDICTION,
    )

    # We can do the same with videos. For example, to investigate the effect image
    # rotations have on our model predictions, we can create a video with rotated
    # versions of the image as frames. That will allow us to quickly see how well
    # our model holds up under image rotations

    # Create list of rotated images, these will be the frames of the video
    rotation_video = []
    for angle in [0, 90, 180, 270, 360]:
        rotation_video.append(rotate_image(image=numpy_image, angle=angle))

    # Create video, upload and predict from the list of frames
    sc_video, video_frames, frame_predictions = client.upload_and_predict_video(
        project_name=PROJECT_NAME,
        video=rotation_video,
        frame_stride=1,
        visualise_output=True,
        delete_after_prediction=DELETE_AFTER_PREDICTION,
    )
