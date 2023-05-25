# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import os

import cv2
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk import Geti
from geti_sdk.utils import show_image_with_annotation_scene

# --------------------------------------------------
# Configuration section
# --------------------------------------------------

# Set the configuration for the Geti server here
# Make sure to replace the following with the appropriate information for your
# Intel Geti server
HOST = "https://127.0.0.1"
USERNAME = "your username"
PASSWORD = "your password"

# `PROJECT_NAME` is the name of the project that will be used to generate
# predictions. A project with this name should exist on the server
PROJECT_NAME = "your project name"

# `PATH_TO_VIDEO` is the path to the video that should be predicted
PATH_TO_VIDEO = "video.mp4"

# --------------------------------------------------
# End of configuration section
# --------------------------------------------------


if __name__ == "__main__":
    # Set up the Geti instance with the server configuration details
    geti = Geti(
        host=HOST, username=USERNAME, password=PASSWORD, verify_certificate=False
    )

    logging.info(
        f"Running video prediction and reconstruction script for "
        f"project `{PROJECT_NAME}` and input video `{PATH_TO_VIDEO}`."
    )

    # Create deployment for the project, and prepare it for running inference
    logging.info("Fetching models from Intel Geti project...")
    deployment = geti.deploy_project(PROJECT_NAME)
    deployment.load_inference_models(device="CPU")

    # Check if video file exists
    if not os.path.isfile(PATH_TO_VIDEO):
        raise ValueError(
            f"Video file at path {PATH_TO_VIDEO} does not exist, please make sure to "
            f"provide a path to a valid video file (including extension)."
        )

    # Open the video capture, this prepares the video to be ready for reading
    cap = cv2.VideoCapture(PATH_TO_VIDEO)

    # Determine the output video path
    video_path, video_extension = os.path.splitext(PATH_TO_VIDEO)
    OUTPUT_VIDEO_PATH = os.path.abspath(video_path + "_reconstructed" + video_extension)

    # Extract original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = num_frames / fps

    logging.info(
        f"Input video contains {num_frames:.1f} frames, for a total duration of "
        f"{video_duration:.1f} seconds"
    )

    # Create a video writer to be able to save the reconstructed video
    video_out = cv2.VideoWriter(
        filename=OUTPUT_VIDEO_PATH,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=(frame_width, frame_height),
    )

    count = 0
    logging.info(
        f"Starting video predictions. Reconstructed video will be saved to "
        f"`{OUTPUT_VIDEO_PATH}`"
    )
    logging.info(
        "Running video prediction and reconstruction... Press `q` on the keyboard to "
        "stop the process"
    )

    with logging_redirect_tqdm(tqdm_class=tqdm), tqdm(
        total=num_frames, desc="Predicting"
    ) as progress_bar:
        while cap.isOpened():

            ret, frame = cap.read()
            if ret is True:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction = deployment.infer(rgb_frame)
                output_frame = show_image_with_annotation_scene(
                    image=rgb_frame, annotation_scene=prediction, show_results=False
                )
                video_out.write(output_frame)

                count = count + 1

                progress_bar.update(1)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            else:
                break

    cap.release()
    video_out.release()

    if count == num_frames:
        logging.info(
            f"Video reconstruction completed successfully. Output video saved to "
            f"`{OUTPUT_VIDEO_PATH}`"
        )
    else:
        logging.warning(
            f"Video reconstruction completed, but not all frames were processed. "
            f"Only {count} out of {num_frames} were processed successfully"
        )
