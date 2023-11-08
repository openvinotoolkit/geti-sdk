import collections
import time
import os
import json
import sys

import cv2
import numpy as np
from argparse import ArgumentParser
from utils import VideoPlayer

from geti_sdk.deployment import Deployment
from geti_sdk.utils import show_image_with_annotation_scene
from geti_sdk.demos import get_person_car_bike_video

PROJECT_NAME = "person-bike-car"
PATH_TO_DEPLOYMENT_FOLDER = os.path.join("deployments", PROJECT_NAME)

# Get the path to the offline deployment
offline_deployment = Deployment.from_folder(PATH_TO_DEPLOYMENT_FOLDER)

# Get the path to the example video. This will download the video if it is not found on your disk yet
VIDEO_PATH = get_person_car_bike_video(video_path="person-bicycle-car-detection.mp4")
video_file = VIDEO_PATH


def build_argparser():
    """Input Arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target infer device to; CPU, GPU"
        "(CPU by default).",
        default="CPU",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Location to store the results of the processing",
        default="results",
        required=True,
        type=str,
    )
    return parser

# Main processing function to run object detection.
def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0, output_path="results"):
    player = None
    fps = 0
    processing_times = collections.deque()
    processing_fps = collections.deque()
    frames_counter = 0
    try:
        # ================1. Create a video player to play with target fps================
        player = VideoPlayer(
            source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )
            
        while True:
            # Grab the frame.
            frame = player.next()
            frames_counter = frames_counter + 1
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Measure processing time.

            start_time = time.time()

            # ================2. Using offline predictions========================

            # Get the results.
            prediction = offline_deployment.infer(frame)
            stop_time = time.time()
            processing_times.append(stop_time - start_time)

            # ================3. Creating output with bounding boxes, labels, and confidence========
            output = show_image_with_annotation_scene(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), prediction, show_results=False
            )

            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            # Save the FPS for your reference
            processing_fps.append(fps)

            # Use processing fps from last 200 frames.
            if len(processing_fps) > 200:
                processing_fps.popleft()

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname=title, mat=output)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=output, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )

    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()
        if len(processing_times) > 0 and len(processing_fps) > 0:
            avg_processing_time = int(np.mean(processing_times) * 1000)
            avg_processing_fps = int(np.mean(processing_fps))
            
            # Data to be written
            results = {
                "time": f"{avg_processing_time}",
                "fps": f"{avg_processing_fps}",
                "frames": f"{frames_counter}"
            }
            
            # Serialized output
            json_object = json.dumps(results, indent=4)
            
            # Calculate output path
            stats_path = os.path.join(output_path, "stats.json")
            
            # Writing to sample.json
            with open(stats_path, "w") as outfile:
                outfile.write(json_object)

def main():
    args = build_argparser().parse_args()
    offline_deployment.load_inference_models(device=args.device)

    # Calculate output path
    job_id = str(os.environ["PBS_JOBID"]).split(".")[0]
    output_path = os.path.join(args.output_path, job_id)
    
    # Create output path
    os.makedirs(output_path, exist_ok=True)
    
    # Run process
    run_object_detection(source=video_file, flip=False, use_popup=False, output_path=output_path)

if __name__ == "__main__":
    sys.exit(main() or 1)