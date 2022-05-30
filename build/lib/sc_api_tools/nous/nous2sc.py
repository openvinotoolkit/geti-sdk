"""
NOUS2SC Migration Script

This script is a reference for demonstrating how to migrate a NOUS project export
to an instance of Sonoma Creek.

A customised NOUSAnnotationReader and NOUSAnnotationManager have been added

NOTE:

Steps:
1 - Login to NOUS using the UI app
2 - Export a given project
  checking options:
    images;
    videos;
    annotations;
3 - Export the .zip
4 - Update this script with the path to the .zip and populate the relevant information
  Single Task: 
    New Project Name on SC;
    Task Type;

  Task Chain:
    New Project Name on SC;
    List of tasks in the chain [];
    List of labels list for each task in the chain [[],[]]

5 - Run
6 - Login to SC instance and verify :
  A - images and videos (quantity) has been uploaded as expected 
  B - annotation (quantitiy) has been uploaded as expected
  C - visually inspect a selection of annotations to ensure they look correct

"""

from zipfile import ZipFile
import uuid
import shutil
import os
from typing import Union, Optional, List, Dict, Any
import sys
from argparse import ArgumentParser, SUPPRESS

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import NOUSAnnotationReader
from sc_api_tools.rest_managers.annotation_manager import NOUSAnnotationManager
from sc_api_tools.utils import get_task_types_by_project_type

from sc_api_tools.rest_managers import (
    ProjectManager,
    ConfigurationManager,
    ImageManager,
    VideoManager,
)
from sc_api_tools.data_models import (
    TaskType,
)


def unzip_to_temp(export_path):
    """
    Unzip the nous project to a temp folder

    :param export_path: Path to the zipped NOUS project
    :return: path to the temporary folder containing the unzipped contents
    """
    zip = ZipFile(export_path)
    temp_dir = uuid.uuid4().hex
    zip.extractall(temp_dir)
    return temp_dir


def migrate_nous_project(
    rest_client: SCRESTClient,
    export_path: Union[str, os.PathLike],
    project_type: str,
    project_name: Optional[str] = None,
    labels: Optional[Union[List[str], List[Dict[str, Any]]]] = None
):
    """
    README:
    For single task projects we only need to supply the exported NOUS project .zip,
    the new project name on SC, and the project type

    From the annotation alone, in single task projects, we can create a task on SC based
    on the labels we discover in the annotation folder from nous

    Example:
    migrate_nous_project(client,
                      [nous_Export.zip],
                      [project_type],
                      [project name])

    DONE:
    Tested with Det->Seg NOUS exported chain with 1 label per task

    DONE:
    Tested with NOUS single task project exports
    - detection
    - classification
    - segmentation

    #TODO
    - Test with anomaly tasks [only anomaly classification is in SC]
    - Test with multiple labels and hierarchical labels
    """

    # if no project name supplied then use the zip filename
    if project_name is None:
        project_name = os.path.split(export_path)[1].split('.')[0]

    temp_dir = unzip_to_temp(export_path)

    # Create NOUS annotation reader
    annotation_reader = NOUSAnnotationReader(
        base_data_folder=os.path.join(temp_dir, "annotation"),
        task_type=TaskType(project_type)
    )

    # Read all the labels from the annotation files
    '''
    For single task projects, all the labels found in the 
    annotation files will be used when creating a new project on SC. 
    
    If the `labels` argument is passed, the labels will not be determined from the 
    annotation reader but the label data inside the `labels` variable will be used 
    instead.
    '''
    if labels is None:
        labels = annotation_reader.get_all_label_names()

    # Create project
    project_manager = ProjectManager(
        session=rest_client.session, workspace_id=rest_client.workspace_id
    )
    project = project_manager.get_or_create_project(
        project_name=project_name,
        project_type=project_type,
        labels=[labels]
    )

    # Disable auto training
    configuration_manager = ConfigurationManager(
        session=rest_client.session,
        workspace_id=rest_client.workspace_id,
        project=project
    )
    configuration_manager.set_project_auto_train(auto_train=False)

    # Upload images
    image_manager = ImageManager(
        session=rest_client.session,
        workspace_id=rest_client.workspace_id,
        project=project
    )
    images = image_manager.upload_folder(
        path_to_folder=os.path.join(temp_dir, "images")
    )

    # Upload videos
    video_manager = VideoManager(
        workspace_id=rest_client.workspace_id,
        session=rest_client.session,
        project=project
    )
    videos = video_manager.upload_folder(
        path_to_folder=os.path.join(temp_dir, "videos")
    )

    # Set annotation reader task type
    annotation_reader.task_type = project.get_trainable_tasks()[0].type
    annotation_reader.prepare_and_set_dataset(
        task_type=project.get_trainable_tasks()[0].type)

    # Upload annotations
    annotation_manager = NOUSAnnotationManager(
        session=rest_client.session,
        project=project,
        workspace_id=rest_client.workspace_id,
        annotation_reader=annotation_reader,
    )

    annotation_manager.upload_annotations_for_images(images)
    annotation_manager.upload_annotations_for_videos(videos)

    # clean up temp folder
    print('Cleaning up...')
    shutil.rmtree(temp_dir)
    print('=================== Done ===================')


def migrate_nous_chain(
    rest_client: SCRESTClient,
    export_path: Union[str, os.PathLike],
    task_types: List[str],
    labels_per_task: List[Union[List[str], List[Dict[str, Any]]]],
    project_name: Optional[str] = None
):
    """
    NOTE:
    I'm sure the task-chain annotations could be uploaded in a single step
    but this was the quickest way I could see to implement it

    README:
    For task-chains we need to tell the script which labels belong to which task.
    For example, a chained annotation for Det->Class would have a structure like:
    shapes:
    rect: [x,y,w,h]
      labels:['person', 'hat']

    From this information alone it is NOT possible to know which label is for the Det
    task or Class task.
    Therefore, when migrating a chain from NOUS to SC we need to provide which labels
    belong to which task.

    Example:
    migrate_nous_chain(client,
                    [nous_Export.zip],
                    [project name],
                    [TASK1, TASK2],
                    [[TASK1_LABEL_1, TASK1_LABEL_2],[TASK2_LABEL_1, TASK2_LABEL_2]])

    DONE:
    Tested with Det->Seg NOUS exported chain with 1 label per task

    TODO
    - Test with other chains
    - Det->Class [supported by SC]
    - Seg->Class [not currently supported by SC]
    - Test with Multiple labels
    - Test with hierarchical labels
    """

    project_type = '_to_'.join(task_types)
    temp_dir = unzip_to_temp(export_path)

    label_source_per_task = []
    for task_type in get_task_types_by_project_type(project_type):
        annotation_reader = NOUSAnnotationReader(
            base_data_folder=os.path.join(temp_dir, "annotation"),
            task_type=task_type
        )
        label_source_per_task.append(annotation_reader)

    annotation_readers_per_task = [
        entry if isinstance(entry, NOUSAnnotationReader) else None
        for entry in label_source_per_task
    ]

    # Create project
    project_manager = ProjectManager(
        session=rest_client.session, workspace_id=rest_client.workspace_id
    )
    project = project_manager.get_or_create_project(
        project_name=project_name,
        project_type=project_type,
        labels=labels_per_task
    )

    # Disable auto training
    configuration_manager = ConfigurationManager(
        session=rest_client.session,
        workspace_id=rest_client.workspace_id,
        project=project
    )
    configuration_manager.set_project_auto_train(auto_train=False)

    # Upload images
    image_manager = ImageManager(
        session=rest_client.session,
        workspace_id=rest_client.workspace_id,
        project=project
    )
    images = image_manager.upload_folder(
        path_to_folder=os.path.join(temp_dir, "images")
    )

    # Upload videos
    video_manager = VideoManager(
        workspace_id=rest_client.workspace_id,
        session=rest_client.session,
        project=project
    )
    videos = video_manager.upload_folder(
        path_to_folder=os.path.join(temp_dir, "videos")
    )

    # Process the first task as normal
    # Filter for the first task labels
    # Upload annotations for the first task

    # Set annotation reader task type
    annotation_readers_per_task[0].task_type = project.get_trainable_tasks()[0].type
    annotation_readers_per_task[0].prepare_and_set_dataset(
        task_type=project.get_trainable_tasks()[0].type
    )
    annotation_readers_per_task[0].set_labels_filter(labels_per_task[0])

    # Upload annotations
    annotation_manager = NOUSAnnotationManager(
        session=rest_client.session,
        project=project,
        workspace_id=rest_client.workspace_id,
        annotation_reader=annotation_readers_per_task[0],
    )

    annotation_manager.upload_annotations_for_images(images)
    annotation_manager.upload_annotations_for_videos(videos)

    # Now process the second task
    # Filter on the second task labels
    # Upload with the append_annotations option to 'add' second task annotation to
    # the first

    # Set annotation reader task type
    annotation_readers_per_task[1].task_type = project.get_trainable_tasks()[1].type
    annotation_readers_per_task[1].prepare_and_set_dataset(
        task_type=project.get_trainable_tasks()[1].type
    )
    annotation_readers_per_task[1].set_labels_filter(labels_per_task[1])

    # Upload annotations
    annotation_manager = NOUSAnnotationManager(
        session=rest_client.session,
        project=project,
        workspace_id=rest_client.workspace_id,
        annotation_reader=annotation_readers_per_task[1],
    )

    annotation_manager.upload_annotations_for_images(images, append_annotations=True)
    annotation_manager.upload_annotations_for_videos(videos, append_annotations=True)

    # clean up temp folder
    print('Cleaning up...')
    shutil.rmtree(temp_dir)
    print('=================== Done ===================')


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument(
        '-h', '--help', action='help', default=SUPPRESS,
        help='Show this help message and exit.'
    )
    args.add_argument(
        "-host", "--host", help="Required. Sonoma creek host.", required=True, type=str
    )
    args.add_argument(
        "-user", "--user", help="Required. Sonoma creek username.",
        required=True, type=str
    )
    args.add_argument(
        "-pass", "--password", help="Required. Sonoma creek password.",
        required=True, type=str
    )
    args.add_argument(
        "-path", "--path", help="Required. Path to nous exported project.",
        required=True, type=str
    )
    args.add_argument(
        "-n", "--name", help="Required. Project name", required=True, type=str
    )
    args.add_argument(
        "-t", "--task", required=True, type=str,
        help="Required. Project task. detection, classification, segmentation or chain"
    )
    args.add_argument(
        "-ct", "--chain_task", required=False, nargs='+',
        help='Optional. Tasks used for chain project. '
             'Format: -tc detection segmentation'
    )
    args.add_argument(
        "-l", "--labels", required=False, nargs='+',
        help='Optional. Labels used for chain project. Format: -l person hat'
    )

    return parser


def main():
    args = build_argparser().parse_args()

    client = SCRESTClient(host=args.host, username=args.user, password=args.password)

    if args.task == 'chain':
        migrate_nous_chain(
            rest_client=client,
            export_path=args.path,
            project_name=args.name,
            task_types=args.chain_task,
            labels_per_task=[[label] for label in args.labels]
        )

    else:
        migrate_nous_project(
            rest_client=client,
            export_path=args.path,
            project_type=args.task,
            project_name=args.name
        )


if __name__ == '__main__':
    sys.exit(main() or 0)
