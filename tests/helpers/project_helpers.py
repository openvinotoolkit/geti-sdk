from typing import List, Optional, Sequence

from sc_api_tools import SCRESTClient

from sc_api_tools.annotation_readers import AnnotationReader
from sc_api_tools.data_models import TaskType
from sc_api_tools.rest_managers import ProjectManager

from .finalizers import force_delete_project
from .constants import PROJECT_PREFIX
from .project_service import ProjectService


def get_or_create_annotated_project_for_test_class(
        project_service: ProjectService,
        annotation_readers: Sequence[AnnotationReader],
        project_name: str,
        project_type: str = "detection",
        enable_auto_train: bool = False,
        learning_parameter_settings: str = "minimal"
):
    """
    This function returns an annotated project with `project_name` of type
    `project_type`.

    :param project_service: ProjectService instance to which the project should be added
    :param annotation_readers: List of AnnotationReader instances from which to get the
        annotations. The number of annotation readers must match the number of
        trainable tasks in the project.
    :param project_name: Name of the project
    :param project_type: Type of the project
    :param enable_auto_train: True to turn auto-training on, False to leave it off
    :param learning_parameter_settings: Settings to use for the learning parameters
        during model training. There are three options:
          'minimal'     - Set hyper parameters such that the training time is minimized
                          (i.e. single epoch, low batch size, etc.)
          'default'     - Use default hyper parameter settings
          'reduced_mem' - Reduce the batch size for memory intensive tasks
    :return: Project instance corresponding to the project on the SC server
    """
    project_exists = project_service.has_project
    labels = [reader.get_all_label_names() for reader in annotation_readers]

    project = project_service.get_or_create_project(
        project_name=project_name,
        project_type=project_type,
        labels=labels
    )
    if not project_exists:
        project_service.set_auto_train(False)
        if learning_parameter_settings == 'minimal':
            project_service.set_minimal_training_hypers()
        elif learning_parameter_settings == 'reduced_mem':
            # Reduce batch size in memory intensive tasks to avoid OOM errors in pods
            for task in project.get_trainable_tasks():
                if task.type in [
                    TaskType.DETECTION,
                    TaskType.ROTATED_DETECTION,
                    TaskType.INSTANCE_SEGMENTATION,
                ]:
                    task_hypers = project_service.configuration_manager.get_task_configuration(task_id=task.id)
                    task_hypers.batch_size.value = 2
                    project_service.configuration_manager.set_configuration(task_hypers)
        elif learning_parameter_settings != 'default':
            print(
                f"Invalid learning parameter settings '{learning_parameter_settings}' "
                f"specified, continuing with default hyper parameters."
            )

        project_service.add_annotated_media(
            annotation_readers=annotation_readers,
            n_images=-1
        )
        project_service.set_auto_train(enable_auto_train)
    return project


def remove_all_test_projects(client: SCRESTClient) -> List[str]:
    """
    Removes all projects created in the REST SDK tests from the server.

    WARNING: This will remove projects without asking for confirmation. Use with
    caution!

    :param client: Client to the server from which to remove all projects created by
        the SDK test suite
    """
    project_manager = ProjectManager(
        session=client.session, workspace_id=client.workspace_id
    )
    projects_removed: List[str] = []
    for project in project_manager.get_all_projects():
        if project.name.startswith(PROJECT_PREFIX):
            force_delete_project(project.name, project_manager)
            projects_removed.append(project.name)
    print(f"{len(projects_removed)} test projects were removed from the server.")
    return projects_removed
