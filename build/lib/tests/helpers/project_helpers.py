from typing import List

from sc_api_tools import SCRESTClient

from sc_api_tools.annotation_readers import AnnotationReader
from sc_api_tools.rest_managers import ProjectManager

from .finalizers import force_delete_project
from .constants import PROJECT_PREFIX
from .project_service import ProjectService


def get_or_create_annotated_project_for_test_class(
        project_service: ProjectService,
        annotation_reader: AnnotationReader,
        project_name: str,
        project_type: str = "detection",
        enable_auto_train: bool = False
):
    """
    This function returns an annotated project with `project_name` of type
    `project_type`.

    :param project_service: ProjectService instance to which the project should be added
    :param annotation_reader: AnnotationReader from which to get the annotations
    :param project_name: Name of the project
    :param project_type: Type of the project
    :param enable_auto_train: True to turn auto-training on, False to leave it off
    :return: Project instance corresponding to the project on the SC server
    """
    project_exists = project_service.has_project
    project = project_service.get_or_create_project(
        project_name=project_name,
        project_type=project_type,
    )
    if not project_exists:
        project_service.set_auto_train(enable_auto_train)
        project_service.set_minimal_training_hypers()
        project_service.add_annotated_media(
            annotation_reader=annotation_reader,
            n_images=-1
        )
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
