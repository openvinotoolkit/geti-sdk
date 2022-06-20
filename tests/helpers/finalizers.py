import time

from sc_api_tools.http_session import SCRequestException
from sc_api_tools.rest_managers import ProjectManager, TrainingManager


def force_delete_project(project_name: str, project_manager: ProjectManager) -> None:
    """
    Deletes the project named 'project_name'. If any jobs are running for the
    project, this finalizer cancels them.

    :param project_name: Name of project to delete
    :param project_manager: ProjectManager to use for project deletion
    """
    try:
        project_manager.delete_project(
            project=project_name, requires_confirmation=False
        )
    except TypeError:
        print(
            f"Project {project_name} was not found on the server, it was most "
            f"likely already deleted."
        )
    except ValueError:
        print(
            f"Unable to delete project '{project_name}' from the server, it "
            f"is most likely locked for deletion due to an operation/training "
            f"session that is in progress. "
            f"\n\n Attempting to cancel the job and re-try project deletion."
        )

        project = project_manager.get_project_by_name(project_name)
        training_manager = TrainingManager(
            workspace_id=project_manager.workspace_id,
            session=project_manager.session,
            project=project
        )
        jobs = training_manager.get_jobs(project_only=True)
        for job in jobs:
            job.cancel(project_manager.session)
        time.sleep(1)
        try:
            project_manager.delete_project(
                project=project_name, requires_confirmation=False
            )
        except ValueError as error:
            raise ValueError(
                f"Unable to force delete project {project_name}, due to: {error} "
            )
