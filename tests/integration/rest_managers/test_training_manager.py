import pytest
import time

from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.data_models import Job, Project, ProjectStatus
from sc_api_tools.data_models.enums import JobState

from tests.helpers import ProjectService, \
    get_or_create_annotated_project_for_test_class, SdkTestMode
from tests.helpers.constants import PROJECT_PREFIX


class TestTrainingManager:
    @staticmethod
    def ensure_annotated_project(
            project_service: ProjectService, annotation_reader: DatumAnnotationReader
    ) -> Project:
        return get_or_create_annotated_project_for_test_class(
            project_service=project_service,
            annotation_readers=[annotation_reader],
            project_type="detection",
            project_name=f"{PROJECT_PREFIX}_training_manager"
        )

    @pytest.mark.vcr()
    def test_train_task_and_get_jobs(
            self,
            fxt_project_service: ProjectService,
            fxt_annotation_reader: DatumAnnotationReader,
            fxt_test_mode: SdkTestMode
    ) -> None:
        """
        Verifies that submitting a training job for a task in a project with
        sufficient annotations works

        """
        project = self.ensure_annotated_project(project_service=fxt_project_service,
                                                annotation_reader=fxt_annotation_reader)
        if fxt_test_mode != SdkTestMode.OFFLINE:
            time.sleep(5)

        task = project.get_trainable_tasks()[0]
        job = fxt_project_service.training_manager.train_task(
            task=task
        )
        assert isinstance(job, Job)

        jobs = fxt_project_service.training_manager.get_jobs(project_only=True)
        assert job.id in [project_job.id for project_job in jobs]

        # Update job status
        job.update(fxt_project_service.session)
        job_state = job.status.state
        if job_state in JobState.active_states():
            # Cancel the job
            print(f"Job '{job.name}' is still active, cancelling...")
            job.cancel(fxt_project_service.session)
        else:
            print(f"Job '{job.name}' has already excited with status {job_state}.")
        print(job_state)

    @pytest.mark.vcr()
    def test_get_status(
            self,
            fxt_project_service: ProjectService,
            fxt_annotation_reader: DatumAnnotationReader
    ) -> None:
        """
        Test that fetching project status works

        """
        self.ensure_annotated_project(project_service=fxt_project_service,
                                      annotation_reader=fxt_annotation_reader)
        status = fxt_project_service.training_manager.get_status()
        assert isinstance(status, ProjectStatus)
