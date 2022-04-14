import pytest

from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.data_models import Job, Project, ProjectStatus
from sc_api_tools.data_models.enums import JobState

from tests.helpers import ProjectService


class TestTrainingManager:
    @staticmethod
    def ensure_test_project(
            project_service: ProjectService, annotation_reader: DatumAnnotationReader
    ) -> Project:
        project_exists = project_service.has_project
        project = project_service.get_or_create_project(
            project_name="sdk_test_training_manager",
            project_type="detection",
        )
        if not project_exists:
            project_service.set_auto_train(False)
            project_service.set_minimal_training_hypers()
            project_service.add_annotated_media(
                annotation_reader=annotation_reader,
                n_images=-1
            )
        return project

    @pytest.mark.vcr()
    def test_train_task_and_get_jobs(
            self,
            fxt_project_service: ProjectService,
            fxt_annotation_reader: DatumAnnotationReader
    ) -> None:
        """
        Verifies that submitting a training job for a task in a project with
        sufficient annotations works

        """
        project = self.ensure_test_project(
            project_service=fxt_project_service,
            annotation_reader=fxt_annotation_reader
        )

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
        self.ensure_test_project(
            project_service=fxt_project_service,
            annotation_reader=fxt_annotation_reader
        )
        status = fxt_project_service.training_manager.get_status()
        assert isinstance(status, ProjectStatus)
