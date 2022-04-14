import pytest

from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.data_models import Job

from tests.helpers import ProjectService


class TestTrainingManager:
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
        project = fxt_project_service.create_project(project_type='detection')
        fxt_project_service.set_auto_train(False)
        fxt_project_service.set_minimal_training_hypers()

        fxt_project_service.add_annotated_media(
            annotation_reader=fxt_annotation_reader,
            n_images=-1
        )

        task = project.get_trainable_tasks()[0]
        job = fxt_project_service.training_manager.train_task(
            task=task
        )
        assert isinstance(job, Job)

        jobs = fxt_project_service.training_manager.get_jobs(project_only=True)
        assert job.id in [project_job.id for project_job in jobs]

        # Cancel the job
        job.cancel(fxt_project_service.session)
