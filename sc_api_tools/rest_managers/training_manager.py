import time
from typing import Union, Optional, List

from sc_api_tools.data_models import ProjectStatus, Project, Task, Algorithm, Job
from sc_api_tools.data_models.containers import AlgorithmList
from sc_api_tools.data_models.enums import JobState
from sc_api_tools.data_models.project import Dataset
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import StatusRESTConverter, JobRESTConverter
from sc_api_tools.utils import get_supported_algorithms


class TrainingManager:
    """
    Class to manage training jobs for a certain project
    """

    def __init__(self, workspace_id: str, project: Project, session: SCSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}"
        self.supported_algos = get_supported_algorithms(session)

    def get_status(self) -> ProjectStatus:
        """
        Gets the current status of the project from the SC cluster

        :return: ProjectStatus object reflecting the current project status
        """
        response = self.session.get_rest_response(
            url=f"{self.base_url}/status",
            method="GET"
        )
        return StatusRESTConverter.from_dict(response)

    def get_jobs(self, project_only: bool = True) -> List[Job]:
        """
        Returns a list of all jobs on the SC cluster.

        If `project_only = True` (the default), only those jobs related to the project
        managed by this TrainingManager will be returned. If set to False, all jobs in
        the workspace are returned.

        :param project_only: True to return only those jobs pertaining to the project
            for which the TrainingManager is active. False to return all jobs in the
            SC workspace.
        :return: List of Jobs
        """
        response = self.session.get_rest_response(
            url=f"workspaces/{self.workspace_id}/jobs",
            method="GET"
        )
        job_list: List[Job] = []
        for job_dict in response["items"]:
            job = JobRESTConverter.from_dict(job_dict)
            job.workspace_id = self.workspace_id
            job_list.append(job)

        if project_only:
            return [job for job in job_list if job.project_id == self.project.id]
        else:
            return job_list

    def get_algorithms_for_task(self, task: Union[Task, int]) -> AlgorithmList:
        """
        Returns a list of supported algorithms for a specific task

        The `task` parameter accepts both a Task object and an integer. If an int is
        passed, this will be considered the index of the task in the list of trainable
        tasks for the project which is managed by the TrainingManager.

        :param task: Task to get the supported algorithms for. If an integer is passed,
            this is considered the index of the task in the trainable task list of the
            project. So passing `task=0` will return the algorithms for the first
            trainable task, etc.
        :return: List of supported algorithms for the task
        """
        if isinstance(task, int):
            task = self.project.get_trainable_tasks()[task]
        return self.supported_algos.get_by_task_type(task.type)

    def train_task(
            self,
            task: Union[Task, int],
            dataset: Optional[Dataset] = None,
            algorithm: Optional[Algorithm] = None,
            train_from_scratch: bool = False,
            enable_pot_optimization: bool = False
    ) -> Job:
        """
        Start training of a specific task in the project

        The `task` parameter accepts both a Task object and an integer. If an int is
        passed, this will be considered the index of the task in the list of trainable
        tasks for the project which is managed by the TrainingManager.

        :param task: Task or index of Task to train
        :param dataset: Optional Dataset to train on
        :param algorithm: Optional Algorithm to use in training. If left as None (the
            default), the first algorithm on the list of supported algorithms for the
            task will be used.
        :param train_from_scratch: True to train the model from scratch, False to
            continue training from an existing checkpoint (if any)
        :param enable_pot_optimization: True to optimize the trained model with POT
            after training is complete
        :return:
        """
        if isinstance(task, int):
            task = self.project.get_trainable_tasks()[task]
        if dataset is None:
            dataset = self.project.datasets[0]
        if algorithm is None:
            algorithm = self.get_algorithms_for_task(task)[0]
        request_data = [
            {
                "dataset_id": dataset.id,
                "task_id": task.id,
                "train_from_scratch": train_from_scratch,
                "enable_pot_optimization": enable_pot_optimization,
                "model_template_id": algorithm.model_template_id
            }
        ]
        response = self.session.get_rest_response(
            url=f"{self.base_url}/train",
            method="POST",
            data=request_data
        )
        job = JobRESTConverter.from_dict(response)
        job.workspace_id = self.workspace_id
        return job

    def monitor_jobs(self, jobs: List[Job]) -> List[Job]:
        """
        Monitors and prints the progress of all jobs in the list `jobs`. Execution is
        halted until all jobs have either finished, failed or were cancelled.

        Progress will be reported in 15s intervals

        :param jobs: List of jobs to monitor
        :return: List of finished (or failed) jobs with their status updated
        """
        monitoring = True
        print('---------------- Monitoring progress -------------------')
        try:
            while monitoring:
                msg = ''
                complete_count = 0
                for job in jobs:
                    job.update(self.session)
                    msg += (
                        f"{job.description} -- State: {job.status.state} -- "
                        f"Progress: {job.status.progress:.2f}%\n"
                    )
                    if job.status.state in [
                        JobState.FINISHED,
                        JobState.CANCELLED,
                        JobState.FAILED,
                        JobState.ERROR
                    ]:
                        complete_count += 1
                if complete_count == len(jobs):
                    monitoring = False
                print(msg, end='')
                time.sleep(15)
        except KeyboardInterrupt:
            print("Job monitoring interrupted, stopping...")
            for job in jobs:
                job.update(self.session)
            return jobs
        print("All jobs completed, monitoring stopped.")
        return jobs
