from typing import Optional, Dict, Any

import attr
from sc_api_tools.data_models.enums import JobState, JobType
from sc_api_tools.data_models.status import StatusSummary
from sc_api_tools.data_models.utils import str_to_enum_converter
from sc_api_tools.http_session import SCSession


@attr.s(auto_attribs=True)
class JobStatus(StatusSummary):
    """
    This class represents a the current status of a job on the SC cluster

    :var state: Current state of the job
    """
    state: str = attr.ib(converter=str_to_enum_converter(JobState))

    @classmethod
    def from_dict(cls, status_dict: Dict[str, Any]) -> 'JobStatus':
        """
        Creates a JobStatus object from a dictionary

        :param status_dict: Dictionary representing a status, as returned by the SC
            /status and /jobs endpoints
        :return: JobStatus object holding the status data contained in `status_dict`
        """
        return cls(**status_dict)


@attr.s(auto_attribs=True)
class TaskMetadata:
    """
    This class holds metadata related to a task on the SC cluster

    :var name: Name of the task
    :var model_template_id: Identifier of the model template used by the task
    :var model_architecture: Name of the neural network architecture used for the model
    :var model_version: Version of the model currently used by the job
    :var dataset_storage_id: Unique database ID of the dataset storage used by the job
    """
    model_template_id: str
    model_architecture: str
    model_version: int
    name: Optional[str] = None
    dataset_storage_id: Optional[str] = None


@attr.s(auto_attribs=True)
class JobMetadata:
    """
    This class holds the metadata for a particular job on the SC cluster

    :var task: TaskMetadata object holding information regarding the task from which
        the job originates
    :var base_model_id: Optional unique database ID of the base model. Only used for
        optimization jobs
    :var model_storage_id: Optional unique database ID of the model storage used by
        the job.
    :var optimization_type: Optional type of the optimization method used in the job.
        Only used for optimization jobs
    :var optimized_model_id: Optional unique database ID of the optimized model
        produced by the job. Only used for optimization jobs.
    """
    task: TaskMetadata
    base_model_id: Optional[str] = None
    model_storage_id: Optional[str] = None
    optimization_type: Optional[str] = None
    optimized_model_id: Optional[str] = None


@attr.s(auto_attribs=True)
class Job:
    """
    This class contains information about a job on the SC cluster

    :var name: Name of the job
    :var description: Description of the job
    :var id: Unique database ID of the job
    :var project_id: Unique database ID of the project from which the job originates
    :var status: JobStatus object holding the current status of the job
    :var type: Type of the job
    :var metadata: JobMetadata object holding metadata for the job
    """
    name: str
    description: str
    id: str
    project_id: str
    status: JobStatus
    type: str = attr.ib(converter=str_to_enum_converter(JobType))
    metadata: JobMetadata

    def __attrs_post_init__(self):
        self._workspace_id: Optional[str] = None

    @property
    def workspace_id(self) -> str:
        """
        Returns the unique database ID of the workspace to which the job belongs

        :return: Unique database ID of the workspace to which the job belongs
        """
        if self._workspace_id is None:
            raise ValueError(
                f"Workspace ID for job {self} is unknown, it was never set."
            )
        return self._workspace_id

    @workspace_id.setter
    def workspace_id(self, workspace_id: str):
        """
        Sets the workspace id for the job

        :param workspace_id: Unique database ID of the workspace to which the job
            belongs
        """
        self._workspace_id = workspace_id

    @property
    def relative_url(self) -> str:
        """
        Returns the url at which the Job can be addressed on the SC cluster, relative
        to the url of the cluster itself

        :return: Relative url for the Job instance
        """
        return f"workspaces/{self.workspace_id}/jobs/{self.id}"

    def update(self, session: SCSession) -> 'Job':
        """
        Updates the job status to its current value, by making a request to the SC
        cluster addressed by `session`

        :param session: SCSession to the cluster from which the Job originates
        :raises ValueError: If no workspace_id has been set for the job prior to
            calling this method
        :return: Job with its status updated
        """
        response = session.get_rest_response(
            url=self.relative_url,
            method='GET'
        )
        updated_status = JobStatus.from_dict(response["status"])
        self.status = updated_status
        return self

    def cancel(self, session: SCSession) -> 'Job':
        """
        Cancels and deletes the job, by making a request to the SC cluster addressed
        by `session`

        :param session: SCSession to the cluster on which the Job is running
        :return: Job with updated status
        """
        session.get_rest_response(
            url=self.relative_url,
            method='DELETE'
        )
        self.status.state = JobState.CANCELLED
        return self
