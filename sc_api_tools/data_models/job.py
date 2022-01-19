from typing import Optional

import attr
from sc_api_tools.data_models.enums import JobState, JobType
from sc_api_tools.data_models.status import StatusSummary
from sc_api_tools.data_models.utils import str_to_enum_converter


@attr.s(auto_attribs=True)
class JobStatus(StatusSummary):
    """
    This class represents a the current status of a job on the SC cluster

    :var state: Current state of the job
    """
    state: str = attr.ib(converter=str_to_enum_converter(JobState))


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
