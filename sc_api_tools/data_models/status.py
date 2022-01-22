from typing import List, Dict, Any

import attr


@attr.s(auto_attribs=True)
class StatusSummary:
    """
    This class represents a summary of the status of a project or task on the SC
    cluster

    :var message: Human readable message describing the status
    :var progress: Training progress, if a model is being trained
    :var time_remaining: Estimated time remaining on the training process, if a model
        is being trained.
    """
    message: str
    progress: float
    time_remaining: float

    @classmethod
    def from_dict(cls, status_dict: Dict[str, Any]) -> 'StatusSummary':
        """
        Creates a StatusSummary object from a dictionary

        :param status_dict: Dictionary representing a status, as returned by the SC
            /status and /jobs endpoints
        :return: StatusSummary object holding the status data contained in `status_dict`
        """
        return cls(**status_dict)


@attr.s(auto_attribs=True)
class LabelAnnotationRequirements:
    """
    This class holds information regarding the required number of annotations for a
    specific label

    :var id: Unique database ID of the label
    :var label_name: Name of the label
    :var label_color: Color of the label
    :var value: Required number of annotations for this label
    """
    id: str
    label_name: str
    color: str
    value: int


@attr.s(auto_attribs=True)
class AnnotationRequirements:
    """
    This class holds information regarding the required number of annotations before
    auto-training can be triggered for a task in SC

    :var value: Total number of required annotations for the task
    :var details: Required annotations per label
    """
    details: List[LabelAnnotationRequirements]
    value: int


@attr.s(auto_attribs=True)
class TaskStatus:
    """
    This class represents the status of a single task in SC.

    :var id: Unique database ID of the task
    :var is_training: True if a training job is currently running for the task
    :var required_annotations: AnnotationRequirements object that holds details
        related to the required number of annotations before auto-training will be
        started for the task
    :var status: StatusSummary object that contains (among others) a human readable
        message describing the status of the task
    :var title: Title of the taks
    """
    id: str
    is_training: bool
    required_annotations: AnnotationRequirements
    status: StatusSummary
    title: str


@attr.s(auto_attribs=True)
class ProjectStatus:
    """
    This class represents the status of a project in SC

    :var is_training: True if a training job is currently running for any of the
        tasks in the project
    :var n_required_annotations: Total number of required annotations for the project,
        before auto-training can be started
    :var project_score: Accuracy score for the project
    :var status: StatusSummary object that contains (among others) a human readable
        message describing the status of the project
    :var tasks: List of TaskStatus objects, detailing the status of each task in the
        project
    """
    is_training: bool
    n_required_annotations: int
    project_score: float
    status: StatusSummary
    tasks: List[TaskStatus]
