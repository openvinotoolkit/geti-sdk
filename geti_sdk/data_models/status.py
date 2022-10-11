# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, Dict, List, Optional

import attr

from geti_sdk.data_models.performance import Performance


@attr.s(auto_attribs=True)
class StatusSummary:
    """
    Summary of the status of a project or task on the GETi cluster.

    :var message: Human readable message describing the status
    :var progress: Training progress, if a model is being trained
    :var time_remaining: Estimated time remaining on the training process, if a model
        is being trained.
    """

    message: str
    progress: float
    time_remaining: float

    @classmethod
    def from_dict(cls, status_dict: Dict[str, Any]) -> "StatusSummary":
        """
        Create a StatusSummary object from a dictionary.

        :param status_dict: Dictionary representing a status, as returned by the GETi
            /status and /jobs endpoints
        :return: StatusSummary object holding the status data contained in `status_dict`
        """
        return cls(**status_dict)


@attr.s(auto_attribs=True)
class LabelAnnotationRequirements:
    """
    Detailed information regarding the required number of annotations for a
    specific label.

    :var id: Unique database ID of the label
    :var label_name: Name of the label
    :var label_color: Color of the label
    :var value: Required number of annotations for this label
    """

    id: str
    label_name: str
    label_color: str
    value: int


@attr.s(auto_attribs=True)
class AnnotationRequirements:
    """
    Container holding the required number of annotations before auto-training can be
    triggered for a task in GETi.

    :var value: Total number of required annotations for the task
    :var details: Required annotations per label
    """

    details: List[LabelAnnotationRequirements]
    value: int


@attr.s(auto_attribs=True)
class TaskStatus:
    """
    Status of a single task in GETi.

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
    Status of a project in GETi.

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
    status: StatusSummary
    tasks: List[TaskStatus]
    project_performance: Optional[Performance] = None
    project_score: Optional[float] = None  # Deprecated in Geti 1.0, to be removed
    n_running_jobs: Optional[int] = None
    n_running_jobs_project: Optional[int] = None

    @property
    def summary(self) -> str:
        """
        Return a string that gives a very brief summary of the project status.

        :return: String holding a brief summary of the project status
        """
        summary_str = f"Project status:\n  {self.status.message}\n"
        for task in self.tasks:
            summary_str += (
                f"    Task: {task.title}\n" f"      State: {task.status.message}\n"
            )
            if task.is_training or task.status.progress != -1.0:
                summary_str += f"      Progress: {task.status.progress:.1f}%\n"
        summary_str += f"  Latest score: {self.project_performance.score*100:.1f}%\n"
        return summary_str
