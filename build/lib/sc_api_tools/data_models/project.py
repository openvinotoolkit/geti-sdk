import copy
from typing import List, Dict, Optional, ClassVar, Any, Union
from pprint import pformat

import attr

from sc_api_tools.utils.dictionary_helpers import remove_null_fields

from .performance import Performance
from .label import Label
from .task import Task
from .utils import deidentify, attr_value_serializer


@attr.s(auto_attribs=True)
class Connection:
    """
    Class representing a connection between two tasks in SC

    :var to: Name of the task to which the connection is leading
    :var from_: Name of the task from which the connection originates
    """

    to: str
    from_: str


@attr.s(auto_attribs=True)
class Pipeline:
    """
    Class representing a project pipeline in SC

    :var tasks: List of tasks in the pipeline
    :var connections: List of connections between the tasks in the pipeline
    """

    tasks: List[Task]
    connections: List[Connection]

    @property
    def trainable_tasks(self) -> List[Task]:
        """
        Returns an ordered list of trainable tasks

        :return: List of trainable tasks in the pipeline
        """
        return [task for task in self.tasks if task.is_trainable]

    def resolve_connections(self):
        """
        Replaces the task id's in the connections attribute of the pipeline with the
        actual task titles

        :return:
        """
        task_id_to_name = {task.id: task.title for task in self.tasks}
        for connection in self.connections:
            if connection.to in task_id_to_name.keys():
                connection.to = task_id_to_name[connection.to]
            if connection.from_ in task_id_to_name.keys():
                connection.from_ = task_id_to_name[connection.from_]

    def resolve_parent_labels(self):
        """
        Replaces the parent_id's in the pipeline labels by the actual label names

        :return:
        """
        for task in self.trainable_tasks:
            for label in task.labels:
                if label.parent_id in self.label_id_to_name_mapping.keys():
                    label.parent_id = self.label_id_to_name_mapping[label.parent_id]

    @property
    def label_id_to_name_mapping(self) -> Dict[str, str]:
        """
        Returns a mapping of label ID's to label names for all labels in the pipeline

        :return: dictionary containing the label ID's as keys and the label names as
            values
        """
        label_mapping: Dict[str, str] = {}
        for task in self.trainable_tasks:
            label_mapping.update({label.id: label.name for label in task.labels})
        return label_mapping

    def get_labels_per_task(self, include_empty: bool = True) -> List[List[Label]]:
        """
        Returns a nested list of labels for each task in the pipeline.

        Each entry in the outermost list corresponds to a task in the pipeline. The
        innermost list holds the labels for that specific task

        :param include_empty: True to include empty labels in the output
        :return: nested list of labels for each task in the pipeline
        """
        outer_list: List[List[Label]] = []
        for task in self.trainable_tasks:
            if not include_empty:
                task_labels = [label for label in task.labels if not label.is_empty]
            else:
                task_labels = task.labels
            outer_list.append(task_labels)
        return outer_list

    def get_all_labels(self) -> List[Label]:
        """
        Returns a list of all labels in the pipeline

        :return: List of all labels for every task in the pipeline
        """
        label_list: List[Label] = []
        for task in self.trainable_tasks:
            label_list.extend(task.labels)
        return label_list

    def deidentify(self):
        """
        Removes all unique database ID's from the tasks and labels in the pipeline.

        :return:
        """
        self.resolve_connections()
        self.resolve_parent_labels()
        for task in self.tasks:
            task.deidentify()


@attr.s(auto_attribs=True)
class Dataset:
    """
    Class representing a dataset for a project in SC

    :var id: Unique database ID of the dataset
    :var name: name of the dataset
    """

    _identifier_fields: ClassVar[str] = ["id", "thumbnail"]

    name: str
    id: Optional[str] = None

    def deidentify(self):
        """
        Removes unique database ID from the Dataset
        :return:
        """
        deidentify(self)


@attr.s(auto_attribs=True)
class Project:
    """
    Class representing a project in SC

    :var id: Unique database ID of the project
    :var name: Name of the project
    :var creation_time: Time at which the project was created
    :var pipeline: Pipeline for the project
    :var datasets: List of datasets belonging to the project
    :var score: Score achieved by the AI assistant for the project
    :var thumbnail: URL at which a thumbnail for the project can be obtained
    """
    _identifier_fields: ClassVar[str] = ["id", "thumbnail", "creation_time"]

    name: str
    pipeline: Pipeline
    datasets: List[Dataset]
    score: Optional[float] = None  # 'score' is removed in v1.1
    performance: Optional[Performance] = None
    creation_time: Optional[str] = None
    id: Optional[str] = None
    thumbnail: Optional[str] = None

    def get_trainable_tasks(self) -> List[Task]:
        """
        Returns an ordered list of trainable tasks

        :return: List of trainable tasks in the project
        """
        return self.pipeline.trainable_tasks

    @property
    def project_type(self) -> str:
        """
        String containing the project type. The type is constructed by stringing the
        types of tasks in the task chain together, separated by a '_to_' sequence.

        For example, a project with a detection task followed by a segmentation task
        will have it's project type set as: 'detection_to_segmentation'

        :return: string containing the project type
        """
        tasks = self.get_trainable_tasks()
        project_type = ""
        for index, task in enumerate(tasks):
            if index > 0:
                project_type += f"_to_{task.task_type}"
            else:
                project_type += f"{task.task_type}"
        return project_type

    def deidentify(self):
        """
        Removes all unique database ID's from the project, pipeline and datasets

        :return:
        """
        deidentify(self)
        self.pipeline.deidentify()
        for dataset in self.datasets:
            dataset.deidentify()

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the project to a dictionary representation

        :return: Dictionary holding the project data
        """
        output_dict = attr.asdict(self, value_serializer=attr_value_serializer)
        for connection in output_dict["pipeline"]["connections"]:
            from_value = connection.pop("from_")
            connection["from"] = from_value
        return output_dict

    def get_labels_per_task(
            self, include_empty: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Returns a nested list containing the labels for each task in the project.
        Each entry in the outermost list corresponds to a trainable task in the project.

        :param include_empty: True to include empty labels in the output, False to
            exclude them
        :return:
        """
        return [
            [attr.asdict(label) for label in task_labels]
            for task_labels in self.pipeline.get_labels_per_task(
                include_empty=include_empty
            )
        ]

    def get_parameters(self) -> Dict[str, Union[str, List[str]]]:
        """
        Returns the parameters used to create the project

        :return: Dictionary containing the keys `project_name`, `project_type` and
            `labels`
        """
        return {
            "project_name": self.name,
            "project_type": self.project_type,
            "labels": self.get_labels_per_task(include_empty=False)
        }

    def get_all_labels(self) -> List[Label]:
        """
        Returns a list of all labels in the project

        :return: List of all labels in the project
        """
        return self.pipeline.get_all_labels()

    @property
    def overview(self) -> str:
        """
        Returns a string that shows an overview of the project. This still shows all
        the detailed information of the project. If less details are required, please
        use the `summary` property

        :return: String holding an overview of the project
        """
        deidentified = copy.deepcopy(self)
        deidentified.deidentify()
        overview_dict = deidentified.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)

    @property
    def summary(self) -> str:
        """
        Returns a string that gives a very brief summary of the project. This is the
        least detailed representation of the project, if more details are required
        please use the `overview` property

        :return: String holding a brief summary of the project
        """
        summary_str = f"Project: {self.name}\n"
        for task_index, (task, labels) in enumerate(
                zip(self.get_trainable_tasks(), self.pipeline.get_labels_per_task())
        ):
            summary_str += f"  Task {task_index+1}: {task.title}\n    Labels: " \
                            f"{[label.name for label in labels]}\n"
        return summary_str
