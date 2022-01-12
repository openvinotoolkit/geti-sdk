from typing import ClassVar, List, Optional, Union

import attr

from sc_api_tools.data_models.utils import str_to_datetime
from sc_api_tools.http_session import SCSession
from sc_api_tools.utils.algorithm_helpers import get_supported_algorithms

from .algorithms import Algorithm
from .model import Model


@attr.s(auto_attribs=True)
class ModelSummary:
    """
    Class representing a Model in SC, containing only the minimal information about
    the model

    :var name: Name of the model
    :var creation_date: Creation date of the model
    :var version: Model version
    :var score: Score that was achieved upon evaluation of the model on the test set
    :var active_model: True if this model was the active model for the project it was
        created in, False if it was not the active model
    :var id: Unique database ID of the model
    :var model_storage_id: Unique database ID of the model storage (also referred to
        as model group) that this model belongs to
    """
    _identifier_fields: ClassVar[List[str]] = ["id", "model_storage_id"]

    name: str
    creation_date: str = attr.ib(converter=str_to_datetime)
    version: int
    score_up_to_date: bool
    score: Optional[float] = attr.ib(default=None)
    active_model: bool = attr.ib(default=False)
    id: Optional[str] = attr.ib(default=None, repr=False)
    model_storage_id: Optional[str] = attr.ib(default=None, repr=False)


@attr.s(auto_attribs=True)
class ModelGroup:
    """
    Class representing a ModelGroup in SC
    """
    _identifier_fields: ClassVar[List[str]] = ["id", "task_id"]

    name: str
    model_template_id: str
    models: List[ModelSummary] = attr.ib(repr=False)
    task_id: Optional[str] = attr.ib(default=None)
    id: Optional[str] = attr.ib(default=None)

    def __attrs_post_init__(self):
        self._algorithm: Optional[Algorithm] = None

    @property
    def has_trained_models(self) -> bool:
        """
        Returns True if the ModelGroup contains at least one trained model

        :return: True if the model group holds at least one trained model, False
            otherwise
        """
        trained_models = [model for model in self.models if model.score is not None]
        return len(trained_models) > 0

    def get_latest_model(self) -> Optional[ModelSummary]:
        """
        Returns the latest model in the model group

        :return:
        """
        if not self.has_trained_models:
            return None
        versions = [model.version for model in self.models]
        return [model for model in self.models if model.version == max(versions)][0]

    def get_algorithm_details(self, session: SCSession) -> Algorithm:
        """
        Get the details for the algorithm corresponding to this ModelGroup

        :param session: REST session to an SC cluster
        :return: Algorithm object holding the algorithm details for the ModelGroup
        """
        if self._algorithm is not None:
            return self._algorithm
        supported_algos = get_supported_algorithms(session)
        self._algorithm = supported_algos.get_by_model_template(self.model_template_id)
        return self._algorithm

    @property
    def algorithm(self) -> Optional[Algorithm]:
        """
        Returns the details for the algorithm corresponding to the ModelGroup
        This property will return None unless the `get_algorithm_details` method is
        called to retrieve the algorithm information from the SC cluster

        :return: Algorithm details, if available
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: Algorithm):
        """
        Sets the algorithm details for this ModelGroup instance

        :param algorithm: Algorithm information to set
        """
        self._algorithm = algorithm

    def contains_model(self, model: Union[ModelSummary, Model]) -> bool:
        """
        Returns True if the model group contains the `model`

        :param model: Model or ModelSummary object
        :return: True if the group contains the model, False otherwise
        """
        return model.id in [model.id for model in self.models]