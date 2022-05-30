from typing import Optional

from sc_api_tools.data_models import TaskType
from sc_api_tools.data_models.containers import AlgorithmList
from sc_api_tools.data_models.enums import Domain
from sc_api_tools.http_session import SCSession


def get_supported_algorithms(
        rest_session: SCSession,
        domain: Optional[Domain] = None,
        task_type: Optional[TaskType] = None
) -> AlgorithmList:
    """
    Returns the id of the default workspace on the cluster

    :param rest_session: HTTP session to the cluster
    :param domain: Optional domain for which to get the supported algorithms. If left
        as None (the default), the supported algorithms for all domains are returned.
        NOTE: domain is deprecated in SC1.1, please use `task_type` instead.
    :param task_type: Optional TaskType for which to get the supported algorithms.
    :return: AlgorithmList holding the supported algorithms
    """
    if task_type is not None and domain is not None:
        raise ValueError("Please specify either task type or domain, but not both")
    elif task_type is not None:
        if rest_session.version == '1.0':
            query = f'?domain={Domain.from_task_type(task_type)}'
        else:
            query = f'?task_type={task_type}'
    elif domain is not None:
        if rest_session.version == '1.0':
            query = f'?domain={domain}'
        else:
            query = f'?task_type={TaskType.from_domain(domain)}'
    else:
        query = ''

    algorithm_rest_response = rest_session.get_rest_response(
        url=f"supported_algorithms{query}",
        method="GET"
    )
    return AlgorithmList.from_rest(algorithm_rest_response)
