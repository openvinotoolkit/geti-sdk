from typing import Optional

from sc_api_tools.data_models.containers import AlgorithmList
from sc_api_tools.data_models.enums import Domain
from sc_api_tools.http_session import SCSession


def get_supported_algorithms(
        rest_session: SCSession, domain: Optional[Domain] = None
) -> AlgorithmList:
    """
    Returns the id of the default workspace on the cluster

    :param rest_session: HTTP session to the cluster
    :param domain: Optional domain for which to get the supported algorithms. If left
        as None (the default), the supported algorithms for all domains are returned
    :return: AlgorithmList holding the supported algorithms
    """
    if domain is None:
        query = ''
    else:
        query = f'?domain={domain}'
    algorithm_rest_response = rest_session.get_rest_response(
        url=f"supported_algorithms{query}",
        method="GET"
    )
    return AlgorithmList.from_rest(algorithm_rest_response)
