from sc_api_tools.http_session import SCSession


def get_default_workspace_id(rest_session: SCSession) -> str:
    """
    Returns the id of the default workspace on the cluster

    :param rest_session: HTTP session to the cluser
    :return: string containing the id of the default workspace
    """
    workspaces = rest_session.get_rest_response(
        url="workspaces",
        method="GET"
    )
    if isinstance(workspaces, list):
        workspace_list = workspaces
    elif isinstance(workspaces, dict):
        workspace_list = workspaces["items"]
    else:
        raise ValueError(
            f"Unexpected response from cluster: {workspaces}. Expected to receive a "
            f"dictionary containing workspace data."
        )
    default_workspace = next(
        (workspace for workspace in workspace_list
         if workspace["name"] == "Default Workspace")
    )
    return default_workspace["id"]
