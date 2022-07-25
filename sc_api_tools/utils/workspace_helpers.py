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

from sc_api_tools.http_session import SCSession


def get_default_workspace_id(rest_session: SCSession) -> str:
    """
    Return the id of the default workspace on the cluster.

    :param rest_session: HTTP session to the cluser
    :return: string containing the id of the default workspace
    """
    workspaces = rest_session.get_rest_response(url="workspaces", method="GET")
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
        (
            workspace
            for workspace in workspace_list
            if workspace["name"] == "Default Workspace"
        )
    )
    return default_workspace["id"]