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

from geti_sdk.http_session import GetiSession


def get_default_workspace_id(rest_session: GetiSession) -> str:
    """
    Return the id of the default workspace on the cluster.

    :param rest_session: HTTP session to the cluser
    :return: string containing the id of the default workspace
    """
    workspaces = rest_session.get_rest_response(url="workspaces", method="GET")
    if isinstance(workspaces, list):
        workspace_list = workspaces
    elif isinstance(workspaces, dict):
        if rest_session.version.is_sc_mvp or rest_session.version.is_sc_1_1:
            workspace_list = workspaces["items"]
        else:
            workspace_list = workspaces["workspaces"]
    else:
        raise ValueError(
            f"Unexpected response from cluster: {workspaces}. Expected to receive a "
            f"dictionary containing workspace data."
        )
    default_workspace_names = ["default", "default workspace", "default_workspace"]
    default_workspace = next(
        (
            workspace
            for workspace in workspace_list
            if workspace["name"].lower() in default_workspace_names
        )
    )
    return default_workspace["id"]
