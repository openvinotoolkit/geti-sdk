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
import os
from typing import Dict, Tuple

from dotenv import dotenv_values


def get_server_details_from_env(
    env_file_path: str = ".env", use_global_variables: bool = False
) -> Tuple[str, Dict[str, str]]:
    """
    Retrieve the server information (hostname and authentication details) from
    environment variables.

    By default, this method will attempt to find a file ".env" in the current working
    directory and attempt to read the information from that file. To read from a
    different file, simply specify the path to the file in `env_file_path`. The
    following variables are relevant:

        HOST     -> hostname or ip address of the Geti server
        TOKEN    -> Personal Access Token that can be used for authorization

    In addition, authentication via credentials is also supported. In that case, the
    following variables should be provided:

        USERNAME -> username to log in to the Geti server
        PASSWORD -> password for logging in to the Geti server

    If both TOKEN, USERNAME and PASSWORD are provided, the method will use the preferred
    token authorization.

    If `use_global_variables` is set to `True`, the method attempts to retrieve the
    server details from the global environment variables instead of
    reading from file. In that case, the `GETI_` prefix should be added to the
    variable names (i.e. `HOST` becomes `GETI_HOST`, `TOKEN` becomes `GETI_TOKEN`, etc.)

    :param env_file_path: Path to the file containing the server details
    :param use_global_variables: If set to True, the method will not read the server
        details from a file but will use environment variables instead
    :return: Tuple consisting of:
      - string holding the hostname or ip address of the server
      - Dictionary containing the authentication information. This can either contain
          the personal access token or the username/password to log in to the Geti
          server.
    """
    if use_global_variables:
        host_key = "GETI_HOST"
        token_key = "GETI_TOKEN"
        username_key = "GETI_USERNAME"
        password_key = "GETI_PASSWORD"

        retrieval_func = os.environ.get
        env_name = "environment variables"
    else:
        host_key = "HOST"
        token_key = "TOKEN"
        username_key = "USERNAME"
        password_key = "PASSWORD"

        env_variables = dotenv_values(dotenv_path=env_file_path)
        if not env_variables:
            raise ValueError(
                f"Unable to load Geti server details from environment file at path "
                f"'{env_file_path}', please make sure the file exists."
            )
        retrieval_func = env_variables.get
        env_name = "environment file"

    hostname = retrieval_func(host_key, None)
    if hostname is None:
        raise ValueError(
            f"Unable to find Geti hostname in the {env_name}. Please make sure "
            f"that the variable {host_key} is defined."
        )

    token = retrieval_func(token_key, None)
    if token is None:
        user = retrieval_func(username_key, None)
        password = retrieval_func(password_key, None)
        if user is None or password is None:
            raise ValueError(
                f"Unable to find either personal access token or user credentials in "
                f"the {env_name}. Please make sure that either '{token_key}' or "
                f"'{username_key}' and '{password_key}' is defined in the environment."
            )
        authentication_dict = {"username": user, "password": password}
    else:
        authentication_dict = {"token": token}
    return hostname, authentication_dict
