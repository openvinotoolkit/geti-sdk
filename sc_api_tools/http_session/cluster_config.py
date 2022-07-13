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

from dataclasses import dataclass
from typing import Dict, Optional

API_PATTERN = "/api/v1.0/"


@dataclass
class ClusterConfig:
    """
    Configuration for requests sessions, with host, username and password.

    :var host: full hostname or ip address of the SC instance.
        Note: this should include the protocol (i.e. https://your_sc_hostname.com)
    :var username: Username to log in to the instance.
    :var password: Password required to log in to the instance
    :var has_valid_certificate: Set to True if the server has a valid SSL certificate
        that should be validated and used to establish an encrypted HTTPS connection
    :var proxies: Optional dictionary containing proxy information, if this is
        required to connect to the SC server. For example:

        proxies = {
            'http': http://proxy-server.com:<http_port_number>,
            'https': http://proxy-server.com:<https_port_number>
        },

        if set to None (the default), no proxy settings will be used.
    """

    host: str
    username: str
    password: str
    has_valid_certificate: bool = False
    proxies: Optional[Dict[str, str]] = None

    @property
    def base_url(self) -> str:
        """
        Return the base UR for accessing the cluster.
        """
        return f"{self.host}{API_PATTERN}"
