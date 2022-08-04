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
import warnings
from json import JSONDecodeError
from typing import Dict, Optional, Union

import requests
import simplejson
import urllib3
from requests import Response
from requests.exceptions import RequestException
from requests.structures import CaseInsensitiveDict
from urllib3.exceptions import InsecureRequestWarning

from .exception import SCRequestException
from .server_config import LEGACY_API_VERSION, ServerCredentialConfig, ServerTokenConfig

CSRF_COOKIE_NAME = "_oauth2_proxy_csrf"
PROXY_COOKIE_NAME = "_oauth2_proxy"

INITIAL_HEADERS = {"Connection": "keep-alive", "Upgrade-Insecure-Requests": "1"}


class SCSession(requests.Session):
    """
    Wrapper for requests.session that sets the correct headers and cookies.

    :param server_config: Server configuration holding the hostname of the SC server,
        as well as the details required for authentication (either username and
        password or personal access token)
    """

    def __init__(
        self,
        server_config: Union[ServerTokenConfig, ServerCredentialConfig],
    ):
        super().__init__()
        self.headers.update(INITIAL_HEADERS)
        self.allow_redirects = False
        self.token = None
        self._cookies: Dict[str, Optional[str]] = {
            CSRF_COOKIE_NAME: None,
            PROXY_COOKIE_NAME: None,
        }

        # Configure proxies
        if server_config.proxies is None:
            self._proxies: Dict[str, str] = {}
        else:
            self._proxies = {"proxies": server_config.proxies}

        # Sanitize hostname
        if not server_config.host.startswith("https://"):
            if server_config.host.startswith("http://"):
                raise ValueError(
                    "HTTP connections are not supported, please use HTTPS instead."
                )
            else:
                server_config.host = "https://" + server_config.host

        # Configure certificate verification
        if not server_config.has_valid_certificate:
            warnings.warn(
                "You have disabled TLS certificate validation, HTTPS requests made to "
                "the SonomaCreek server may be compromised. For optimal security, "
                "please enable certificate validation.",
                InsecureRequestWarning,
            )
            urllib3.disable_warnings(InsecureRequestWarning)
        self.verify = server_config.has_valid_certificate

        self.config = server_config
        self.logged_in = False

        # Determine authentication method
        if isinstance(server_config, ServerCredentialConfig):
            self.authenticate()
            self.use_token = False
        else:
            self.headers.update({"Authorization": f"Bearer {server_config.token}"})
            self.use_token = True

        # Get server version
        self._product_info = self._get_product_info_and_set_api_version()

    @property
    def version(self) -> str:
        """
        Return the version of SonomaCreek that is running on the server.

        :return: string holding the SC version number
        """
        version_string = self._product_info.get("product-version", "1.0.0-")
        return version_string.split("-")[0]

    def _follow_login_redirects(self, response: Response) -> str:
        """
        Recursively follow redirects in the initial login request. Updates the
        session._cookies with the cookie and the login uri.

        :param response: REST response to follow redirects for
        :return: url to the redirected location
        """
        if response.status_code in [302, 303]:
            redirect_url = response.next.url
            redirected = self.get(redirect_url, allow_redirects=False, **self._proxies)
            proxy_csrf = redirected.cookies.get(CSRF_COOKIE_NAME, None)
            if proxy_csrf:
                self._cookies[CSRF_COOKIE_NAME] = proxy_csrf
            return self._follow_login_redirects(redirected)
        else:
            return response.url

    def _get_initial_login_url(self) -> str:
        """
        Retrieve the initial login url by making a request to the login page, and
        following the redirects.

        :return: string containing the URL to the login page
        """
        response = self.get(
            f"{self.config.host}/user/login", allow_redirects=False, **self._proxies
        )
        login_page_url = self._follow_login_redirects(response)
        return login_page_url

    def authenticate(self, verbose: bool = True):
        """
        Get a new authentication cookie from the server.

        :param verbose: True to print progress output, False to suppress output
        """
        try:
            login_path = self._get_initial_login_url()
        except requests.exceptions.ConnectionError as error:
            if "dummy" in self.config.password or "dummy" in self.config.username:
                raise ValueError(
                    "Connection to Sonoma Creek failed, please make sure to update "
                    "the user login information for the SC cluster."
                ) from error
            raise ValueError(
                f"Connection to Sonoma Creek at host '{self.config.host}' failed,"
                f" please provide a valid cluster hostname or ip address as well as "
                f"valid login details."
            ) from error
        self.headers.clear()
        self.headers.update({"Content-Type": "application/x-www-form-urlencoded"})
        if verbose:
            print(f"Authenticating on host {self.config.host}...")
        response = self.post(
            url=login_path,
            data={"login": self.config.username, "password": self.config.password},
            cookies={CSRF_COOKIE_NAME: self._cookies[CSRF_COOKIE_NAME]},
            headers={"Cookie": self._cookies[CSRF_COOKIE_NAME]},
            allow_redirects=True,
            **self._proxies,
        )
        try:
            previous_response = response.history[-1]
        except IndexError:
            raise ValueError(
                "The cluster responded to the request, but authentication failed. "
                "Please verify that you have provided correct credentials."
            )
        cookie = {PROXY_COOKIE_NAME: previous_response.cookies.get(PROXY_COOKIE_NAME)}
        self._cookies.update(cookie)
        if verbose:
            print("Authentication successful. Cookie received.")
        self.logged_in = True

    def get_rest_response(
        self, url: str, method: str, contenttype: str = "json", data=None
    ) -> Union[Response, dict, list]:
        """
        Return the REST response from a request to `url` with `method`.

        :param url: the REST url without the hostname and api pattern
        :param method: 'GET', 'POST', 'PUT', 'DELETE'
        :param contenttype: currently either 'json', 'jpeg', 'multipart', 'zip', or '',
            defaults to "json"
        :param data: the data to send in a post request, as json
        """
        if url.startswith(self.config.api_pattern):
            url = url[len(self.config.api_pattern) :]

        if contenttype == "json":
            self.headers.update({"Content-Type": "application/json"})
        elif contenttype == "jpeg":
            self.headers.update({"Content-Type": "image/jpeg"})
        elif contenttype == "multipart":
            self.headers.pop("Content-Type", None)
        elif contenttype == "":
            self.headers.pop("Content-Type", None)
        elif contenttype == "zip":
            self.headers.update({"Content-Type": "application/zip"})

        requesturl = f"{self.config.base_url}{url}"
        if contenttype == "json":
            kw_data_arg = {"json": data}
        else:
            kw_data_arg = {"files": data}

        request_params = {
            "method": method,
            "url": requesturl,
            **kw_data_arg,
            "stream": True,
            "cookies": self._cookies,
        }
        response = self.request(**request_params, **self._proxies)

        if response.status_code in [401, 403] or "text/html" in response.headers.get(
            "Content-Type", []
        ):
            # Authentication has likely expired, re-authenticate
            if not self.use_token:
                print("Authorization expired, re-authenticating...", end=" ")
                self.authenticate(verbose=False)
                print("Done!")
                response = self.request(**request_params, **self._proxies)
            else:
                # In case of token authentication, SCRequestException will be raised
                # upon authentication failure
                pass

        if response.status_code not in [200, 201]:
            try:
                response_data = response.json()
            except (JSONDecodeError, simplejson.errors.JSONDecodeError):
                response_data = None
            raise SCRequestException(
                method=method,
                url=url,
                status_code=response.status_code,
                request_data=kw_data_arg,
                response_data=response_data,
            )

        if response.headers.get("Content-Type", None) == "application/json":
            result = response.json()
        else:
            result = response

        return result

    def logout(self, verbose: bool = True) -> None:
        """
        Log out of the server and end the session. All HTTPAdapters are closed and
        cookies and headers are cleared.

        :param verbose: False to suppress output messages
        """
        if not self.logged_in or self.use_token:
            return

        sign_out_url = (
            self.config.base_url[: -len(self.config.api_pattern)] + "/oauth2/sign_out"
        )
        try:
            response = self.request(url=sign_out_url, method="GET", **self._proxies)

            if response.status_code == 200:
                if verbose:
                    print("Logout successful.")
            else:
                raise SCRequestException(
                    method="GET",
                    url=sign_out_url,
                    status_code=response.status_code,
                    request_data={},
                )
        except RequestException:
            if verbose:
                print(
                    f"The {self.__class__.__name__} is closed successfully, but the "
                    f"client was not able to logout from the server."
                )

        super().close()
        self._cookies = {CSRF_COOKIE_NAME: None, PROXY_COOKIE_NAME: None}
        self.cookies.clear()
        self.headers = CaseInsensitiveDict(INITIAL_HEADERS)
        self.logged_in = False

    def _get_product_info_and_set_api_version(self) -> Dict[str, str]:
        """
        Return the product info as retrieved from the SC server.

        This method will also attempt to set the API version correctly, based on the
        retrieved product info.

        :return: Dictionary containing the product info.
        """
        try:
            product_info = self.get_rest_response("product_info", "GET")
        except SCRequestException:
            self.config.api_version = LEGACY_API_VERSION
            product_info = self.get_rest_response("product_info", "GET")
        return product_info

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Log out of the SC server. This method is called when exiting the runtime
        context related to the session, in case the session is used as a context
        manager.
        """
        if self.logged_in:
            self.logout(verbose=False)
        super().__exit__(exc_type, exc_value, traceback)

    def __del__(self):
        """
        Log out of the SC server. This method is called when the session is deleted
        from memory.
        """
        if self.logged_in:
            self.logout(verbose=False)
