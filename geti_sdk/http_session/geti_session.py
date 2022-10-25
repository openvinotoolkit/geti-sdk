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
import logging
import time
import warnings
from json import JSONDecodeError
from typing import Any, Dict, Optional, Union

import requests
import simplejson
import urllib3
from requests import Response
from requests.exceptions import RequestException
from requests.structures import CaseInsensitiveDict
from urllib3.exceptions import InsecureRequestWarning

from geti_sdk.platform_versions import GetiVersion

from .exception import GetiRequestException
from .server_config import LEGACY_API_VERSION, ServerCredentialConfig, ServerTokenConfig

CSRF_COOKIE_NAME = "_oauth2_proxy_csrf"
PROXY_COOKIE_NAME = "_oauth2_proxy"

INITIAL_HEADERS = {"Connection": "keep-alive", "Upgrade-Insecure-Requests": "1"}
SUCCESS_STATUS_CODES = [200, 201]


class GetiSession(requests.Session):
    """
    Wrapper for requests.session that sets the correct headers and cookies, and
    handles authentication and authorization.

    :param server_config: Server configuration holding the hostname (or ip address) of
        the Intel® Geti™ server, as well as the details required for authentication
        (either username and password or personal access token)
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

        # Configure certificate verification
        if not server_config.has_valid_certificate:
            warnings.warn(
                "You have disabled TLS certificate validation, HTTPS requests made to "
                "the Intel® Geti™ server may be compromised. For optimal security, "
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
            self.use_token = True
            access_token = self._acquire_access_token()
            self.headers.update({"Authorization": f"Bearer {access_token}"})
            self.headers.pop("Connection")

        # Get server version
        self._product_info = self._get_product_info_and_set_api_version()

    @property
    def version(self) -> GetiVersion:
        """
        Return the version of the Intel® Geti™ platform that is running on the server.

        :return: Version object holding the Intel® Geti™ version number
        """
        if "build-version" in self._product_info.keys():
            version_string = self._product_info.get("build-version")
        else:
            version_string = self._product_info.get("product-version")
        return GetiVersion(version_string)

    def _acquire_access_token(self) -> str:
        """
        Request an access token from the server, in exchange for the
        PersonalAccessToken.
        """
        try:
            response = self.get_rest_response(
                url="service_accounts/access_token",
                method="POST",
                data={"service_id": self.config.token},
                contenttype="json",
            )
        except GetiRequestException as error:
            if error.status_code == 401:
                raise ValueError(
                    "Token authorization failed. Please make sure to provide a valid "
                    "Personal Access Token."
                )
            raise
        logging.info(f"Personal access token validated on host {self.config.host}")
        return response.get("access_token")

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
        response = self.get(self.config.host, allow_redirects=False, **self._proxies)
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
                    "Connection to the Intel® Geti™ server failed, please make sure to "
                    "update the user login information for the Intel® Geti™ instance."
                ) from error
            raise ValueError(
                f"Connection to the Intel® Geti™ server at host '{self.config.host}' "
                f"failed, please provide a valid cluster hostname or ip address as"
                f" well as valid login details."
            ) from error
        self.headers.clear()
        self.headers.update({"Content-Type": "application/x-www-form-urlencoded"})
        if verbose:
            logging.info(f"Authenticating on host {self.config.host}...")
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
            logging.info("Authentication successful. Cookie received.")
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
        }

        if not self.use_token:
            request_params.update({"cookies": self._cookies})

        response = self.request(**request_params, **self._proxies)

        if (
            response.status_code not in SUCCESS_STATUS_CODES
            or "text/html" in response.headers.get("Content-Type", [])
        ):
            response = self._handle_error_response(
                response=response,
                request_params=request_params,
                request_data=kw_data_arg,
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
        requires_closing = True
        requires_sign_out = True

        if not self.logged_in or self.use_token:
            requires_closing = False
            requires_sign_out = False

        if requires_sign_out:
            sign_out_url = (
                self.config.base_url[: -len(self.config.api_pattern)]
                + "/oauth2/sign_out"
            )
            try:
                response = self.request(url=sign_out_url, method="GET", **self._proxies)

                if response.status_code in SUCCESS_STATUS_CODES:
                    if verbose:
                        logging.info("Logout successful.")
                else:
                    raise GetiRequestException(
                        method="GET",
                        url=sign_out_url,
                        status_code=response.status_code,
                        request_data={},
                    )
            except (RequestException, AttributeError):
                if verbose:
                    logging.info(
                        f"The {self.__class__.__name__} is closed successfully, but "
                        f"the Geti instance was not able to logout from the server."
                    )
            except TypeError:
                # The session is already closed
                requires_closing = False

        self._cookies = {CSRF_COOKIE_NAME: None, PROXY_COOKIE_NAME: None}
        self.cookies.clear()
        self.headers = CaseInsensitiveDict(INITIAL_HEADERS)
        self.logged_in = False
        if requires_closing:
            try:
                super().close()
            except TypeError as error:
                # Sometimes a TypeError is raised if logout is called during garbage
                # collection, however we can safely ignore this since the session will
                # be deleted in that case anyway.
                logging.debug(f"{error} encountered during GetiSession closure.")

    def _get_product_info_and_set_api_version(self) -> Dict[str, str]:
        """
        Return the product info as retrieved from the Intel® Geti™ server.

        This method will also attempt to set the API version correctly, based on the
        retrieved product info.

        :return: Dictionary containing the product info.
        """
        try:
            product_info = self.get_rest_response("product_info", "GET")
        except GetiRequestException:
            self.config.api_version = LEGACY_API_VERSION
            product_info = self.get_rest_response("product_info", "GET")
        return product_info

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Log out of the Intel® Geti™ server. This method is called when exiting the
        runtime context related to the session, in case the session is used as a context
        manager.
        """
        if self.logged_in:
            try:
                self.logout(verbose=False)
            except Exception as exc:
                logging.debug(f"{exc} encountered while exiting GetiSession context.")
        super().__exit__(exc_type, exc_value, traceback)

    def __del__(self):
        """
        Log out of the Intel® Geti™ server. This method is called when the session is
        deleted from memory.
        """
        if self.logged_in:
            try:
                self.logout(verbose=False)
            except Exception as exc:
                logging.debug(f"{exc} encountered during GetiSession deletion.")

    def _handle_error_response(
        self,
        response: Response,
        request_params: Dict[str, Any],
        request_data: Dict[str, Any],
    ) -> Response:
        """
        Handle error responses from the server.

        :param response: The Response object received from the server
        :param request_params: Dictionary containing the original parameters of the
            request
        :raises: GetiRequestException in case the error cannot be handled
        :return: Response object resulting from the request
        """
        if response.status_code in [200, 401, 403]:
            # Authentication has likely expired, re-authenticate
            logging.info("Authentication may have expired, re-authenticating...")
            if not self.use_token:
                self.authenticate(verbose=False)
                logging.info("Authentication complete.")

            else:
                access_token = self._acquire_access_token()
                logging.info("New bearer token obtained.")
                self.headers.update({"Authorization": f"Bearer {access_token}"})

            response = self.request(**request_params, **self._proxies)
            if response.status_code in SUCCESS_STATUS_CODES:
                return response

        elif response.status_code == 503:
            # In case of Service Unavailable, wait some time and try again. If it
            # still doesn't work, raise exception
            time.sleep(1)
            response = self.request(**request_params, **self._proxies)

            if response.status_code in SUCCESS_STATUS_CODES:
                return response

        try:
            response_data = response.json()
        except (JSONDecodeError, simplejson.errors.JSONDecodeError):
            response_data = None

        raise GetiRequestException(
            method=request_params["method"],
            url=request_params["url"],
            status_code=response.status_code,
            request_data=request_data,
            response_data=response_data,
        )
