from typing import Dict, Optional, Union

import requests
import urllib3

from dataclasses import dataclass

from requests import Response

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_PATTERN = "/api/v1.0/"
CSRF_COOKIE_NAME = "_oauth2_proxy_csrf"
PROXY_COOKIE_NAME = "_oauth2_proxy"


@dataclass
class ServerConfig:
    """
    Configuration for requests sessions, with host, username and password.
    """

    host: str
    username: str
    password: str

    @property
    def base_url(self) -> str:
        """
        Returns the base UR for accessing the server
        """
        return f"{self.host}{API_PATTERN}"


class SCSession(requests.Session):
    def __init__(self, serverconfig: ServerConfig):
        """
        Wrapper for requests.session that sets the correct headers and cookies.

        :param serverconfig: ServerConfig with the parameters for host, username,
            password
        """
        super().__init__()
        self.headers.update({"Connection": "keep-alive"})
        self.config = serverconfig
        self.verify = False
        self.allow_redirects = False
        self.token = None
        self._cookies: Dict[str, Optional[str]] = {
            CSRF_COOKIE_NAME: None, PROXY_COOKIE_NAME: None
        }

        # Authentication is only used for https servers.
        if "https" in serverconfig.host:
            self.authenticate()
        elif "http:" in serverconfig.host:
            # http hosts should include port number in REST request
            if serverconfig.host.count(":") != 2:
                raise ValueError(
                    f"Please add a port number to the hostname, for "
                    f"example: http://10.0.0.1:5001"
                )
        else:
            raise ValueError(
                f"Please use a full hostname, including the protocol for http "
                f"servers. For example: https://10.0.0.1"
            )

    def _follow_login_redirects(self, response: Response) -> str:
        """
        Recursively follow redirects in the initial login request. Updates the
        session._cookies with the cookie and the login uri

        :param response: REST response to follow redirects for
        :return: url to the redirected location
        """
        if response.status_code in [302, 303]:
            redirect_url = response.next.url
            redirected = self.get(redirect_url, allow_redirects=False)
            proxy_csrf = redirected.cookies.get(CSRF_COOKIE_NAME, None)
            if proxy_csrf:
                self._cookies[CSRF_COOKIE_NAME] = proxy_csrf
            return self._follow_login_redirects(redirected)
        else:
            return response.url

    def _get_initial_login_url(self) -> str:
        """
        Retrieves the initial login url by making a request to the login page, and
        following the redirects.
        :return: current state dictionary of the session
        """
        response = self.get(f"{self.config.host}/user/login", allow_redirects=False)
        login_page_url = self._follow_login_redirects(response)
        return login_page_url

    def authenticate(self):
        """
        Get a new authentication cookie from the server
        """
        login_path = self._get_initial_login_url()
        self.headers.clear()
        self.headers.update({'Content-Type': 'application/x-www-form-urlencoded'})
        print(f"Authenticating on host {self.config.host}...")
        response = self.post(
            url=login_path,
            data={"login": self.config.username, "password": self.config.password},
            cookies={CSRF_COOKIE_NAME: self._cookies[CSRF_COOKIE_NAME]},
            headers={"Cookie": self._cookies[CSRF_COOKIE_NAME]},
            allow_redirects=True,
            )
        try:
            previous_response = response.history[-1]
        except IndexError:
            raise ValueError(
                "The cluster responded to the request, but authentication failed. "
                "Please verify that you have provided correct credentials."
            )
        cookie = {
            PROXY_COOKIE_NAME: previous_response.cookies.get(PROXY_COOKIE_NAME)
        }
        self._cookies.update(cookie)
        print("Authentication cookie received.")

    def get_rest_response(
            self, url: str, method: str, contenttype: str = "json", data=None
    ) -> Union[Response, dict]:
        """
        Returns the REST response from a request to `url` with `method`

        :param url: the REST url without the hostname and api pattern
        :param method: 'GET', 'POST', 'PUT', 'DELETE'
        :param contenttype: currently either 'json', 'jpeg' or '', defaults to "json"
        :param data: the data to send in a post request, as json
        """
        if url.startswith(API_PATTERN):
            url = url[len(API_PATTERN):]

        if contenttype == "json":
            self.headers.update({"Content-Type": "application/json"})
        elif contenttype == "jpeg":
            self.headers.update({"Content-Type": "image/jpeg"})
        elif contenttype == "multipart":
            self.headers.pop("Content-Type", None)
        elif contenttype == "":
            self.headers.pop("Content-Type", None)

        requesturl = f"{self.config.base_url}{url}"
        if contenttype == "json":
            kw_data_arg = {"json": data}
        else:
            kw_data_arg = {"files": data}
        response = self.request(
            method, requesturl, **kw_data_arg, stream=True, cookies=self._cookies
        )
        if response.status_code in [401, 403]:
            # Authentication has likely expired, re-authenticate
            self.authenticate()
            response = self.request(
                method, requesturl, **kw_data_arg, stream=True, cookies=self._cookies
            )

        if response.status_code not in [200, 201]:
            raise ValueError(url, response.json())

        if contenttype == "json":
            return response.json()
        else:
            return response
