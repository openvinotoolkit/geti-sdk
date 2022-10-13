from typing import Any, Callable, Dict, List, Union

import pytest

from geti_sdk import Geti
from geti_sdk.http_session import GetiSession, ServerCredentialConfig


@pytest.fixture
def fxt_mocked_server_config():
    yield ServerCredentialConfig(
        host="dummy_host", username="dummy_user", password="dummy_password"
    )


@pytest.fixture
def fxt_mocked_session_factory(
    mocker, fxt_mocked_server_config: ServerCredentialConfig
) -> Callable[[Any], GetiSession]:
    def _mocked_session_factory(return_value: Union[List, Dict]) -> GetiSession:
        mocker.patch("geti_sdk.http_session.geti_session.GetiSession.authenticate")
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession.get_rest_response",
            return_value=return_value,
        )
        mocker.patch(
            "geti_sdk.http_session.geti_session.GetiSession._get_product_info_and_set_api_version",
            return_value={
                "build-version": "1.0.0-release-20221005164936",
                "product-version": "1.0.0",
                "smtp-defined": "True",
            },
        )
        return GetiSession(fxt_mocked_server_config)

    yield _mocked_session_factory


@pytest.fixture
def fxt_mocked_geti(mocker):
    mocker.patch("geti_sdk.http_session.geti_session.GetiSession.authenticate")
    mocker.patch("geti_sdk.geti.get_default_workspace_id", return_value=1)
    mocker.patch(
        "geti_sdk.http_session.geti_session.GetiSession._get_product_info_and_set_api_version",
        return_value={
            "build-version": "1.0.0-release-20221005164936",
            "product-version": "1.0.0",
            "smtp-defined": "True",
        },
    )

    geti = Geti(host="dummy_host", password="dummy_password", username="dummy_username")

    yield geti
