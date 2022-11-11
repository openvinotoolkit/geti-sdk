import pytest

from tests.helpers.constants import DUMMY_HOST, DUMMY_PASSWORD, DUMMY_TOKEN, DUMMY_USER


@pytest.fixture()
def fxt_server_credential_config_parameters():
    yield (
        {
            "host": DUMMY_HOST,
            "username": DUMMY_USER,
            "password": DUMMY_PASSWORD,
            "has_valid_certificate": False,
            "proxies": {"https": "http://dummy_proxy.com"},
        }
    )


@pytest.fixture()
def fxt_server_token_config_parameters():
    yield (
        {
            "host": DUMMY_HOST,
            "token": DUMMY_TOKEN,
            "has_valid_certificate": True,
        }
    )
