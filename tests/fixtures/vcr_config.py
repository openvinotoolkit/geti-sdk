""" Configuration fixtures for VCR """
import os
import pytest

from vcr import VCR


@pytest.fixture(scope='session')
def vcr_cassette_dir(base_test_path) -> str:
    return os.path.join(base_test_path, 'fixtures', 'cassettes')


@pytest.fixture(scope='session')
def vcr_config(vcr_record_config):
    vcr_config_dict = {
        "filter_headers": ["authorization"],
        "ignore_localhost": True,
    }
    vcr_config_dict.update(vcr_record_config)
    return vcr_config_dict


@pytest.fixture(scope='session')
def fxt_vcr(vcr_config, vcr_cassette_dir):
    yield VCR(**vcr_config, cassette_library_dir=vcr_cassette_dir)
