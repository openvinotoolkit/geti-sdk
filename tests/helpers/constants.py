import os

BASE_TEST_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY_HOST = "dummy_host"
CASSETTE_PATH = os.path.join(BASE_TEST_PATH, 'fixtures', 'cassettes')
