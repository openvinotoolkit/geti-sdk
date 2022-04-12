from enum import Enum


class TestMode(Enum):
    """
    This Enum represents the different modes available for running the tests. The
    available modes are:
        - ONLINE:  The tests are run against an actual SC server instance. Real http
                     requests are being made but no cassettes are recorded
                     Use this mode to verify that the SDK data models are still up to
                     date with the current SC REST contracts
        - OFFLINE: The tests are run using the recorded requests and responses. All
                     http requests are intercepted an no actual requests will be made.
                     Use this mode in a CI environment, or for fast testing of the SDK
                     logic
        - RECORD:  The tests are run against an actual SC server instance. HTTP requests
                     are being made and all requests and responses are recorded to a new
                     set of cassettes for the SDK test suite. The old cassettes for the
                     tests are deleted.
    """
    ONLINE = 'online'
    OFFLINE = 'offline'
    RECORD = 'record'
