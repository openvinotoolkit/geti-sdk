# SC SDK test suite
## Introduction
This directory contains the test suite for the SC REST SDK. To maintain flexibility, we 
mostly make use of integration tests rather than unit tests for all but the core 
elements of the SDK. 

The test suite for this package makes use of recording of HTTP requests and responses, 
relying on the VCR.py package. By default, the tests are run in offline mode, meaning 
that no actual SC server is needed and no real http requests are being made during 
testing. All requests are intercepted, and a previously recorded response is returned. 
The recorded interactions can be found in [fixtures/cassettes](fixtures/cassettes).

## Running the tests
First, install the requirements for the test suite using 
`pip install -r requirements.txt`. Then, run the tests using `pytest ./tests`, or 
(optionally) enable coverage using `pytest --cov=sc_api_tools ./tests`.

### Test modes
By default, tests are executed in `OFFLINE` mode. The following other modes are available:

##### Online mode
If you want to run the test suite against a real SC server (for example to make sure 
that the SDK data models are still up to date with the SC REST contracts), make sure to set
the server configuration in [conftest.py](./conftest.py) properly: This file contains variables 
defining the server configuration (`HOST`, `USERNAME` and `PASSWORD`). In addition, 
the `TEST_MODE` variable should be set to `TestMode.ONLINE` to allow making actual 
http requests to the server.

##### Record mode
If you have added a new test that makes http requests, all cassettes should be deleted 
and re-recorded to maintain consistency across the recorded responses. This can be done 
by deleting the full [cassettes](fixtures/cassettes) folder and running the test suite 
in `RECORD` mode by setting `TEST_MODE = TestMode.RECORD`. 

> **WARNING**: Running the test suite in `ONLINE` or `RECORD` mode will increase the 
> time required for test execution considerably
