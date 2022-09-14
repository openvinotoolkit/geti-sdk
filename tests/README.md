This directory contains the test suite for the Intel® Geti™ SDK. To maintain flexibility, we
mostly make use of integration tests rather than unit tests for all but the core
elements of the SDK. In addition, we have a suite of nightly tests that are executed
every night at midnight (Amsterdam time).

# Integration tests
The integration tests for this package makes use of recording of HTTP requests and responses,
relying on the VCR.py package. The tests are located in the `integration` directory.
By default, the tests are run in offline mode, meaning
that no actual Intel® Geti™ server is needed and no real http requests are being made during
testing. All requests are intercepted, and a previously recorded response is returned.
The recorded interactions can be found in [fixtures/cassettes](fixtures/cassettes).

# Nightly tests
The nightly tests are located in the `nightly` directory. They can only be ran in
`ONLINE` mode, meaning that a live Intel® Geti™ server is required to run them against. The
nightly tests need to be run using a `online.ini` file that contains the host name and
login details for the Intel® Geti™ server to run the tests against (see section
[Running the tests](#running-the-tests) below).

# Running the tests
First, install the requirements for the test suite using
`pip install -r requirements/requirements-dev.txt`. Then, run the tests using
`pytest ./tests/integration`, or
(optionally) enable coverage using `pytest --cov=geti_sdk ./tests/integration`.

## Test modes
By default, the integration tests are executed in `OFFLINE` mode. In addition, they
can be run in `ONLINE` or `RECORD` mode. The simplest way to change modes is to
define a custom pytest configuration file for each mode, as explained below:

### Online mode
If you want to run the test suite against a live Intel® Geti™ server (for example to make sure
that the SDK data models are still up to date with the Intel® Geti™ REST contracts), the tests
can be executed in `ONLINE` mode. To do so, define a file `online.ini` in the `tests`
directory. This file should have similar content to the existing `offline.ini`, but
with the Intel® Geti™ server hostname and login details set appropriately:

> ```shell
> [pytest]
> env =
>   TEST_MODE=ONLINE
>   SC_USERNAME=your_username
>   SC_PASSWORD=your_password
>   SC_HOST=https://your_sc_instance.com
> ```

### Record mode
If you have added a new test that makes http requests, all cassettes should be deleted
and re-recorded to maintain consistency across the recorded responses. This can be done
by running the tests in `RECORD` mode. The easiest way to do this is to create a file
`record.ini` with the same contents as the `online.ini` file above, but set
`TEST_MODE=RECORD` instead of `ONLINE`.

> **WARNING**: Running the test suite in `ONLINE` or `RECORD` mode will increase the
> time required for test execution considerably

## Running the tests in a non-default mode
Once you created the custom `online.ini` or `record.ini` configurations, you can run
the tests using `pytest -c online.ini ./integration`. This will execute the tests in
online mode.
