# v0.2.2 Intel® Geti™ SDK (04-10-2022)
## What's Changed
* Add coverage report to pre-merge and nightly test artifacts by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/47
* Correctly set permissions on extracted files for anomaly dataset by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/51
* Update pytest-cov requirement from ==3.0.* to ==4.0.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/50
* Update pillow requirement from ==9.1.* to ==9.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/48
* Workflow update: Run nightly and integration tests in one step and get coverage by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/53

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.1...v0.2.2

# v0.2.1 Intel® Geti™ SDK (30-09-2022)
## What's Changed
* Replace SC references in docstrings by Geti by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/33
* Change package name from `geti_sdk` to `geti-sdk`. Import names are unchanged by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/34
* Update vcrpy requirement from ==4.1.* to ==4.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/35
* Bump datumaro from 0.3 to 0.3.1 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/37
* Bump openvino from 2022.1.0 to 2022.2.0 in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/38
* Update requests requirement from ==2.26.* to ==2.28.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/39
* Handle exceptions in data deserialization by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/40
* Fix image path in notebook 008 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/41
* Use personal access token instead of credential authentication by default by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/42
* Add image showing the personal access token menu to README by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/43
* Add nightly tests for `demos` module by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/44
* Add screenshot of jupyter lab landing page to README by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/45

## New Contributors
* @dependabot made their first contribution in https://github.com/openvinotoolkit/geti-sdk/pull/35

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.0...v0.2.1

# v0.2.0 Intel® Geti™ SDK (27-09-2022)

This is the first official release of the Intel® Geti™ Software Development Kit (SDK).

The purpose of this SDK is twofold:

1. Provide an easy-to-use interface to the [Intel® Geti™ platform](www.geti.intel.com), to manipulate
Intel® Geti™ projects and other entities or automate tasks on the platform. All
of this from a Python script or Jupyter notebook.


2. Provide an API to deploy and run models trained on the Intel® Geti™ server on your local
machine. The SDK Deployment module provides a straightforward
route to create a deployment for your Intel® Geti™ project, save it to a local disk and run
it offline.

This SDK includes various example scripts and Jupyter notebooks which illustrate a
range of use cases for the SDK. Make sure to check them out if you're getting
started!
