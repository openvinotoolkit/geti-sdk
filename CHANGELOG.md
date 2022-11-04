# v1.0.0rc1 Intel® Geti™ SDK (04-11-2022)
## What's Changed
* Add a re-authentication mechanism when using token authentication by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/72
* Update pytest requirement from ==7.1.* to ==7.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/73
* Update pillow requirement from ==9.2.* to ==9.3.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/74
* Update pytest-html requirement from ==3.1.* to ==3.2.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/75
* Catch value error when invalid datetime string is converted by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/76
* Update nightly test workflow to include tests against Geti `develop` branch by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/78
* Enable SSL certificate validation by default by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/77
* Remove disallowed fields from project before POSTing by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/79
* Ignore false positive bandit detections by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/80
* Update numpy, ipython, jupyterlab versions by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/81
* Fix and unify folder naming for project download by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/82


## Breaking Changes
* The `Geti` class now has SSL certificate validation enabled by default. This will result in an error when connecting to a server with a certificate that can't be validated. Note that it is still possible to disable certificate validation by passing `verify_certificate = False` when initializing the `Geti` instance. Please note that disabling certificate validation is unsafe and should only be considered in a secure network environment.


**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.4...v1.0.0rc1

# v0.2.4 Intel® Geti™ SDK (25-10-2022)
## What's Changed
* Auto detect normalized annotation files for GetiAnnotationReader by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/63
* Fix version detection mechanism and add tests for GetiVersion by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/64
* Minor changes for backward compatibility with SCv1.1 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/67
* Enable proxies in ONLINE test mode by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/66
* Fix proxy config in tests for online mode by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/68
* Updated Attrs Classes and Fields by @HiteshManglani123 in https://github.com/openvinotoolkit/geti-sdk/pull/65
* Validate media filename upon download by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/69
* Update pytest-env requirement from ==0.6.* to ==0.8.* in /requirements by @dependabot in https://github.com/openvinotoolkit/geti-sdk/pull/70
* Fix anomaly classification deployment by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/71

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.3...v0.2.4

# v0.2.3 Intel® Geti™ SDK (06-10-2022)
## What's Changed
* Remove VCR from nightly test for demos by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/54
* Improve nightly tests for `demos` module by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/55
* Update sc_annotation_reader by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/56
* Add version to optimized model by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/57
* Update SDK platform version parsing mechanism by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/58
* Add nightly tests against Geti-MVP by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/59
* Handle failed training jobs in notebook 007 by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/60
* Update example script to store prediction results to file by @ljcornel in https://github.com/openvinotoolkit/geti-sdk/pull/61

**Full Changelog**: https://github.com/openvinotoolkit/geti-sdk/compare/v0.2.2...v0.2.3

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
