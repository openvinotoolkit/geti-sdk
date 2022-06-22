This package contains tools to interact with a Sonoma Creek cluster via the SonomaCreek REST API. It provides functionality for:

- Project creation from datasets on disk
- Project downloading (images, videos, configuration, annotations, predictions and models)
- Project creation and upload from a previous download
- Deploying a project for local inference with OpenVINO

This initial pre-release is not meant for distribution.

## What's Changed
* Video support by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/1
* Openvino inference support for SC detection and segmentation models by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/2
* Nous2sc merge by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/5
* fix missing modules by @pskindel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/8
* Add a basic test suite with integration tests by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/9
* Sphinx documentation by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/11
* Add sphinx build directory by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/12
* Add dependency to OTE SDK, update deployment to use OTE model wrappers by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/14
* Docs action by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/13
* Add nightly tests  by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/15
* Converted shapes to pixel coordinates instead of normalized coordinates by @ljcornel in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/16

## New Contributors
* @ljcornel made their first contribution in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/1
* @pskindel made their first contribution in https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/pull/8
* @olkham contributed the `nous2sc` module, supporting migration of NOUS projects to SonomaCreek

**Full Changelog**: https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/commits/v0.0.1
