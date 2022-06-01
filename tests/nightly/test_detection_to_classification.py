from test_nightly_project import TestNightlyProject


class TestDetectionToClassification(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a detection_to_classification project
    """
    PROJECT_TYPE = "detection_to_classification"
    __test__ = True
