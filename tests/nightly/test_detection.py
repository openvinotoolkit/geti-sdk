from test_nightly_project import TestNightlyProject


class TestDetection(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a detection project
    """
    PROJECT_TYPE = "detection"
    __test__ = True
