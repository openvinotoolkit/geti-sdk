from test_nightly_project import TestNightlyProject


class TestRotatedDetection(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a rotated detection project
    """
    PROJECT_TYPE = "rotated_detection"
    __test__ = True
