from test_nightly_project import TestNightlyProject


class TestClassification(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a classification project
    """
    PROJECT_TYPE = "classification"
    __test__ = True
