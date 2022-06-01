from test_nightly_project import TestNightlyProject


class TestSegmentation(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a segmentation project
    """
    PROJECT_TYPE = "segmentation"
    __test__ = True
