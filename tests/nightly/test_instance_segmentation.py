from test_nightly_project import TestNightlyProject


class TestInstanceSegmentation(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for an instance segmentation project
    """
    PROJECT_TYPE = "instance_segmentation"
    __test__ = True
