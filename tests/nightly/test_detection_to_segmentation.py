from test_nightly_project import TestNightlyProject


class TestDetectionToSegmentation(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a detection_to_segmentation project
    """
    PROJECT_TYPE = "detection_to_segmentation"
    __test__ = True