from typing import Optional

import attr


@attr.define()
class Performance:
    """
    Class holding the performance metrics for a project or model in SC

    :var score: Overall score of the project or model
    :var local_score: Accuracy of the model or project with respect to object
        localization
    :var global_score: Accuracy of the model or project with respect to global
        classification of the full image
    """
    score: Optional[float] = None
    local_score: Optional[float] = None
    global_score: Optional[float] = None
