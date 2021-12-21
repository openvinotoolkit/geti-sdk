from typing import Optional, ClassVar, List

import attr


@attr.s(auto_attribs=True)
class LabelSource:
    """
    Class representing a source for a ScoredLabel in SC
    """
    id: str
    type: str


@attr.s(auto_attribs=True)
class Label:
    """
    Class representing a Label in SC

    :var name: Name of the label
    :var id: Unique database ID of the label
    :var color: Color (in hex representation) of the label
    :var group: Name of the label group to which the label belongs
    :var is_empty: True if the label represents an empty label, False otherwise
    :var parent_id: Optional name of the parent label, if any
    """

    _identifier_fields: ClassVar[List[str]] = ["id", "hotkey"]

    name: str
    color: str
    group: str
    is_empty: bool
    hotkey: str = ""
    id: Optional[str] = None
    parent_id: Optional[str] = None


@attr.s(auto_attribs=True)
class ScoredLabel:
    """
    Class representing a Label with a probability in SC

    :var name: Name of the label
    :var id: Unique database ID of the label
    :var color: Color (in hex representation) of the label
    :var probability:
    :var source:
    """
    _identifier_fields: ClassVar[List[str]] = ["id"]

    probability: float
    name: Optional[str] = None
    color: Optional[str] = None
    id: Optional[str] = None
    source: Optional[LabelSource] = None
