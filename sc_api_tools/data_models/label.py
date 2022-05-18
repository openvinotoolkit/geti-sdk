import copy

from typing import Optional, ClassVar, List, Tuple

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

    @property
    def color_tuple(self) -> Tuple[int, int, int]:
        """
        Returns the color of the label as an RGB tuple

        :return:
        """
        hex_color_str = copy.deepcopy(self.color).strip('#')
        return tuple(int(hex_color_str[i:i+2], 16) for i in (0, 2, 4))

    @classmethod
    def from_label(cls, label: Label, probability: float) -> 'ScoredLabel':
        """
        Creates a ScoredLabel instance from an input Label and probability score

        :param label: Label to convert to ScoredLabel
        :param probability: probability score for the label
        :return: ScoredLabel instance corresponding to `label` and `probability`
        """
        return ScoredLabel(
            name=label.name,
            probability=probability,
            color=label.color,
            id=label.id
        )
