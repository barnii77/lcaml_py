from typing import Any


class DType:
    """
    This class represents a data type enum.
    """

    INT = 0
    FLOAT = 1
    STRING = 2
    BOOL = 3
    FUNCTION = 4


class Object:
    """
    This class represents an object in the interpreter.
    """

    def __init__(self, type: int, value: Any):
        self.type = type
        self.value = value

    def __str__(self):
        return "self.type(" + self.value + ")"

    def __repr__(self):
        return "DType(" + self.value + ")"

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return self.type == other.type and self.value == other.value
