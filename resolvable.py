from typing import Dict
from parser_types import AstIdentifier


class Object:
    """
    this is a dummy class used for type hints replacing the object class from interpreter_types
    """
    pass


Context = Dict[AstIdentifier, Object]


class Resolvable:
    def resolve(self, context: Context) -> Object:
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()
