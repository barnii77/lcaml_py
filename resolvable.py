from typing import Dict
from interpreter_types import Object
from parser_types import AstIdentifier


Context = Dict[AstIdentifier, Object]


class Resolvable:
    def resolve(self, context: Context) -> Object:
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()
