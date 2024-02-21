from typing import Dict, Any
from parser_types import AstIdentifier


Context = Dict[AstIdentifier, Any]


class Resolvable:
    def resolve(self, context: Context) -> Any:
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()
