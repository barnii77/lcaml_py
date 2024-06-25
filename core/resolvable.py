from typing import Dict, Any


Context = Dict['AstIdentifier', Any]


class Resolvable:
    def resolve(self, context: Context) -> Any:
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()
