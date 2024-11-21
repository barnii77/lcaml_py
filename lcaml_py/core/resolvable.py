from typing import Dict, Any


Context = Dict[str, "Object"]


class Resolvable:
    def resolve(self, context: Context) -> Any:
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()

    def to_python(self) -> tuple[str, str, str]:
        """
        This function returns the python code that is used to resolve the expression.
        """
        raise NotImplementedError()
