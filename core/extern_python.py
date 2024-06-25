# import parser_types
import interpreter_types
from typing import Dict, List
from resolvable import Resolvable

Context = Dict['parser_types.AstIdentifier', 'interpreter_types.Object']


class ExternPython(Resolvable):
    """
    Abstract class for built-in functions that are implemented in Python.
    """

    def resolve(self, context: Context) -> 'interpreter_types.Object':
        return interpreter_types.Object(interpreter_types.DType.EXTERN_PYTHON, self)

    @staticmethod
    def execute(
            context: Context, args: List['interpreter_types.Object']
    ) -> 'interpreter_types.Object':
        """
        This function is called when the built-in function is called.

        Args:
            context: The current context outside of function call.
            args: The **resolved** arguments to the function.

        """
        # NOTE: args are already resolved
        raise NotImplementedError()

    def __str__(self) -> str:
        return "ExternPython"
