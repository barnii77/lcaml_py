from typing import List
from core.token_type import Token

TokenStream = List[Token]


class AstRelated:
    """
    Abstract parent class for all AST related classes.
    """

    @classmethod
    def from_stream(cls, stream: TokenStream):
        raise NotImplementedError()

    def __repr__(self):
        return self.__str__()

    def to_python(self) -> tuple[str, str, str]:
        """
        Function that returns 3 strings that are used for transpiling lcaml to python code.

        Returns:
            tuple[str, str, str]: pre-statement-insertion, expression, post-statement-insertion
        """
        raise NotImplementedError()
