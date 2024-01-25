from typing import List
from token_type import Token

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
