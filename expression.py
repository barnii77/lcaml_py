from ast_related import AstRelated
from token_type import Token, TokenKind
from typing import List


TokenStream = List[Token]


class AstOperation(AstRelated):
    def __init__(self, left: Token, operation: Token, right: Token):
        self.left = left
        self.operation = operation
        self.right = right

    def __str__(self):
        return "AstOperation(" + str(self.left) + str(self.operation) + str(self.right) + ")"

    def __call__(self, left, right):
        """
        This function evaluates the operation for values of left and right
        """
        if self.operation.type == TokenKind.PLUS:
            return left + right
        elif self.operation.type == TokenKind.MINUS:
            return left - right



class AstExpression(AstRelated):
    """

    Attributes:
        expression: TokenStream of expression

    """

    def __init__(self, expression: TokenStream):
        self.expression = expression

    def __str__(self):
        return "AstExpression(" + str(self.expression) + ")"

    @classmethod
    def from_stream(cls, stream: TokenStream):
        """

        Args:
            stream: TokenStream to parse

        Raises:
            ValueError: Semicolon not found (no end of expression found)

        Returns:
            AstExpression: AstExpression object built from tokenstream
            Stream: Remaining tokenstream
        """
        # FIXME will not work once functions are supported
        # TODO support parentheses
        semicolon = Token(TokenKind.SEMICOLON, ";")
        if semicolon not in stream:
            raise ValueError("Expression must end with a semicolon")
        semicolon_idx = stream.index(semicolon)
        expression_stream = stream[:semicolon_idx]
        remaining_stream = stream[
            semicolon_idx:
        ]  # leave semicolon in stream so the parser can be sure syntax is valid
        return cls(expression_stream), remaining_stream
