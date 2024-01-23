import lexer
from abc import abstractmethod
from lexer import Token, TokenKind
from typing import List


TokenStream = List[lexer.Token]


class ParserError(Exception):
    """
    Exception raised when the parser cannot parse the code.
    """

    pass


class AstRelated:
    """
    Abstract parent class for all AST related classes.
    """

    @abstractmethod
    @classmethod
    def from_stream(cls, stream: TokenStream):
        raise NotImplementedError()


class AstIdentifier(AstRelated):
    """

    Attributes:
        name: name of identifier

    """

    def __init__(self, name_token: Token):
        if name_token.type != TokenKind.IDENTIFIER:
            raise ValueError("Token must be an identifier")
        self.name = name_token.value


class AstExpression(AstRelated):
    """

    Attributes:
        expression: TokenStream of expression

    """

    def __init__(self, expression: TokenStream):
        self.expression = expression

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
        remaining_stream = stream[semicolon_idx + 1:]
        return cls(expression_stream), remaining_stream


class AstAssignment(AstRelated):
    def __init__(self, identifier: AstIdentifier, value: AstExpression):
        self.identifier = identifier
        self.value = value


class Ast(AstRelated):
    def __init__(self, assignments: List[AstAssignment]):
        self.assignments = assignments

    @classmethod
    def from_stream(cls, stream: TokenStream):
        """

        Args:
            stream: TokenStream to parse

        Returns:
            Ast: Ast object built from tokenstream

        Raises:
            ParseError: _description_

        """
        assignments = []
        while stream:
            identifier = AstIdentifier(stream[0])
            expression, stream = AstExpression.from_stream(stream[2:])
            assignments.append(AstAssignment(identifier, expression))
        return cls(assignments)


class Parser:
    def __init__(self, tokens: TokenStream):
        self.tokens = tokens

    def parse(self) -> Ast:
        pass
