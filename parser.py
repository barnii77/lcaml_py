from token import Token, TokenKind
from typing import List


TokenStream = List[Token]


class ParseState:
    """
    Enum for parser state
    """

    ExpectLetOrEnd = 0
    ExpectIdentfier = 1
    ExpectEquals = 2
    ExpectExpression = 3
    ExpectSemicolon = 4


class ParseError(Exception):
    """
    Exception raised when the parser cannot parse the code.
    """

    pass


class AstRelated:
    """
    Abstract parent class for all AST related classes.
    """

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
        remaining_stream = stream[semicolon_idx + 1 :]
        return cls(expression_stream), remaining_stream


class AstAssignment(AstRelated):
    """

    Attributes:
        identifier: identifier to write to
        value: value to assign

    """

    def __init__(self, identifier: AstIdentifier, value: AstExpression):
        self.identifier = identifier
        self.value = value


class Ast(AstRelated):
    """

    Attributes:
        assignments: List of AstAssignment objects with parse function

    """

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
        state = ParseState.ExpectLetOrEnd
        position = 0
        stream_len = len(stream)
        identifier = None
        while position < stream_len:
            token = stream[position]
            if state == ParseState.ExpectLetOrEnd:
                if token.type == TokenKind.LET:
                    state = ParseState.ExpectIdentfier
                    identifier = None
                else:
                    raise ParseError("Expected let or end of file")
            elif state == ParseState.ExpectIdentfier:
                if token.type == TokenKind.IDENTIFIER:
                    identifier = AstIdentifier(token)
                    state = ParseState.ExpectEquals
                else:
                    raise ParseError("Expected identifier")
            elif state == ParseState.ExpectEquals:
                if token.type == TokenKind.EQUALS:
                    state = ParseState.ExpectExpression
                else:
                    raise ParseError("Expected equals sign")
            elif state == ParseState.ExpectExpression:
                expression, stream = AstExpression.from_stream(stream[position:])
                if identifier is None:
                    raise ParseError(
                        "Could not parse out identifier (probably syntax error, maybe interpreter bug)"
                    )
                assignment = AstAssignment(identifier, expression)
                assignments.append(assignment)
                state = ParseState.ExpectSemicolon

            position += 1
        return cls(assignments)


class Parser:
    """

    Attributes:
        stream: TokenStream to parse

    """

    def __init__(self, stream: TokenStream):
        self.stream = stream

    def __call__(self) -> Ast:
        return Ast.from_stream(self.stream)


if __name__ == "__main__":
    code = """
    let x = 10; -- x y z
    let y = 20;
    let z = x + y + zahl;
    """
    from lexer import Lexer, Syntax

    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    ast = Ast.from_stream(tokens)
    print(ast)
