from typing import Union
from token_type import Token, TokenKind
from typing import List


TokenStream = List[Token]


class ParseState:
    """
    Enum for parser state
    """

    ExpectStatementOrCommentOrEnd = 0
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

    def __repr__(self):
        return self.__str__()


class AstIdentifier(AstRelated):
    """

    Attributes:
        name: name of identifier

    """

    def __init__(self, name_token: Token):
        if name_token.type != TokenKind.IDENTIFIER:
            raise ValueError("Token must be an identifier")
        self.name = name_token.value

    def __str__(self):
        return "AstIdentifier(" + self.name + ")"


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
        remaining_stream = stream[semicolon_idx:]  # leave semicolon in stream so the parser can be sure syntax is valid
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

    def __str__(self):
        return "AstAssignment(" + str(self.identifier) + ", " + str(self.value) + ")"


class AstReturn(AstRelated):
    """

    Attributes:
        value: value to return

    """

    def __init__(self, value: AstExpression):
        self.value = value

    def __str__(self):
        return "AstReturn(" + str(self.value) + ")"


class AstStatementType:
    ASSIGNMENT = "assignment"
    RETURN = "return"


class AstStatement(AstRelated):
    """

    Attributes:
        type: type of statement (AstStatementType)
        value: value to assign

    """

    def __init__(self, type: str, value: Union[AstAssignment, AstReturn]):
        self.type = type
        self.value = value

    def __str__(self):
        return "AstStatement(" + self.type + ", " + str(self.value) + ")"


class Ast(AstRelated):
    """

    Attributes:
        assignments: List of AstAssignment objects with parse function

    """

    def __init__(self, statements: List[AstStatement]):
        self.statements = statements

    def __str__(self):
        return "Ast(" + str(self.statements) + ")"

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
        statements = []
        state = ParseState.ExpectStatementOrCommentOrEnd
        identifier = None
        while stream:
            next_token = stream.pop(0)

            if state == ParseState.ExpectStatementOrCommentOrEnd:
                if next_token.type == TokenKind.COMMENT:
                    continue
                elif next_token.type == TokenKind.LET:
                    state = ParseState.ExpectIdentfier
                    identifier = None
                else:
                    raise ParseError("Expected let or end of file")

            elif state == ParseState.ExpectIdentfier:
                if next_token.type == TokenKind.IDENTIFIER:
                    identifier = AstIdentifier(next_token)
                    state = ParseState.ExpectEquals
                else:
                    raise ParseError("Expected identifier")

            elif state == ParseState.ExpectEquals:
                if next_token.type == TokenKind.EQUALS:
                    state = ParseState.ExpectExpression
                else:
                    raise ParseError("Expected equals sign")

            elif state == ParseState.ExpectExpression:
                expression, stream = AstExpression.from_stream([next_token] + stream)
                if identifier is None:
                    raise ParseError(
                        "Could not parse out identifier (probably syntax error, maybe interpreter bug)"
                    )
                assignment = AstAssignment(identifier, expression)
                statement = AstStatement(AstStatementType.ASSIGNMENT, assignment)
                statements.append(statement)
                state = ParseState.ExpectSemicolon

            elif state == ParseState.ExpectSemicolon:
                if next_token.type == TokenKind.SEMICOLON:
                    state = ParseState.ExpectStatementOrCommentOrEnd
                else:
                    raise ParseError("Expected semicolon")

            else:
                raise ParseError("Invalid state reached, please report bug to lcaml maintainers")
        return cls(statements)


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
    from lcaml_lexer import Lexer, Syntax

    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    parser = Parser(tokens)
    ast = parser()
    for statement in ast.statements:
        print(statement)
