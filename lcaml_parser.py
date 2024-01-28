from typing import Union, List
from token_type import Token, TokenKind
from ast_related import AstRelated
from parser_types import AstIdentifier, AstStatementType, AstAssignment, AstReturn
from expression import Expression


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


class AstStatement(AstRelated):
    """

    Attributes:
        type: type of statement (AstStatementType)
        value: value to assign

    """

    def __init__(self, type: int, value: Union[AstAssignment, AstReturn]):
        self.type = type
        self.value = value

    def __str__(self):
        return "AstStatement(" + str(self.type) + ", " + str(self.value) + ")"


class Ast(AstRelated):
    """

    Attributes:
        assignments: List of AstAssignment objects with parse function

    """

    def __init__(self, statements: List):
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
            ParseError: Parser could not parse the code

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
                expression, stream = Expression.from_stream([next_token] + stream)
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
                raise ParseError(
                    "Invalid state reached, please report bug to lcaml maintainers"
                )
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
    let z = x + y;
    """
    from lcaml_lexer import Lexer, Syntax

    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    parser = Parser(tokens)
    ast = parser()
    for statement in ast.statements:
        print(statement)
