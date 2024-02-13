import expression as lcaml_expression
import parser_types

from typing import Union, List, Set
from token_type import Token, TokenKind
from ast_related import AstRelated
from lcaml_lexer import Syntax


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

    def __init__(
        self,
        type: int,
        value: Union[parser_types.AstAssignment, parser_types.AstReturn],
    ):
        self.type = type
        self.value = value

    def __str__(self):
        return "AstStatement(" + str(self.type) + ", " + str(self.value) + ")"


class Ast(AstRelated):
    """
    Abstract Syntax Tree

    Attributes:
        statements: List of AstStatement objects with parse function

    """

    def __init__(self, statements: List):
        self.statements = statements

    def __str__(self):
        return "Ast(" + str(self.statements) + ")"

    @classmethod
    def from_stream(cls, stream: TokenStream, syntax: Syntax = Syntax()):
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
        all_symbols_used: Set[lcaml_expression.Variable] = set()
        while stream:
            token = stream.pop(0)

            if state == ParseState.ExpectStatementOrCommentOrEnd:
                if token.type == TokenKind.COMMENT:
                    continue
                elif token.type == TokenKind.LET:
                    state = ParseState.ExpectIdentfier
                    identifier = None
                elif token.type == TokenKind.RETURN:
                    expression, stream, symbols_used = lcaml_expression.Expression.from_stream(
                        stream, syntax
                    )
                    all_symbols_used.update(symbols_used)
                    return_statement = parser_types.AstReturn(expression)
                    statement = AstStatement(
                        parser_types.AstStatementType.RETURN, return_statement
                    )
                    statements.append(statement)
                    state = ParseState.ExpectSemicolon
                elif token.type == TokenKind.IF:
                    control_flow, stream, symbols_used = parser_types.AstControlFlow.from_stream(
                        stream, syntax
                    )
                    all_symbols_used.update(symbols_used)
                    statement = AstStatement(
                        parser_types.AstStatementType.CONTROL_FLOW, control_flow
                    )
                    statements.append(statement)
                elif token.type == TokenKind.SEMICOLON:
                    pass  # semicolon is always ok
                else:
                    raise ParseError("Expected let or end of file")

            elif state == ParseState.ExpectIdentfier:
                if token.type == TokenKind.IDENTIFIER:
                    identifier = parser_types.AstIdentifier(token)
                    state = ParseState.ExpectEquals
                else:
                    raise ParseError("Expected identifier")

            elif state == ParseState.ExpectEquals:
                if token.type == TokenKind.EQUALS:
                    state = ParseState.ExpectExpression
                else:
                    raise ParseError("Expected equals sign")

            elif state == ParseState.ExpectExpression:
                expression, stream, symbols_used = lcaml_expression.Expression.from_stream(
                    [token] + stream, syntax
                )
                all_symbols_used.update(symbols_used)
                if identifier is None:
                    raise ParseError(
                        "Could not parse out identifier (probably syntax error, maybe interpreter bug)"
                    )
                assignment = parser_types.AstAssignment(identifier, expression)
                statement = AstStatement(
                    parser_types.AstStatementType.ASSIGNMENT, assignment
                )
                statements.append(statement)
                state = ParseState.ExpectSemicolon

            elif state == ParseState.ExpectSemicolon:
                if token.type == TokenKind.SEMICOLON:
                    state = ParseState.ExpectStatementOrCommentOrEnd
                else:
                    raise ParseError("Expected semicolon")

            else:
                raise ParseError(
                    "Invalid state reached, please report bug to lcaml maintainers"
                )
        if state != ParseState.ExpectStatementOrCommentOrEnd:
            raise ParseError("Unexpected end of file")
        return cls(statements), all_symbols_used


class Parser:
    """

    Attributes:
        stream: TokenStream to parse
        syntax: Syntax object to use for parsing

    """

    def __init__(self, stream: TokenStream, syntax: Syntax):
        self.stream = stream
        self.syntax = syntax

    def __call__(self) -> Ast:
        ast, _ = Ast.from_stream(self.stream, self.syntax)
        return ast


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
    parser = Parser(tokens, syntax)
    ast = parser()
    for statement in ast.statements:
        print(statement)
