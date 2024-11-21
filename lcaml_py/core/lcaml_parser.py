from . import lcaml_expression as lcaml_expression
from . import parser_types as parser_types

from typing import Union, List, Set
from .token_type import Token, TokenKind
from .ast_related import AstRelated
from .lcaml_lexer import Syntax


TokenStream = List[Token]


class ParseState:
    """
    Enum for parser state
    """

    ExpectStatementOrEnd = 0
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
        value: Union[
            "parser_types.AstAssignment",
            "parser_types.AstReturn",
            "parser_types.AstControlFlow",
            "parser_types.AstWhileLoop",
            "parser_types.AstExpressionStatement",
        ],
        line: int = -1,
    ):
        self.type = type
        self.value = value
        self.line = line

    def __str__(self):
        return "AstStatement(" + str(self.type) + ", " + str(self.value) + ", line " + str(self.line) + ")"

    def to_python(self):
        return self.value.to_python()


class Ast(AstRelated):
    """
    Abstract Syntax Tree

    Attributes:
        statements: List of AstStatement objects with parse function

    """

    def __init__(self, statements: List["AstStatement"], line: int = -1):
        self.statements = statements
        self.line = line

    def __str__(self):
        return "Ast(" + str(self.statements) + ", line " + str(self.line) + ")"

    def to_python(self):
        return (
            "",
            "\n".join(
                "\n".join(statement.to_python()) for statement in self.statements
            ),
            "",
        )

    @classmethod
    def from_stream(cls, stream: TokenStream, syntax: Syntax = Syntax()):
        """

        Args:
            syntax: Syntax to be used
            stream: TokenStream to parse

        Returns:
            Ast: Ast object built from tokenstream

        Raises:
            ParseError: Parser could not parse the code

        """
        stream = [token for token in stream if token.type != TokenKind.COMMENT]
        statements = []
        state = ParseState.ExpectStatementOrEnd
        identifier = None
        all_symbols_used: Set[lcaml_expression.Variable] = set()
        while stream:
            token = stream.pop(0)

            try:
                if state == ParseState.ExpectStatementOrEnd:
                    if token.type == TokenKind.LET:
                        state = ParseState.ExpectIdentfier
                        identifier = None
                    elif token.type == TokenKind.RETURN:
                        (
                            expression,
                            stream,
                            symbols_used,
                        ) = lcaml_expression.Expression.from_stream(stream, syntax)
                        all_symbols_used.update(symbols_used)
                        return_statement = parser_types.AstReturn(expression)
                        statement = AstStatement(
                            parser_types.AstStatementType.RETURN,
                            return_statement,
                            token.line,
                        )
                        statements.append(statement)
                        state = ParseState.ExpectSemicolon
                    elif token.type == TokenKind.IF:
                        (
                            while_loop,
                            stream,
                            symbols_used,
                        ) = parser_types.AstControlFlow.from_stream(stream, syntax)
                        all_symbols_used.update(symbols_used)
                        statement = AstStatement(
                            parser_types.AstStatementType.CONTROL_FLOW,
                            while_loop,
                            token.line,
                        )
                        statements.append(statement)
                    elif token.type == TokenKind.WHILE:
                        (
                            while_loop,
                            stream,
                            symbols_used,
                        ) = parser_types.AstWhileLoop.from_stream(stream, syntax)
                        all_symbols_used.update(symbols_used)
                        statement = AstStatement(
                            parser_types.AstStatementType.WHILE_LOOP, while_loop, token.line
                        )
                        statements.append(statement)
                    elif token.type == TokenKind.SEMICOLON:
                        pass  # semicolon is always ok
                    else:
                        (
                            expression,
                            stream,
                            symbols_used,
                        ) = lcaml_expression.Expression.from_stream(
                            [token] + stream, syntax
                        )
                        all_symbols_used.update(symbols_used)
                        expr_stmt = parser_types.AstExpressionStatement(expression)
                        statement = AstStatement(
                            parser_types.AstStatementType.EXPRESSION, expr_stmt, token.line
                        )
                        statements.append(statement)
                        state = ParseState.ExpectSemicolon
                        # raise ParseError("Expected let, return, if or end of file")

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
                    (
                        expression,
                        stream,
                        symbols_used,
                    ) = lcaml_expression.Expression.from_stream([token] + stream, syntax)
                    all_symbols_used.update(symbols_used)
                    if identifier is None:
                        raise ParseError(
                            "Could not parse out identifier (probably syntax error, maybe interpreter bug)"
                        )
                    assignment = parser_types.AstAssignment(identifier, expression)
                    statement = AstStatement(
                        parser_types.AstStatementType.ASSIGNMENT, assignment, token.line
                    )
                    statements.append(statement)
                    state = ParseState.ExpectSemicolon

                elif state == ParseState.ExpectSemicolon:
                    if token.type == TokenKind.SEMICOLON:
                        state = ParseState.ExpectStatementOrEnd
                    else:
                        raise ParseError("Expected semicolon")

                else:
                    raise ParseError(
                        "Invalid state reached, please report bug to lcaml maintainers"
                    )
            except Exception as e:
                if not hasattr(e, "__lcaml_traceback_info"):
                    setattr(e, "__lcaml_traceback_info", [token.line])
                raise e
        if state != ParseState.ExpectStatementOrEnd:
            raise ParseError("Unexpected end of file")
        return cls(statements), all_symbols_used


class Parser:
    """

    Attributes:
        stream: TokenStream to parse
        syntax: Syntax object to use for parsing

    """

    def __init__(self, stream: TokenStream, syntax: Syntax, file="<unknown>", code=""):
        self.stream = stream
        self.syntax = syntax
        self.file = file
        self.code = code

    def __call__(self) -> Ast:
        try:
            ast, _ = Ast.from_stream(self.stream, self.syntax)
        except Exception as e:
            if not hasattr(e, "__lcaml_traceback_info"):
                setattr(e, "__lcaml_traceback_info", [])
            getattr(e, "__lcaml_traceback_info").append((self.file, self.code))
            raise e
        return ast
