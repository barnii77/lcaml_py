import lcaml_expression
import lcaml_parser

from typing import List, Any, Tuple, Set
from lcaml_utils import split_at_context_end, PhantomType
from token_type import Token, TokenKind
from ast_related import AstRelated
from lcaml_lexer import Syntax


TokenStream = List[Token]


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

    def __eq__(self, other):
        if not isinstance(other, AstIdentifier):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class AstStatementType:
    ASSIGNMENT = 0
    RETURN = 1
    CONTROL_FLOW = 2


class AstAssignment(AstRelated):
    """

    Attributes:
        identifier: identifier to write to
        value: (Expression) value to assign

    """

    def __init__(self, identifier: AstIdentifier, value):
        self.identifier = identifier
        self.value = value

    def __str__(self):
        return "AstAssignment(" + str(self.identifier) + ", " + str(self.value) + ")"


class AstReturn(AstRelated):
    """

    Attributes:
        value: (Expression) value to return

    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "AstReturn(" + str(self.value) + ")"


class AstControlFlowBranch(AstRelated):
    """

    Attributes:
        condition: (Expression) condition to check
        body: (Ast) body to execute if condition is true

    """

    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __str__(self):
        return (
            "AstControlFlowBranch(" + str(self.condition) + ", " + str(self.body) + ")"
        )


class AstControlFlow(AstRelated):
    """
    Attributes:
        conditions: (List[AstControlFlowBranch]) list of conditions
    """

    def __init__(self, branches: list):
        self.branches = branches

    def __str__(self):
        return "AstControlFlow(" + str(self.branches) + ")"

    @classmethod
    def from_stream(
        cls, stream: TokenStream, syntax: Syntax = Syntax()
    ) -> Tuple[Any, TokenStream, Set[Any]]:
        # Any because AstControlFlow not yet defined
        """

        Args:
            stream: TokenStream to parse

        Returns:
            AstControlFlow: AstControlFlow object built from tokenstream

        Raises:
            ParseError: Parser could not parse the code

        """
        all_symbols_used: Set[lcaml_expression.Variable] = set()
        # constants
        STATEMENT_END_TOKEN = Token(TokenKind.SEMICOLON, PhantomType())
        BODY_END_TOKEN = Token(TokenKind.RCURLY, PhantomType())
        CONDITION_END_TOKEN = Token(TokenKind.RPAREN, PhantomType())
        # construct artificial if expression for else using artificial stream of boolean true followed by semicolon
        ELSE_ARTIFICIAL_IF_EXPRESSION, _, _ = lcaml_expression.Expression.from_stream(
            [
                Token(TokenKind.BOOLEAN, syntax._true),
                STATEMENT_END_TOKEN,
            ],
            syntax,
            STATEMENT_END_TOKEN,
        )
        # parse entire if - else if - else chain

        branches = []

        # parse initial if
        # followed by expression
        if stream.pop(0).type != TokenKind.LPAREN:  # check/remove LPAREN
            raise ValueError("Expected LPAREN after if statement")

        expression, stream, symbols_used = lcaml_expression.Expression.from_stream(
            stream, syntax, CONDITION_END_TOKEN
        )
        stream.pop(0)  # remove RPAREN
        all_symbols_used.update(symbols_used)

        if stream.pop(0).type != TokenKind.LCURLY:  # check/remove LCURLY
            raise ValueError("Expected LCURLY after if statement")

        body, stream = split_at_context_end(stream, BODY_END_TOKEN)
        stream.pop(0)  # remove RCURLY
        body, symbols_used = lcaml_parser.Ast.from_stream(body, syntax)
        all_symbols_used.update(symbols_used)
        branch = AstControlFlowBranch(expression, body)
        branches.append(branch)

        # parse all the else ifs
        while stream:
            token = stream.pop(0)

            if token.type != TokenKind.ELSE_IF:
                stream.insert(0, token)
                break

            if stream.pop(0).type != TokenKind.LPAREN:  # check/remove LPAREN
                raise ValueError("Expected LPAREN after if statement")

            expression, stream, symbols_used = lcaml_expression.Expression.from_stream(
                stream, syntax, CONDITION_END_TOKEN
            )
            stream.pop(0)  # remove RPAREN
            all_symbols_used.update(symbols_used)

            if stream.pop(0).type != TokenKind.LCURLY:
                raise ValueError("Expected LCURLY after else if statement")

            body, stream = split_at_context_end(stream, BODY_END_TOKEN)
            stream.pop(0)  # remove RCURLY
            body, symbols_used = lcaml_parser.Ast.from_stream(body, syntax)
            all_symbols_used.update(symbols_used)

            branch = AstControlFlowBranch(expression, body)
            branches.append(branch)

        token = stream.pop(0)

        if token.type != TokenKind.ELSE:
            stream.insert(0, token)
            return AstControlFlow(branches), stream, all_symbols_used

        if stream.pop(0).type != TokenKind.LCURLY:
            raise ValueError("Expected LCURLY after else statement")

        body, stream = split_at_context_end(stream, BODY_END_TOKEN)
        stream.pop(0)  # remove RCURLY
        body, symbols_used = lcaml_parser.Ast.from_stream(body, syntax)
        all_symbols_used.update(symbols_used)

        branch = AstControlFlowBranch(ELSE_ARTIFICIAL_IF_EXPRESSION, body)
        branches.append(branch)

        return cls(branches), stream, all_symbols_used
