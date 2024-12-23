from . import lcaml_expression as lcaml_expression
from . import lcaml_parser as lcaml_parser

from typing import List, Any, Tuple, Set
from .lcaml_utils import split_at_context_end, PhantomType, indent
from .token_type import Token, TokenKind
from .ast_related import AstRelated
from .lcaml_lexer import Syntax


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

    def to_python(self):
        return "", self.name, ""


class AstStatementType:
    ASSIGNMENT = 0
    RETURN = 1
    CONTROL_FLOW = 2
    WHILE_LOOP = 3
    EXPRESSION = 4


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

    def to_python(self):
        id_pre_insert, id_expr, id_post_insert = self.identifier.to_python()
        (
            value_pre_insert,
            value_expr,
            value_post_insert,
        ) = self.value.to_python()
        id_expr = f'_ad7aaf167f237a94dc2c3ad2["{id_expr}"]'
        return (
            "",
            "\n".join(
                (
                    id_pre_insert,
                    value_pre_insert,
                    id_expr + " = " + value_expr,
                    id_post_insert,
                    value_post_insert,
                )
            ),
            "",
        )


class AstReturn(AstRelated):
    """

    Attributes:
        value: (Expression) value to return

    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "AstReturn(" + str(self.value) + ")"

    def to_python(self):
        (
            value_pre_insert,
            value_expr,
            value_post_insert,
        ) = self.value.to_python()
        return (
            "",
            "\n".join((value_pre_insert, "return " + value_expr, value_post_insert)),
            "",
        )


class AstControlFlowBranch(AstRelated):
    """

    Attributes:
        condition: (Expression) condition to check
        body: (Ast) body to execute if condition is true

    """

    def __init__(self, condition, body, _is_first: bool = False, line: int = -1):
        self.condition = condition
        self.body = body
        self._is_first = _is_first
        self.line = line

    def __str__(self):
        return (
            "AstControlFlowBranch(" + str(self.condition) + ", " + str(self.body) + ", line " + str(self.line) + ")"
        )

    def to_python(self):
        kw = "if" if self._is_first else "elif"
        cond_pre_insert, cond_expr, cond_post_insert = self.condition.to_python()
        block_pre_insert, block, block_post_insert = self.body.to_python()

        return (
            cond_pre_insert + "\n" + block_pre_insert,
            kw + " " + cond_expr + ":\n" + indent(block),
            cond_post_insert + "\n" + block_post_insert,
        )


class AstControlFlow(AstRelated):
    """
    Attributes:
        conditions: (List[AstControlFlowBranch]) list of conditions
    """

    def __init__(self, branches: List["AstControlFlowBranch"]):
        self.branches = branches

    def __str__(self):
        return "AstControlFlow(" + str(self.branches) + ")"

    def to_python(self):
        pre_inserts, branches, post_inserts = zip(
            *(branch.to_python() for branch in self.branches)
        )
        pre_insert = "\n".join(pre_inserts)
        post_insert = "\n".join(post_inserts)
        insert = pre_insert + "\n" + "\n".join(branches) + "\n" + post_insert
        return "", insert, ""

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
        cond_open_tok = stream.pop(0)
        if cond_open_tok.type != TokenKind.LPAREN:  # check/remove LPAREN
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
        branch = AstControlFlowBranch(expression, body, _is_first=True, line=cond_open_tok.line)
        branches.append(branch)

        # parse all the else ifs
        while stream:
            else_if_tok = stream.pop(0)

            if else_if_tok.type != TokenKind.ELSE_IF:
                stream.insert(0, else_if_tok)
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

            branch = AstControlFlowBranch(expression, body, line=else_if_tok.line)
            branches.append(branch)

        else_tok = stream.pop(0)

        if else_tok.type != TokenKind.ELSE:
            stream.insert(0, else_tok)
            return AstControlFlow(branches), stream, all_symbols_used

        if stream.pop(0).type != TokenKind.LCURLY:
            raise ValueError("Expected LCURLY after else statement")

        body, stream = split_at_context_end(stream, BODY_END_TOKEN)
        stream.pop(0)  # remove RCURLY
        body, symbols_used = lcaml_parser.Ast.from_stream(body, syntax)
        all_symbols_used.update(symbols_used)

        branch = AstControlFlowBranch(ELSE_ARTIFICIAL_IF_EXPRESSION, body, line=else_tok.line)
        branches.append(branch)

        return cls(branches), stream, all_symbols_used


class AstWhileLoop(AstRelated):
    def __init__(self, condition: "lcaml_expression.Expression", body: "lcaml_parser.Ast"):
        self.condition = condition
        self.body = body

    def __str__(self):
        return "AstWhileLoop(" + str(self.condition) + ", " + str(self.body) + ")"

    def to_python(self):
        kw = "while"
        cond_pre_insert, cond_expr, cond_post_insert = self.condition.to_python()
        block_pre_insert, block, block_post_insert = self.body.to_python()

        return (
            cond_pre_insert + "\n" + block_pre_insert,
            kw + " " + cond_expr + ":\n" + indent(block),
            cond_post_insert + "\n" + block_post_insert,
        )

    @classmethod
    def from_stream(
        cls, stream: TokenStream, syntax: Syntax = Syntax()
    ) -> Tuple[Any, TokenStream, Set[Any]]:
        """

        Args:
            stream: TokenStream to parse

        Returns:
            AstWhileLoop: object built from tokenstream

        Raises:
            ParseError: Parser could not parse the code

        """
        all_symbols_used: Set[lcaml_expression.Variable] = set()
        # constants
        BODY_END_TOKEN = Token(TokenKind.RCURLY, PhantomType())
        CONDITION_END_TOKEN = Token(TokenKind.RPAREN, PhantomType())

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

        return cls(expression, body), stream, all_symbols_used


class AstExpressionStatement(AstRelated):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return f"AstExpressionStatement({self.expression})"

    def to_python(self):
        return "", "\n".join(self.expression.to_python()), ""
