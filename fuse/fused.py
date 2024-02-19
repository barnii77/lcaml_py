################# ast_related.py ##################

from typing import List, Dict
import re

class ParseState:
    """
    Enum for parser state
    """

    ExpectStatementOrCommentOrEnd = 0
    ExpectIdentfier = 1
    ExpectEquals = 2
    ExpectExpression = 3
    ExpectSemicolon = 4
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


def split_at_context_end(
        stream: TokenStream, terminating_token: Token
) -> tuple[TokenStream, TokenStream]:
    context_stack = []
    terminating_idx = 0
    while terminating_idx < len(stream):
        token = stream[terminating_idx]

        # add token if it starts a context
        if token.type in (
                TokenKind.LPAREN,
                TokenKind.LSQUARE,
                TokenKind.LCURLY,
        ):
            context_stack.append(token)

        # remove token if it ends a context
        if context_stack and is_token_pair(context_stack[-1], token):
            context_stack.pop()
        # break if terminating token found and no inner contexts
        elif not context_stack and token == terminating_token:  # no inner contexts
            break

        terminating_idx += 1

    body_stream, remaining_stream = (
        stream[:terminating_idx],
        stream[terminating_idx:],  # skip RCURLY
    )

    return body_stream, remaining_stream



class Syntax:
    """
    This class defines the syntax of the language by containing named regex patterns.

    Attributes:
        let:
        identifier:
        integer:
        floating_point:
        string_literal:
        equals:
        semicolon:
        comment:
        operator:

    """

    def __init__(self, **kwargs):
        # non-syntax-pattern stuff
        self.extract_fn_args: Callable = lambda args_str: args_str.strip("|").split(",")

        # keywords
        self.let = r"let\s"
        self.return_keyword = r"return\s"
        self.struct_keyword = r"struct(?![a-zA-Z0-9_])"
        self.if_keyword = r"if\s"
        self.else_if_keyword = r"else\s+if\s"
        self.else_keyword = r"else\s"

        # identifiers and builtins
        self.identifier = r"[a-zA-Z_][a-zA-Z0-9_]*"
        self.function_args = (
            f"\\|\\s*({self.identifier}\\s*,\\s*)*{self.identifier}\\s*,?\\s*\\|"
        )
        self.unit_type = r"\(\)"
        self.floating_point = r"[0-9]+\.[0-9]+"  # be careful - define this before int so it first checks this
        self.integer = r"-?[0-9]+"
        self._true = r"true"
        self.boolean = r"true|false"
        self.string_literal = r"\".*\""
        self.comment = r"--.*\n"

        # operators
        operators = (
            "!",
            "~",
            "*",
            "/",
            "%",
            "+",
            "-",
            "==",
            "!=",
            "<=",
            ">=",
            "<",
            ">",
            "||",
            "&&",
            "|",
            "&",
        )
        self.operator = "|".join("".join(f"\\{c}" for c in op) for op in operators)

        # symbols
        self.comma = r","
        self.equals = r"="
        self.semicolon = r";"
        self.lparen = r"\("
        self.rparen = r"\)"
        self.lsquare = r"\["
        self.rsquare = r"\]"
        self.lcurly = r"\{"
        self.rcurly = r"\}"

        self.set_custom(kwargs)
        self.compiled_patterns = self.get_compiled_patterns()

    def set_custom(self, kwargs: Dict[str, str]):
        """
        This function is used by the initializer to set the custom syntax patterns.

        Args:
            kwargs: the kwargs passed to the init function

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def patterns(self):
        """

        Returns:
            Iterable[Tuple[str, str]]: A iterable of tuples of the form [(name, pattern), ...] that excludes fields starting with _ (_fieldname)

        """
        return filter(lambda i: not i[0].startswith("_"), vars(self).items())

    def get_compiled_patterns(self) -> dict:
        result = {}
        for k, pattern in self.patterns():
            result[k] = re.compile(f"^\\s*({pattern})")
        return result


class Lexer:
    """

    Attributes:
        code (str): The code to lex
        syntax (Syntax): The syntax to use for lexing
        num_symbols (int): The number of symbols in the code
        tokens (list[Token]): The tokens found in the code (when called multiple times, will not recompute)

    """

    def __init__(self, code: str, syntax: Syntax):
        self.code = code
        self.syntax = syntax
        # self.state = LexState()
        self.num_symbols = len(code)
        self.tokens = []

    def __call__(self) -> List[Token]:
        """

        Returns:
            list[Token]: A list of tokens

        Raises:
            LexError: If no matching pattern is found

        """
        if self.tokens:  # if already lexed, just return the tokens
            return self.tokens

        code = self.code

        while code.strip() != "":
            # match all the patterns in the syntax
            for kind, pattern in self.syntax.compiled_patterns.items():
                m = pattern.match(code)
                if m:
                    break
            else:
                raise LexError("No matching pattern for " + code)

            # save the match as a token
            token_value = m.group(1).strip()
            self.tokens.append(Token(kind, token_value))

            # increment position
            token_len = m.end(1)
            code = code[token_len:]

        return self.tokens

class Ast(AstRelated):
    """

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
        all_symbols_used: Set[Variable] = set()
        while stream:
            token = stream.pop(0)

            if state == ParseState.ExpectStatementOrCommentOrEnd:
                if token.type == TokenKind.COMMENT:
                    continue
                elif token.type == TokenKind.LET:
                    state = ParseState.ExpectIdentfier
                    identifier = None
                elif token.type == TokenKind.RETURN:
                    expression, stream, symbols_used = Expression.from_stream(
                        stream, syntax
                    )
                    all_symbols_used.update(symbols_used)
                    return_statement = AstReturn(expression)
                    statement = AstStatement(
                        AstStatementType.RETURN, return_statement
                    )
                    statements.append(statement)
                    state = ParseState.ExpectSemicolon
                elif token.type == TokenKind.IF:
                    control_flow, stream, symbols_used = AstControlFlow.from_stream(
                        stream, syntax
                    )
                    all_symbols_used.update(symbols_used)
                    statement = AstStatement(
                        AstStatementType.CONTROL_FLOW, control_flow
                    )
                    statements.append(statement)
                elif token.type == TokenKind.SEMICOLON:
                    pass  # semicolon is always ok
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
                expression, stream, symbols_used = Expression.from_stream(
                    [token] + stream, syntax
                )
                all_symbols_used.update(symbols_used)
                if identifier is None:
                    raise ParseError(
                        "Could not parse out identifier (probably syntax error, maybe interpreter bug)"
                    )
                assignment = AstAssignment(identifier, expression)
                statement = AstStatement(
                    AstStatementType.ASSIGNMENT, assignment
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

class AstRelated:
    pass


class Object:
    pass


class Resolvable:
    pass


class Gettable:
    pass




class TokenKind:
    LET = "let"
    RETURN = "return_keyword"  # NOTE: because return is a python keyword, this naming is required
    STRUCT = "struct_keyword"
    IF = "if_keyword"
    ELSE_IF = "else_if_keyword"
    ELSE = "else_keyword"
    IDENTIFIER = "identifier"
    FUNCTION_ARGS = "function_args"
    UNIT_TYPE = "unit_type"
    INTEGER = "integer"
    FLOATING_POINT = "floating_point"
    STRING_LITERAL = "string_literal"
    BOOLEAN = "boolean"
    EQUALS = "equals"
    SEMICOLON = "semicolon"
    COLON = "colon"
    COMMA = "comma"
    COMMENT = "comment"
    OPERATOR = "operator"
    LPAREN = "lparen"
    RPAREN = "rparen"
    LSQUARE = "lsquare"
    RSQUARE = "rsquare"
    LCURLY = "lcurly"
    RCURLY = "rcurly"
    _builtin_types = [
        UNIT_TYPE,
        INTEGER,
        FLOATING_POINT,
        STRING_LITERAL,
        BOOLEAN,
    ]



class PhantomType:
    """
    A phantom type that will always be equal to everything.
    """

    def __init__(self, *_, **__):
        pass

    def __eq__(self, _):
        return True

    def __repr__(self):
        return "PhantomType()"

    def __radd__(self, other):
        if not isinstance(other, str):
            raise TypeError("Can only add PhantomType to string")
        return other + "PhantomType()"

class AstReturn:
    pass

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


class OperationKind:
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4
    NOT = 5
    EQ = 6
    NEQ = 7
    LT = 8
    GT = 9
    FLIP = 10
    LTE = 11
    GTE = 12
    OR = 13
    AND = 14
    BITOR = 15
    BITAND = 16

    _unary = [NOT, FLIP]





################# expression.py ##################


from typing import List, Dict, Optional, Set, Iterable


TokenStream = List[Token]
Context = Dict[AstIdentifier, Object]


SYMBOL_TO_OPKIND = {
    "+": OperationKind.ADD,
    "-": OperationKind.SUB,
    "*": OperationKind.MUL,
    "/": OperationKind.DIV,
    "%": OperationKind.MOD,
    "!": OperationKind.NOT,
    "==": OperationKind.EQ,
    "!=": OperationKind.NEQ,
    "<": OperationKind.LT,
    ">": OperationKind.GT,
    "~": OperationKind.FLIP,
    "<=": OperationKind.LTE,
    ">=": OperationKind.GTE,
    "||": OperationKind.OR,
    "&&": OperationKind.AND,
    "|": OperationKind.BITOR,
    "&": OperationKind.BITAND,
}


class FunctionParseState:
    ExpectCommaOrEndOfArgs = 0
    ExpectArgumentOrEndOfArgs = 1


class Function(AstRelated, Resolvable):
    """

    Attributes:
        ast: (lcaml_parser.Ast) the AST that makes up the function
        arguments: the arguments of the function
        bounds: the bounds of the function (functions can bind to values defined when they are defined, then these values go out of scope but the function is still bound to them)

    """

    def __init__(
        self, body, arguments: List[AstIdentifier], bounds: Iterable[AstIdentifier]
    ):
        self.body = body
        self.arguments = arguments
        if isinstance(bounds, dict):
            self.bounds: Dict[AstIdentifier, Object] = bounds
        else:
            self.bounds: Dict[AstIdentifier, Object] = {ident: None for ident in bounds}

    def __str__(self):
        return "Function(" + str(self.body) + ", " + str(self.arguments) + ")"

    def resolve(self, context: Context) -> Object:
        # if this is called, it's probably trying to resolve identifier but actually already has function
        # use context to resolve bounds
        intersecting_keys = self.bounds.keys() & context.keys()
        for key in intersecting_keys:
            self.bounds[key] = context[key]
        return Object(DType.FUNCTION, self)

    @classmethod
    def from_stream(
        cls,
        stream: TokenStream,
        syntax: Syntax = Syntax(),
    ):
        """
        Builds function from stream

        Args:
            stream: stream to build from

        Returns:
            Function: function built from stream

        Raises:
            ValueError: Syntax error or interpreter bug
        """
        # check for function argument token
        # this token contains a bunch of function arguments
        # token.value = "|x, y, z|"
        # first, extract arguments
        first_token = stream.pop(0)
        if first_token.type != TokenKind.FUNCTION_ARGS:
            raise ValueError(
                f"Expected function arguments, got {first_token.type} instead"
            )

        identifiers_raw = map(str.strip, syntax.extract_fn_args(first_token.value))
        arguments = list(
            map(
                lambda raw_id: AstIdentifier(Token(TokenKind.IDENTIFIER, raw_id)),
                identifiers_raw,
            )
        )
        # consume function body
        first_body_token = stream.pop(0)
        if first_body_token.type != TokenKind.LCURLY:
            raise ValueError(
                f"Expected lcurly for function body, got {first_body_token.type}"
            )

        body_stream, remaining_stream = split_at_context_end(
            stream, Token(TokenKind.RCURLY, PhantomType())
        )
        remaining_stream.pop(0)  # remove RCURLY
        body, symbols_used = Ast.from_stream(body_stream, syntax)
        return (
            cls(
                body, arguments, map(lambda variable: variable.identifier, symbols_used)
            ),
            remaining_stream,
            symbols_used,
        )


class Operation(AstRelated, Resolvable):
    def __init__(self, left: Optional[Resolvable], operation: Token, right: Resolvable):
        optype = SYMBOL_TO_OPKIND.get(operation.value)
        if optype is None:
            raise ValueError(f"Unknown operation {operation.value}")
        self.left = left
        self.operation = optype
        self.right = right
        self.is_unary = optype in [OperationKind.FLIP, OperationKind.NOT]

    def __str__(self):
        return (
            "Operation("
            + str(self.left)
            + ", "
            + str(self.operation)
            + ", "
            + str(self.right)
            + ")"
        )

    def resolve(self, context: Context) -> Object:
        """
        This function evaluates the operation for values of left and right
        """
        if self.left is None:
            left: Object = None  # typehint so the lsp doesnt spam me with warnings
        else:
            left: Object = self.left.resolve(context)

        right = self.right.resolve(context)
        if left is None and self.operation not in OperationKind._unary:
            raise ValueError("Left operand must not be None")

        if self.operation == OperationKind.ADD:
            return left.add(right)
        elif self.operation == OperationKind.SUB:
            return left.sub(right)
        elif self.operation == OperationKind.MUL:
            return left.mul(right)
        elif self.operation == OperationKind.DIV:
            return left.div(right)
        elif self.operation == OperationKind.MOD:
            return left.mod(right)
        elif self.operation == OperationKind.NOT:
            return right.bool_not()
        elif self.operation == OperationKind.EQ:
            return left.eq(right)
        elif self.operation == OperationKind.NEQ:
            return left.neq(right)
        elif self.operation == OperationKind.LT:
            return left.lt(right)
        elif self.operation == OperationKind.GT:
            return left.gt(right)
        elif self.operation == OperationKind.FLIP:
            return right.flip()
        elif self.operation == OperationKind.LTE:
            return left.lte(right)
        elif self.operation == OperationKind.GTE:
            return left.gte(right)
        elif self.operation == OperationKind.OR:
            return left.bool_or(right)
        elif self.operation == OperationKind.AND:
            return left.bool_and(right)
        elif self.operation == OperationKind.BITOR:
            return left.bitor(right)
        elif self.operation == OperationKind.BITAND:
            return left.bitand(right)
        else:
            raise ValueError(f"Unknown operation type {self.operation}")


class StructParseState:
    ExpectFieldOrEnd = 0
    ExpectCommaOrEnd = 1


class StructType(AstRelated, Resolvable):
    """

    Attributes:
        fields: the names of the fields of a struct

    """

    def __init__(self, fields: List[AstIdentifier]):
        self.fields = fields

    def __str__(self):
        return f"StructType({self.fields})"

    def resolve(self, context: Context):
        return Object(DType.STRUCT_TYPE, self)

    @classmethod
    def from_stream(cls, stream: TokenStream):
        token = stream.pop(0)
        if token.type != TokenKind.STRUCT:
            raise ValueError(f"expected struct keyword, got {token}")
        token = stream.pop(0)
        if token.type != TokenKind.LCURLY:
            raise ValueError(f"expected lcurly, got {token}")

        state = StructParseState.ExpectFieldOrEnd
        fields: List[AstIdentifier] = []
        while stream:
            token = stream.pop(0)
            if state == StructParseState.ExpectFieldOrEnd:
                if token.type == TokenKind.IDENTIFIER:
                    field = AstIdentifier(token)
                    fields.append(field)
                    state = StructParseState.ExpectCommaOrEnd
                elif token.type == TokenKind.RCURLY:
                    break
                else:
                    raise ValueError(f"Expected field, got {token}")
            elif state == StructParseState.ExpectCommaOrEnd:
                if token.type == TokenKind.COMMA:
                    state = StructParseState.ExpectFieldOrEnd
                elif token.type == TokenKind.RCURLY:
                    break
                else:
                    raise ValueError(f"expected comma, got {token}")
            else:
                raise RuntimeError(
                    f"Internal Bug: invalid state reached in parsing of struct definition (state code: {state})"
                )
        else:
            raise ValueError("Unexpected end of tokenstream")

        return cls(fields), stream, set()


class StructInstance(AstRelated, Resolvable, Gettable):
    def __init__(
        self, fields: Dict[AstIdentifier, Resolvable], type: Optional[Resolvable] = None
    ):
        self.type = type
        self.fields = fields

    def __str__(self):
        return f"StructInstance({self.fields})"

    def resolve(self, context: Context):
        return Object(DType.STRUCT_INSTANCE, self)

    def get(self, ident: AstIdentifier) -> Object:
        if ident not in self.fields:
            raise ValueError(f"Field {ident} not found in struct {self}")
        return self.fields[ident]


class FieldAccess(AstRelated, Resolvable):
    """
    A class that represents a field access to a value. Can be stacked by having a FieldAccess as the object field of another FieldAccess

    Attributes:
        object: the resolvable (expression, variable, whatever) to access field from
        field: the field to look for and return

    """

    def __init__(self, object: Resolvable, field: AstIdentifier):
        """

        Args:
            object: the resolvable (expression, variable, whatever) to access field from
            field: the field to look for and return

        """
        self.object = object
        self.field = field

    def __str__(self):
        return "AstFieldAccess(" + str(self.object) + ", " + str(self.field) + ")"

    def resolve(self, context: Context) -> Object:
        obj = self.object.resolve(context)
        if isinstance(obj, Gettable):
            return obj.get(self.field)
        else:
            raise TypeError(
                f"Internal Error: Cannot access field {self.field} on non-struct {obj}"
            )


class FunctionCall(AstRelated, Resolvable):
    """

    Attributes:
        function_resolvable: (Resolvable[Function]) function to call
        arguments: (List[Expression]) arguments to call function with

    """

    def __init__(self, fuction_resolvable: Resolvable, arguments: List):
        """
        Resolved by spawning a new InterpreterVM

        Args:
            function_container: Object[Function]
            arguments: List[Expression]
        """
        self.function_resolvable = fuction_resolvable
        self.arguments = arguments

    def __str__(self):
        return (
            "FunctionCall("
            + str(self.function_resolvable)
            + ", "
            + str(self.arguments)
            + ")"
        )

    def resolve(self, context: Context) -> Optional[Object]:
        """
        Resolves the value of the function call by spawning a new InterpreterVM

        Args:
            context: context to resolve function in

        Returns:
            Object: return value of function call

        Raises:
            TypeError: Cannot call non-function
        """
        # resolve function if it is identifier
        function = self.function_resolvable.resolve(context)
        if isinstance(function, Object):
            function = function.value

        # resolve args and make arg locals
        resolved_args = [arg.resolve(context) for arg in self.arguments]

        if isinstance(function, Function):
            arg_locals = zip(function.arguments, resolved_args)

            if len(resolved_args) < len(function.arguments):
                # not all args provided -> return curried function
                # remove first n elements (curried away)
                remaining_args = function.arguments[len(resolved_args) :]
                # add curried args to bounds
                bounds = function.bounds.copy()
                bounds.update(arg_locals)
                result = Function(function.body, remaining_args, bounds)
                return Object(DType.FUNCTION, result)
            else:
                # all args provided -> execute function
                # create local context
                local_context = context.copy()
                # functions can bind to global values, example in docs
                # https://github.com/barnii77/lcaml_py/blob/main/docs/interpreter.md#functions
                non_none_bounds = {k: v for k, v in function.bounds.items() if v is not None}
                local_context.update(non_none_bounds)
                # overwrite variables from outer context with local args
                local_context.update(arg_locals)
                # spawn new interpreter vm
                interpreter_vm = InterpreterVM(function.body, local_context)
                interpreter_vm.execute()
                return interpreter_vm.return_value

        elif isinstance(function, ExternPython):
            return function.execute(context, resolved_args)

        else:
            raise TypeError("Cannot call non-function")


class Variable(AstRelated, Resolvable):
    def __init__(self, identifier: AstIdentifier):
        self.identifier = identifier

    def __str__(self):
        return "Variable(" + str(self.identifier) + ")"

    def resolve(self, context: Context):
        result = context.get(self.identifier)
        if result is None:
            raise RuntimeError(f"LCamlNameError: {self.identifier} is undefined")
        return result


class Constant(AstRelated, Resolvable):
    """

    Attributes:
        value: value of the constant (int, float, str, bool, ...)
        type: type of the constant (DType)

    """

    def __init__(self, token: Token, syntax: Syntax = Syntax()):
        if token.type == TokenKind.UNIT_TYPE:
            value = None
            kind = DType.UNIT
        elif token.type == TokenKind.INTEGER:
            try:
                value = int(token.value)
                kind = DType.INT
            except ValueError:
                raise ValueError(f"Invalid integer {token.value}")
        elif token.type == TokenKind.FLOATING_POINT:
            try:
                value = float(token.value)
                kind = DType.FLOAT
            except ValueError:
                raise ValueError(f"Invalid floating point {token.value}")
        elif token.type == TokenKind.STRING_LITERAL:
            value = token.value
            kind = DType.STRING
        elif token.type == TokenKind.BOOLEAN:
            try:
                value = syntax._true == token.value
                kind = DType.BOOL
            except ValueError:
                raise ValueError(f"Invalid boolean {token.value}")
        else:
            raise ValueError(f"Invalid type for constant: {token.type}")

        self.value = Object(kind, value)

    def __str__(self):
        return "Constant(" + str(self.value) + ")"

    def resolve(self, context: Context):
        return self.value


class ExpressionBuildState:
    EXPECT_RESOLVABLE = 0
    EXPECT_OPERATOR = 1


class Expression(AstRelated, Resolvable):
    """

    Attributes:
        expression: TokenStream of expression

    """

    def __init__(self, expression: Resolvable):
        self.expression = expression

    def __str__(self):
        return "AstExpression(" + str(self.expression) + ")"

    def resolve(self, context: Context):
        """
        This function resolves the value of the expression.
        """
        return self.expression.resolve(context)

    @classmethod
    def _build_from(cls, stream: TokenStream, syntax: Syntax = Syntax()):
        """
        Build from stream raw (expects stream to not contain any other tokens that do not belong to expression)
        This function does a total of 7 passes across the data:
            1. Make operations without parameters and parse tokens into constants, variables or Exressions
            2. Identify function calls
            (3.x) Fill out operations with their args:
                3.1. ! ~
                3.2. * / % & |
                3.3. + -
                3.4 | &
                3.5. == != < > <= >=
                3.6. || &&

        Args:
            stream: stream (only of Expression) to build from

        Raises:
            ValueError: Syntax error or interpreter bug

        Returns:
            Expression: Expression object built from stream

        """
        all_symbols_used: Set[Variable] = set()
        first_pass_buffer = []
        # first pass across data: make operations without parameters and parse tokens into constants, variables or Exressions
        # NOTE: this pass will not detect function calls
        while stream:
            # NOTE: inplace pop is fine because this is only called on expression_stream
            token = stream.pop(0)

            if token.type == TokenKind.OPERATOR:
                op = Operation(None, token, None)
                first_pass_buffer.append(op)
            elif token.type in TokenKind._builtin_types:
                first_pass_buffer.append(Constant(token, syntax))
            elif token.type == TokenKind.IDENTIFIER:
                first_pass_buffer.append(Variable(AstIdentifier(token)))
            elif token.type == TokenKind.LPAREN:
                expression, stream, symbols_used = cls.from_stream(
                    stream, syntax, Token(TokenKind.RPAREN, PhantomType())
                )
                first_pass_buffer.append(expression)
                stream.pop(0)  # remove RPAREN
                all_symbols_used.update(symbols_used)
            else:
                raise ValueError(f"Unexpected token {token}")

        # TODO: detect struct instantiations
        # second pass: identify function calls
        FUNCTION_CALL_ALLOWED_TYPES = (Constant, Variable, cls)
        second_pass_buffer = []
        # NOTE: will consume tokens from first_pass_buffer (by popping)
        while first_pass_buffer:
            # can be operation, constant, variable or expression
            thing = first_pass_buffer.pop(0)

            if isinstance(thing, Variable):  # add all symbols to all_symbols_used
                all_symbols_used.add(thing)

            if type(thing) in (Variable, cls):
                # might be function call
                # check the following token(s)
                if not first_pass_buffer:
                    # EOS
                    second_pass_buffer.append(thing)
                    break
                next_thing = first_pass_buffer[0]
                if type(next_thing) in FUNCTION_CALL_ALLOWED_TYPES:
                    # function call
                    function_call = FunctionCall(thing, [])
                    while (
                        first_pass_buffer
                        and type(first_pass_buffer[0]) in FUNCTION_CALL_ALLOWED_TYPES
                    ):  # NOTE: safe to run because python interpreter will short circuit type(first_pass_buffer[0]) part
                        # consume all arguments
                        next_arg = first_pass_buffer.pop(0)
                        function_call.arguments.append(next_arg)
                    second_pass_buffer.append(function_call)

                else:
                    # not function call
                    second_pass_buffer.append(thing)

            else:
                second_pass_buffer.append(thing)

        # third set of passes: fill out operations with their args
        # NOTE: this is a sequence of related passes because ! ~ have to be done before * / %
        # NOTE: and * / % have to be done before + -
        # NOTE and + - have to be done before | & which have to be done before == != < > <= >=
        # NOTE: which have to be done before || &&

        # stage one of third pass (unary): ! ~
        third_pass_buffer = []
        while second_pass_buffer:
            thing = second_pass_buffer.pop(0)

            if isinstance(thing, Operation):
                if thing.is_unary:
                    # NOTE: unary operands are always on the right
                    if not second_pass_buffer:
                        raise ValueError("Unary operand must have right operand")
                    right = second_pass_buffer.pop(0)
                    thing.right = right  # thing is unary operation

            third_pass_buffer.append(thing)

        # all the other passes (binary) are kind of the same
        sorted_pass_operations = (
            (
                OperationKind.MUL,
                OperationKind.DIV,
                OperationKind.MOD,
            ),
            (
                OperationKind.ADD,
                OperationKind.SUB,
            ),
            (
                OperationKind.BITOR,
                OperationKind.BITAND,
            ),
            (
                OperationKind.EQ,
                OperationKind.NEQ,
                OperationKind.LT,
                OperationKind.GT,
                OperationKind.LTE,
                OperationKind.GTE,
            ),
            (
                OperationKind.OR,
                OperationKind.AND,
            ),
        )
        this_pass_buffer = third_pass_buffer
        for pass_operations in sorted_pass_operations:
            # use prev_pass_buffer to store previous pass
            this_pass_buffer, prev_pass_buffer = [], this_pass_buffer

            while prev_pass_buffer:
                thing = prev_pass_buffer.pop(0)

                if isinstance(thing, Operation):
                    if thing.operation in pass_operations:
                        # NOTE: binary operands are always on the left and right
                        if not prev_pass_buffer:
                            raise ValueError(
                                "Binary operand must have left and right operand, but doesn't have right"
                            )
                        if not this_pass_buffer:
                            raise ValueError(
                                "Binary operand must have left and right operand, but doesn't have left"
                            )
                        left = this_pass_buffer.pop()
                        right = prev_pass_buffer.pop(0)
                        thing.left = left
                        thing.right = right

                this_pass_buffer.append(thing)

        if len(this_pass_buffer) != 1:
            raise ValueError(
                "Error while parsing expression: Syntax error or interpreter bug"
            )
        expression = this_pass_buffer[0]
        return cls(expression), all_symbols_used

    @classmethod
    def from_stream(
        cls,
        stream: TokenStream,
        syntax: Syntax = Syntax(),
        terminating_token: Token = Token(
            TokenKind.SEMICOLON, PhantomType()
        ),  # use PhantomType so it matches anything of that type
    ):
        """

        Args:
            stream: TokenStream to parse
            terminating_token: Token to terminate expression with

        Raises:
            ValueError: Semicolon not found (no end of expression found)

        Returns:
            Expression: AstExpression object built from tokenstream
            Stream: Remaining tokenstream
        """
        # FIXME will not work once functions are supported
        if not stream:
            raise ValueError("Empty stream")
        elif stream[0].type == TokenKind.FUNCTION_ARGS:
            # expression is a function (because function is a full expression and cannot be paired with other things in one expression, for that, you need subexpressions)
            function, remaining_stream, symbols_used = Function.from_stream(
                stream, syntax
            )
            return cls(function), remaining_stream, symbols_used
        elif stream[0].type == TokenKind.STRUCT:
            struct_type, remaining_stream, symbols_used = StructType.from_stream(stream)
            return cls(struct_type), remaining_stream, symbols_used

        if terminating_token not in stream:
            raise ValueError(f"Expression must end with a {terminating_token}")

        # exclude terminating and starting token
        # leave terminating token in remaining stream so the parser can be sure syntax is valid
        expression_stream, remaining_stream = split_at_context_end(
            stream, terminating_token
        )

        expression, symbols_used = cls._build_from(expression_stream, syntax)
        return expression, remaining_stream, symbols_used


################# extern_python.py ##################

from typing import Dict, List


Context = Dict[AstIdentifier, Object]


class ExternPython(Resolvable):
    """
    Abstract class for built-in functions that are implemented in Python.
    """

    def execute(
        self, context: Context, args: List[Object]
    ) -> Object:
        """
        This function is called when the built-in function is called.

        Args:
            context: The current context outside of function call.
            args: The **resolved** arguments to the function.

        """
        # NOTE: args are already resolved
        raise NotImplementedError()

    def __str__(self) -> str:
        return "Builtin()"


################# gettable.py ##################



class Gettable:
    def get(self, ident: AstIdentifier) -> Object:
        raise NotImplementedError()


################# interpreter.py ##################



class Interpreter:
    """

    Attributes:
        syntax: Syntax object containing language syntax info
        tokens: List of tokens of code
        ast: Abstract Syntax Tree of code
        vm: Virtual Machine to execute code

    Initializer Raises:
        A lot of exceptions depending on the code
        Among the most common are:
            ValueError
            SyntaxError
            RuntimeError (invalid code)
            LexError
            ParseError
    """
    def __init__(self, code: str):
        self.syntax = Syntax()
        self.tokens = Lexer(code, self.syntax)()
        self.ast = Parser(self.tokens, self.syntax)()
        self.vm = InterpreterVM(self.ast)

    def execute(self):
        """

        Returns:
            Any: The return value of the code

        """
        self.vm.variables = {}
        for name, value in BUILTINS.items():
            # construct the value (which is a class)
            name_ast_id = AstIdentifier(Token(TokenKind.IDENTIFIER, name))
            self.vm.variables[name_ast_id] = value()
        self.vm.execute()
        return self.vm.return_value


################# interpreter_test.py ##################

import os
from timeit import timeit


# def python_run():
#     def factorial(n):
#         if n <= 1:
#             return 1
#         else:
#             return n * factorial(n - 1)

#     return factorial(5)


def run():
    interpreter.vm.variables = {}
    result = interpreter.execute()
    print("interpreter returned: ", result)





################# interpreter_types.py ##################

from typing import Any


class DType:
    """
    This class represents a data type enum.
    """

    INT = 0
    FLOAT = 1
    STRING = 2
    BOOL = 3
    UNIT = 4
    FUNCTION = 5
    STRUCT_TYPE = 6
    STRUCT_INSTANCE = 7
    EXTERN_PYTHON = 8

    @staticmethod
    def name(code: int):
        if code == DType.INT:
            return "Int"
        elif code == DType.FLOAT:
            return "Float"
        elif code == DType.STRING:
            return "String"
        elif code == DType.BOOL:
            return "Bool"
        elif code == DType.UNIT:
            return "UnitType"
        elif code == DType.FUNCTION:
            return "Function"
        elif code == DType.STRUCT_TYPE:
            return "StructType"
        elif code == DType.STRUCT_INSTANCE:
            return "StructInstance"
        elif code == DType.EXTERN_PYTHON:
            return "ExternPython"
        else:
            raise ValueError(f"Unknown type {code}")

    def __repr__(self):
        return str(self)

    # rules for what type to return when performing an operation
    # on two objects
    # NOTE: if type is not in there, operation is unsupported
    _operation_result_rules = {
        OperationKind.ADD: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
            STRING: {
                STRING: STRING,
            },
        },
        OperationKind.SUB: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
        },
        OperationKind.MUL: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
                STRING: STRING,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
            STRING: {
                INT: STRING,
            },
        },
        OperationKind.DIV: {
            INT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
        },
        OperationKind.MOD: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
        },
        OperationKind.EQ: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            STRING: {
                STRING: BOOL,
            },
            BOOL: {
                BOOL: BOOL,
            },
        },
        OperationKind.NEQ: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            STRING: {
                STRING: BOOL,
            },
            BOOL: {
                BOOL: BOOL,
            },
        },
        OperationKind.LT: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.GT: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.LTE: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.GTE: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.OR: {
            BOOL: {
                BOOL: BOOL,
            },
        },
        OperationKind.AND: {
            BOOL: {
                BOOL: BOOL,
            },
        },
        OperationKind.BITOR: {
            INT: {
                INT: INT,
            },
        },
        OperationKind.BITAND: {
            INT: {
                INT: INT,
            },
        },
        OperationKind.FLIP: {
            INT: INT,
        },
        OperationKind.NOT: {
            BOOL: BOOL,
        },
    }


class Object:
    """
    This class represents an object in the interpreter.

    Attributes:
        type: (DType) The type of the object (int, float, string, etc.)
        value: (Any) The value of the object.
    """

    def __init__(self, type: int, value: Any):
        self.type = type
        self.value = value

    def __str__(self):
        return f"Object::{DType.name(self.type)}(" + str(self.value) + ")"

    def __repr__(self):
        return "Object(" + str(self) + ")"

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return self.type == other.type and self.value == other.value

    def __bool__(self):
        return bool(self.value)

    def get(self, ident: AstIdentifier):
        return self.value.get(ident)

    def add(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.ADD, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value + other.value)

    def sub(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.SUB, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value - other.value)

    def mul(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.MUL, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value * other.value)

    def div(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.DIV, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value / other.value)

    def mod(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.MOD, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value % other.value)

    def eq(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.EQ, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value == other.value)

    def neq(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.NEQ, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value != other.value)

    def lt(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.LT, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value < other.value)

    def gt(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.GT, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value > other.value)

    def lte(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.LTE, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        result = self.value <= other.value
        return Object(return_type, self.value <= other.value)

    def gte(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.GTE, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value >= other.value)

    def bool_or(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.OR, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value or other.value)

    def bool_and(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.AND, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value and other.value)

    def bool_not(self):
        return_type = DType._operation_result_rules.get(OperationKind.NOT, {}).get(
            self.type
        )
        if return_type is None:
            raise TypeError(f"Unsupported operation for {self.type}")
        return Object(return_type, not self.value)

    def flip(self):
        return_type = DType._operation_result_rules.get(OperationKind.FLIP, {}).get(
            self.type
        )
        if return_type is None:
            raise TypeError(f"Unsupported operation for type {self.type}")
        return Object(return_type, ~self.value)

    def bitor(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.BITOR, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value | other.value)

    def bitand(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.BITAND, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value & other.value)


################# interpreter_vm.py ##################

from typing import Optional


class InterpreterVM:
    """
    This class represents the interpreter virtual machine.

    Attributes:
        ast: AST to interpret
        variables: variables to use in the interpreter
        return_value: return value of the interpreter (None if no return statement was executed)

    """

    def __init__(self, ast, variables: dict[AstIdentifier, Object] = None):
        if variables is None:
            variables = {}
        self.variables = variables
        self.ast = ast
        self.return_value: Optional[Object] = None

    def execute(self):
        for statement in self.ast.statements:
            if statement.type == AstStatementType.ASSIGNMENT:
                assert (
                    type(statement.value) == AstAssignment
                ), "Bug: statement.value is not AstAssignment"
                assignment = statement.value
                identifier: AstIdentifier = assignment.identifier
                value: Object = assignment.value.resolve(self.variables)
                self.variables[identifier] = value

            elif statement.type == AstStatementType.RETURN:
                assert (
                    type(statement.value) == AstReturn
                ), "Bug: statement.value is not AstReturn"
                expression = statement.value.value
                self.return_value = expression.resolve(self.variables)
                return

            elif statement.type == AstStatementType.CONTROL_FLOW:
                control_flow = statement.value
                for branch in control_flow.branches:
                    if branch.condition.resolve(self.variables):
                        interpreter_vm = InterpreterVM(branch.body, self.variables)
                        interpreter_vm.execute()
                        if interpreter_vm.return_value is not None:
                            self.return_value = interpreter_vm.return_value
                            return
                        break

            else:
                raise ValueError("Unknown statement type " + statement.type)


################# lcaml_builtins.py ##################


class Print(ExternPython):
    def execute(self, context, args) -> Object:
        print(*[arg.value for arg in args], sep='', end='')
        return Object(DType.UNIT, None)

    def __str__(self) -> str:
        return "Print()"


class PrintLn(ExternPython):
    def execute(self, context, args) -> Object:
        print(*[arg.value for arg in args], sep='')
        return Object(DType.UNIT, None)

    def __str__(self) -> str:
        return "PrintLn()"


class Input(ExternPython):
    def execute(self, context, args) -> Object:
        if len(args) != 1:
            raise ValueError("Input takes 1 argument: prompt (string)")
        prompt = args[0]
        return Object(DType.STRING, input(prompt))

    def __str__(self) -> str:
        return "Input()"


BUILTINS = {"print": Print, "println": PrintLn, "input": Input}


################# lcaml_lexer.py ##################

import re
from typing import List, Dict, Callable


class LexError(Exception):
    """
    This exception is raised when the lexer cannot find a matching pattern to form a token.
    """

    pass


################# lcaml_parser.py ##################

from typing import Union, List, Set


TokenStream = List[Token]




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
        value: Union[AstAssignment, AstReturn],
    ):
        self.type = type
        self.value = value

    def __str__(self):
        return "AstStatement(" + str(self.type) + ", " + str(self.value) + ")"



################# lcaml_utils.py ##################

from typing import List


TokenStream = List[Token]


def unreachable():
    raise Exception("unreachable")


def is_token_pair(token1: Token, token2: Token):
    return (
        (token1.type == TokenKind.LPAREN and token2.type == TokenKind.RPAREN)
        or (token1.type == TokenKind.LSQUARE and token2.type == TokenKind.RSQUARE)
        or (token1.type == TokenKind.LCURLY and token2.type == TokenKind.RCURLY)
    )




################# operation_kind.py ##################


################# parser_types.py ##################

from typing import List, Any, Tuple, Set


TokenStream = List[Token]




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
        all_symbols_used: Set[Variable] = set()
        # constants
        STATEMENT_END_TOKEN = Token(TokenKind.SEMICOLON, PhantomType())
        BODY_END_TOKEN = Token(TokenKind.RCURLY, PhantomType())
        CONDITION_END_TOKEN = Token(TokenKind.RPAREN, PhantomType())
        # construct artificial if expression for else using artificial stream of boolean true followed by semicolon
        ELSE_ARTIFICIAL_IF_EXPRESSION, _, _ = Expression.from_stream(
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

        expression, stream, symbols_used = Expression.from_stream(
            stream, syntax, CONDITION_END_TOKEN
        )
        stream.pop(0)  # remove RPAREN
        all_symbols_used.update(symbols_used)

        if stream.pop(0).type != TokenKind.LCURLY:  # check/remove LCURLY
            raise ValueError("Expected LCURLY after if statement")

        body, stream = split_at_context_end(stream, BODY_END_TOKEN)
        stream.pop(0)  # remove RCURLY
        body, symbols_used = Ast.from_stream(body, syntax)
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

            expression, stream, symbols_used = Expression.from_stream(
                stream, syntax, CONDITION_END_TOKEN
            )
            stream.pop(0)  # remove RPAREN
            all_symbols_used.update(symbols_used)

            if stream.pop(0).type != TokenKind.LCURLY:
                raise ValueError("Expected LCURLY after else if statement")

            body, stream = split_at_context_end(stream, BODY_END_TOKEN)
            stream.pop(0)  # remove RCURLY
            body, symbols_used = Ast.from_stream(body, syntax)
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
        body, symbols_used = Ast.from_stream(body, syntax)
        all_symbols_used.update(symbols_used)

        branch = AstControlFlowBranch(ELSE_ARTIFICIAL_IF_EXPRESSION, body)
        branches.append(branch)

        return cls(branches), stream, all_symbols_used


################# resolvable.py ##################

from typing import Dict


Context = Dict[AstIdentifier, Object]


class Resolvable:
    def resolve(self, context: Context) -> Object:
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()


################# token_type.py ##################

# NOTE: Cannot change to integer because these values are used
# NOTE: to extract the token type from the syntax object


class Token:
    """
    This class represents a token found in the code.

    Attributes:
        type: type of token as string
        value: value of token as string

    """

    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

    def __str__(self):
        return "Token(" + self.type + ", " + self.value + ")"

    def __repr__(self):
        return "Token(" + self.type + ", " + self.value + ")"

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.type == other.type and self.value == other.value



if __name__ == '__main__':
    for file in os.listdir("tests/end_to_end"):
        with open(f'tests/end_to_end/{file}', 'r') as f:
            code = f.read()
        print(f"Running test: {file}")
        interpreter = Interpreter(code)
        num_runs = 1  # 000
        time_taken = timeit(run, number=num_runs)
        # time_taken_python = timeit(python_run, number=num_runs)
        print(f"Time taken to run {num_runs} times: {time_taken} seconds [average {time_taken / num_runs}]")
        # print(f"Python took {time_taken_python} seconds [average {time_taken_python / num_runs}]")
