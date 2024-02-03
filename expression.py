import lcaml_parser

from ast_related import AstRelated
from lcaml_lexer import Syntax
from token_type import Token, TokenKind
from lcaml_utils import PhantomType, split_at_context_end
from interpreter_types import Object, DType
from parser_types import AstIdentifier
from interpreter_vm import InterpreterVM
from operation_kind import OperationKind
from typing import List, Dict, Optional, Set, Iterable


TokenStream = List[Token]


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


class Resolvable:
    def resolve(self, context: Dict[AstIdentifier, Object]) -> Object:
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()


class FunctionParseState:
    ExpectCommaOrEndOfArgs = 0
    ExpectArgumentOrEndOfArgs = 1


class Function(AstRelated, Resolvable):
    """

    Attributes:
        ast: (lcaml_parser.Ast) the AST that makes up the function
        arguments: the arguments of the function

    """

    def __init__(
        self, body, arguments: List[AstIdentifier], bounds: Iterable[AstIdentifier]
    ):
        self.body = body
        self.arguments = arguments
        self.bounds: Dict[AstIdentifier, Object] = {ident: None for ident in bounds}

    def __str__(self):
        return "Function(" + str(self.body) + ", " + str(self.arguments) + ")"

    def resolve(self, context: Dict[AstIdentifier, Object]) -> Object:
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
                f"Expected {{ for function body, got {first_body_token.type}"
            )

        body_stream, remaining_stream = split_at_context_end(
            stream, Token(TokenKind.RCURLY, PhantomType())
        )
        remaining_stream.pop(0)  # remove RCURLY
        body, symbols_used = lcaml_parser.Ast.from_stream(body_stream, syntax)
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

    def resolve(self, context: Dict[AstIdentifier, Object]) -> Object:
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

    def resolve(self, context: Dict[AstIdentifier, Object]) -> Optional[Object]:
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
            assert (
                function.type == DType.FUNCTION
            ), "Internal interpreter bug: Invalid object"
            function = function.value
        if not isinstance(function, Function):
            raise TypeError("Cannot call non-function")

        # resolve args and make arg locals
        resolved_args = [arg.resolve(context) for arg in self.arguments]
        arg_locals = zip(function.arguments, resolved_args)
        # create local context
        local_context = context.copy()
        # functions can bind to global values, example in docs
        # https://github.com/barnii77/lcaml_py/blob/main/docs/interpreter.md#functions
        local_context.update(function.bounds)
        # overwrite variables from outer context with local args
        local_context.update(arg_locals)
        # spawn new interpreter vm
        interpreter_vm = InterpreterVM(function.body, local_context)
        interpreter_vm.execute()
        return interpreter_vm.return_value


class Variable(AstRelated, Resolvable):
    def __init__(self, identifier: AstIdentifier):
        self.identifier = identifier

    def __str__(self):
        return "Variable(" + str(self.identifier) + ")"

    def resolve(self, context: Dict[AstIdentifier, Object]):
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

    def resolve(self, context: Dict[AstIdentifier, Object]):
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

    def resolve(self, context: Dict[AstIdentifier, Object]):
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
            # FIXME
            function, remaining_stream, symbols_used = Function.from_stream(
                stream, syntax
            )
            return cls(function), remaining_stream, symbols_used

        if terminating_token not in stream:
            raise ValueError(f"Expression must end with a {terminating_token}")

        # exclude terminating and starting token
        # leave terminating token in remaining stream so the parser can be sure syntax is valid
        expression_stream, remaining_stream = split_at_context_end(
            stream, terminating_token
        )

        expression, symbols_used = cls._build_from(expression_stream, syntax)
        return expression, remaining_stream, symbols_used
