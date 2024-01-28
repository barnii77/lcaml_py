from ast_related import AstRelated
from token_type import Token, TokenKind
from lcaml_utils import PhantomType
from interpreter_types import Object
from parser_types import AstIdentifier
from interpreter import InterpreterVM
from typing import List, Dict


TokenStream = List[Token]


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


def is_token_pair(token1: Token, token2: Token):
    return (
        (token1.type == TokenKind.LPAREN and token2.type == TokenKind.RPAREN)
        or (token1.type == TokenKind.LSQUARE and token2.type == TokenKind.RSQUARE)
        or (token1.type == TokenKind.LCURLY and token2.type == TokenKind.RCURLY)
    )


class Resolvable:
    def resolve(self, context: Dict[AstIdentifier, Object]):
        """
        This function resolves the value of the expression.
        """
        raise NotImplementedError()


class Function(AstRelated, Resolvable):
    """

    Attributes:
        ast: (lcaml_parser.Ast) the AST that makes up the function
        arguments: the arguments of the function

    """

    def __init__(self, body, arguments: List[AstIdentifier]):
        self.body = body
        self.arguments = arguments

    def resolve(self, context: Dict[AstIdentifier, Object]):
        # if this is called, it's probably trying to resolve identifier but actually already has function
        return self


class Operation(AstRelated, Resolvable):
    def __init__(self, left: Resolvable | None, operation: Token, right: Resolvable):
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

    def resolve(self, context: Dict[AstIdentifier, Object]):
        """
        This function evaluates the operation for values of left and right
        """
        if self.left is None:
            left = None
        else:
            left = self.left.resolve(context)

        right = self.right.resolve(context)
        if self.operation == OperationKind.ADD:
            return left + right
        elif self.operation == OperationKind.SUB:
            return left - right
        elif self.operation == OperationKind.MUL:
            return left * right
        elif self.operation == OperationKind.DIV:
            return left / right
        elif self.operation == OperationKind.MOD:
            return left % right
        elif self.operation == OperationKind.NOT:
            return not right
        elif self.operation == OperationKind.EQ:
            return left == right
        elif self.operation == OperationKind.NEQ:
            return left != right
        elif self.operation == OperationKind.LT:
            return left < right
        elif self.operation == OperationKind.GT:
            return left > right
        elif self.operation == OperationKind.FLIP:
            return -right
        elif self.operation == OperationKind.LTE:
            return left <= right
        elif self.operation == OperationKind.GTE:
            return left >= right
        elif self.operation == OperationKind.OR:
            return left or right
        elif self.operation == OperationKind.AND:
            return left and right
        elif self.operation == OperationKind.BITOR:
            return left | right
        elif self.operation == OperationKind.BITAND:
            return left & right
        else:
            raise ValueError(f"Unknown operation type {self.operation}")


class FunctionCall(AstRelated, Resolvable):
    def __init__(self, function: Resolvable, arguments: List):
        """
        Resolved by spawning a new InterpreterVM

        Args:
            identifier: Identifier of function
            arguments: List[Expression]
        """
        self.function = function
        self.arguments = arguments

    def __str__(self):
        return "FunctionCall(" + str(self.function) + ", " + str(self.arguments) + ")"

    def resolve(self, context: Dict[AstIdentifier, Object]):
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
        function = self.function.resolve(context)
        if not isinstance(function, Function):
            raise TypeError("Cannot call non-function")

        # resolve args and make arg locals
        resolved_args = [arg.resolve(context) for arg in self.arguments]
        arg_locals = zip(function.arguments, resolved_args)
        # create local context
        local_context = context.copy()
        # overwrite variables from outer context with local args
        local_context.update(arg_locals)
        # spawn new interpreter vm
        interpreter_vm = InterpreterVM(
            function.body, dict(zip(function.arguments, resolved_args))
        )
        interpreter_vm.execute()
        return interpreter_vm.return_value


class Variable(AstRelated, Resolvable):
    def __init__(self, identifier: AstIdentifier):
        self.identifier = identifier

    def __str__(self):
        return "Variable(" + str(self.identifier) + ")"

    def resolve(self, context: Dict[AstIdentifier, Object]):
        return context[self.identifier]


class Constant(AstRelated, Resolvable):
    """

    Attributes:
        value: value of the constant (int, float, str, bool)

    """

    def __init__(self, token: Token):
        if token.type == TokenKind.INTEGER:
            try:
                value = int(token.value)
            except ValueError:
                raise ValueError(f"Invalid integer {token.value}")
        elif token.type == TokenKind.FLOATING_POINT:
            try:
                value = float(token.value)
            except ValueError:
                raise ValueError(f"Invalid floating point {token.value}")
        elif token.type == TokenKind.STRING_LITERAL:
            value = token.value
        elif token.type == TokenKind.BOOLEAN:
            try:
                value = bool(token.value)
            except ValueError:
                raise ValueError(f"Invalid boolean {token.value}")
        else:
            raise ValueError(f"Invalid type for constant: {token.type}")

        self.value = value

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
    def _build_from(cls, stream: TokenStream):
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
        first_pass_buffer = []
        # first pass across data: make operations without parameters and parse tokens into constants, variables or Exressions
        # NOTE: this pass will not detect function calls
        while stream:
            # NOTE: inplace pop is fine because this is only called on expression_stream
            token = stream.pop(0)

            if token.type == TokenKind.OPERATOR:
                op = Operation(None, token, None)
                first_pass_buffer.append(op)
            elif (
                token.type == TokenKind.INTEGER
                or token.type == TokenKind.FLOATING_POINT
                or token.type == TokenKind.STRING_LITERAL
                or token.type == TokenKind.BOOLEAN
            ):
                first_pass_buffer.append(Constant(token))
            elif token.type == TokenKind.IDENTIFIER:
                first_pass_buffer.append(Variable(AstIdentifier(token)))
            elif token.type == TokenKind.LPAREN:
                expression, stream = Expression.from_stream(
                    stream, Token(TokenKind.RPAREN, PhantomType())
                )
                first_pass_buffer.append(expression)
                stream.pop(0)  # remove RPAREN
            elif token.type == TokenKind.LCURLY:
                raise NotImplementedError()
            elif token.type == TokenKind.LSQUARE:
                raise NotImplementedError()
            else:
                raise ValueError(f"Unexpected token {token}")

        # second pass: identify function calls
        second_pass_buffer = []
        # NOTE: will consume tokens from first_pass_buffer (by popping)
        while first_pass_buffer:
            # can be operation, constant, variable or expression
            thing = first_pass_buffer.pop(0)

            if type(thing) in (Variable, Expression):
                # might be function call
                # check the following token(s)
                if not first_pass_buffer:
                    # EOS
                    second_pass_buffer.append(thing)
                    break
                next_thing = first_pass_buffer[0]
                if type(next_thing) in (Variable, Expression):
                    # function call
                    function_call = FunctionCall(thing, [])
                    while first_pass_buffer and type(first_pass_buffer[0]) in (
                        Variable,
                        Expression,
                    ):  # NOTE: safe to run because python interpreter will short circuit type checking
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
        return cls(expression)

    @classmethod
    def from_stream(
        cls,
        stream: TokenStream,
        terminating_token: Token = Token(
            TokenKind.SEMICOLON, PhantomType()
        ),  # use PhantomType so it matches anything of that type
    ):
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
        if terminating_token not in stream:
            raise ValueError(f"Expression must end with a {terminating_token}")
        # go through stream to identify terminating token (which might not be it's first occurance because that occurance may be linked to another inner expression)
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

        # exclude terminating and starting token
        expression_stream = stream[:terminating_idx]
        # leave terminating token in remaining stream so the parser can be sure syntax is valid
        remaining_stream = stream[terminating_idx:]

        return cls._build_from(expression_stream), remaining_stream
