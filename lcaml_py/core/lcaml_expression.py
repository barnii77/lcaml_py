from . import lcaml_parser as lcaml_parser
from . import parser_types as parser_types
from . import interpreter_vm as interpreter_vm_mod
from . import extern_python as extern_python

from .resolvable import Resolvable
from .ast_related import AstRelated
from .lcaml_lexer import Syntax
from .token_type import Token, TokenKind
from .lcaml_utils import (
    PhantomType,
    split_at_context_end,
    EqualsAny,
    get_unique_name,
    indent,
    expect_only_expression,
)
from .interpreter_types import Object, DType
from .operation_kind import OperationKind
from typing import List, Dict, Optional, Set, Iterable, Any


TokenStream = List[Token]
Context = Dict["parser_types.AstIdentifier", Object]


SYMBOL_TO_OPKIND = {
    "+": OperationKind.ADD,
    "-": OperationKind.SUB,
    "**": OperationKind.POW,
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

OPKIND_TO_SYMBOL = {
    OperationKind.ADD: "+",
    OperationKind.SUB: "-",
    OperationKind.POW: "**",
    OperationKind.MUL: "*",
    OperationKind.DIV: "/",
    OperationKind.MOD: "%",
    OperationKind.NOT: "!",
    OperationKind.EQ: "==",
    OperationKind.NEQ: "!=",
    OperationKind.LT: "<",
    OperationKind.GT: ">",
    OperationKind.FLIP: "~",
    OperationKind.LTE: "<=",
    OperationKind.GTE: ">=",
    OperationKind.OR: "||",
    OperationKind.AND: "&&",
    OperationKind.BITOR: "|",
    OperationKind.BITAND: "&",
}


class Gettable:
    def get(self, ident: "parser_types.AstIdentifier") -> "Object":
        raise NotImplementedError()


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
        self,
        body,
        arguments: List["parser_types.AstIdentifier"],
        bounds: Iterable["parser_types.AstIdentifier"],
        syntax: Syntax = Syntax(),
    ):
        self.body = body
        self.arguments = arguments
        self._syntax = syntax
        if isinstance(bounds, dict):
            self.bounds: Dict["parser_types.AstIdentifier", Object] = bounds
        else:
            self.bounds: Dict["parser_types.AstIdentifier", Object] = {
                ident: None for ident in bounds
            }

    def __str__(self):
        return "Function(" + str(self.body) + ", " + str(self.arguments) + ")"

    def to_python(self):
        name = get_unique_name()
        args = [expect_only_expression(arg.to_python()) for arg in self.arguments]
        body_pre_insert, body_block, body_post_insert = self.body.to_python()
        function_def = (
            f"def {name}("
            + ", ".join(args)
            + f", {Syntax._this_keyword}):\n{indent(body_block)}\n{name}_self_referral_list = [_dc937b59892604f5a86ac969]"  # the random ident here is the runtime ident of python None
        )
        value = (
            "".join(f"lambda {arg}: " for arg in args)
            + f"{name}("
            + ", ".join(args)
            + f", {name}_self_referral_list[0])"
        )
        populate_self_referral_list = f"{name}_self_referral_list[0] = {value}"
        return (
            body_pre_insert + "\n" + function_def + "\n" + body_post_insert,
            value,
            populate_self_referral_list,
        )

    def resolve(self, context: Context) -> Object:
        # if this is called, it's probably trying to resolve identifier but actually already has function
        # use context to resolve bounds
        intersecting_keys = self.bounds.keys() & context.keys()
        for key in intersecting_keys:
            self.bounds[key] = context[key]
        ret = Object(DType.FUNCTION, self)
        this = parser_types.AstIdentifier(
            Token(TokenKind.IDENTIFIER, self._syntax._this_keyword)
        )
        self.bounds[this] = ret
        return ret

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
        # token.value = "|x y z|"
        # first, extract arguments
        first_token = stream.pop(0)
        if first_token.type != TokenKind.FUNCTION_ARGS:
            raise ValueError(
                f"Expected function arguments, got {first_token.type} instead"
            )

        identifiers_raw = map(str.strip, syntax._extract_fn_args(first_token.value))
        arguments = list(
            map(
                lambda raw_id: parser_types.AstIdentifier(
                    Token(TokenKind.IDENTIFIER, raw_id)
                ),
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
        body, symbols_used = lcaml_parser.Ast.from_stream(body_stream, syntax)
        return (
            cls(
                body,
                arguments,
                map(lambda variable: variable.identifier, symbols_used),
                syntax,
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

    def to_python(self):
        # FIXME actually, there can be a pre and post inserts. they should be \n joined and bubbled up
        if self.is_unary:
            pre_insert, expr, post_insert = self.right.to_python()
            return (
                pre_insert,
                f"{OPKIND_TO_SYMBOL[self.operation]}({expr})",
                post_insert,
            )
        else:
            assert self.left is not None
            pre_inserts, exprs, post_inserts = zip(
                *[value.to_python() for value in (self.left, self.right)]
            )
            pre_insert = "\n".join(pre_inserts)
            post_insert = "\n".join(post_inserts)
            left, right = exprs
            return (
                pre_insert,
                f"{left} {OPKIND_TO_SYMBOL[self.operation]} {right}",
                post_insert,
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
        elif self.operation == OperationKind.POW:
            return left.pow(right)
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


class StructTypeParseState:
    ExpectFieldOrEnd = 0
    ExpectCommaOrEnd = 1


class StructType(AstRelated, Resolvable):
    """

    Attributes:
        fields: the names of the fields of a struct

    """

    def __init__(self, fields: List["parser_types.AstIdentifier"]):
        self.fields = fields

    def __str__(self):
        return f"StructType({self.fields})"

    def to_python(self):
        return (
            "",
            "set(["
            + ", ".join(
                expect_only_expression(field.to_python()) for field in self.fields
            )
            + "])",
            "",
        )

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

        state = StructTypeParseState.ExpectFieldOrEnd
        fields: List["parser_types.AstIdentifier"] = []
        while stream:
            token = stream.pop(0)
            if state == StructTypeParseState.ExpectFieldOrEnd:
                if token.type == TokenKind.IDENTIFIER:
                    field = parser_types.AstIdentifier(token)
                    fields.append(field)
                    state = StructTypeParseState.ExpectCommaOrEnd
                elif token.type == TokenKind.RCURLY:
                    break
                else:
                    raise ValueError(f"Expected field, got {token}")
            elif state == StructTypeParseState.ExpectCommaOrEnd:
                if token.type == TokenKind.COMMA:
                    state = StructTypeParseState.ExpectFieldOrEnd
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


class ListParseState:
    ExpectExpressionOrEnd = 0
    ExpectCommaOrEnd = 1


class LList(AstRelated, Resolvable):
    """

    Attributes:
        type: currently not in use
        fields: the fields of the struct instance

    """

    def __init__(self, values: List[Resolvable], type: Optional[Resolvable] = None):
        self.type = type
        self.values = values

    def __str__(self):
        return f"List({self.values})"

    def to_python(self):
        pre_inserts, exprs, post_inserts = zip(
            *[value.to_python() for value in self.values]
        )
        pre_insert = "\n".join(pre_inserts)
        post_insert = "\n".join(post_inserts)
        return pre_insert, "[" + ", ".join(exprs) + "]", post_insert

    def resolve(self, context: Context):
        for i, expression in enumerate(self.values):
            self.values[i] = expression.resolve(context)
        return Object(DType.LIST, self)

    @classmethod
    def from_stream(cls, stream: TokenStream, syntax: Syntax = Syntax()):
        token = stream.pop(0)
        if token.type != TokenKind.LSQUARE:
            raise ValueError(f"expected lsquare, got {token}")
        state = ListParseState.ExpectExpressionOrEnd
        all_symbols_used = set()
        values = []
        # expression after : can either end with a } or a , and both cases should be handled
        _expression_terminating_token = Token(
            EqualsAny(TokenKind.COMMA, TokenKind.RSQUARE), PhantomType()
        )
        while stream:
            token = stream.pop(0)
            if token.type == TokenKind.RSQUARE:
                break
            if state == ListParseState.ExpectExpressionOrEnd:
                stream.insert(0, token)
                expression, stream, symbols_used = Expression.from_stream(
                    stream, syntax, terminating_token=_expression_terminating_token
                )
                all_symbols_used.update(symbols_used)
                values.append(expression)
                state = ListParseState.ExpectCommaOrEnd
            elif state == ListParseState.ExpectCommaOrEnd:
                if token.type == TokenKind.COMMA:
                    state = ListParseState.ExpectExpressionOrEnd
                elif token.type == TokenKind.RSQUARE:
                    break
                else:
                    raise ValueError(f"expected comma or rcurly, got {token}")
                state = ListParseState.ExpectExpressionOrEnd
        return cls(values), stream, all_symbols_used


class TableParseState:
    ExpectFieldOrEnd = 0
    ExpectColon = 1
    ExpectExpression = 2
    ExpectCommaOrEnd = 3


class Table(AstRelated, Resolvable, Gettable):
    """

    Attributes:
        type: currently not in use
        fields: the fields of the struct instance

    """

    def __init__(
        self, fields: Dict[Any, Resolvable], type: Optional[Resolvable] = None
    ):
        self.type = type
        self.fields = fields

    def __str__(self):
        return f"Table({self.fields})"

    def to_python(self):
        keys = self.fields.keys()
        pre_inserts, exprs, post_inserts = zip(
            *[value.to_python() for value in self.fields.values()]
        )
        pre_insert = "\n".join(pre_inserts)
        post_insert = "\n".join(post_inserts)
        return (
            pre_insert,
            "{"
            + ", ".join(f'"{key}": ' + expr for key, expr in zip(keys, exprs))
            + "}",
            post_insert,
        )

    def resolve(self, context: Context):
        for field, expression in self.fields.items():
            self.fields[field] = expression.resolve(context)
        return Object(DType.TABLE, self)

    def get(self, ident: "parser_types.AstIdentifier") -> Object:
        if not isinstance(ident, parser_types.AstIdentifier):
            raise TypeError(f"Expected parser_types.AstIdentifier, got {ident}")
        if ident.name not in self.fields:
            raise ValueError(f"Field {ident} not found in struct {self}")
        return self.fields[ident.name]

    @classmethod
    def from_stream(cls, stream: TokenStream, syntax: Syntax = Syntax()):
        token = stream.pop(0)
        if token.type != TokenKind.LCURLY:
            raise ValueError(f"expected lcurly, got {token}")
        state = TableParseState.ExpectFieldOrEnd
        field = None
        all_symbols_used = set()
        assignments = []
        # expression after : can either end with a } or a , and both cases should be handled
        _expression_terminating_token = Token(
            EqualsAny(TokenKind.COMMA, TokenKind.RCURLY), PhantomType()
        )
        while stream:
            token = stream.pop(0)
            # (or does it?) FIXME: this does not work for stacked structures (struct inside struct)
            if token.type == TokenKind.RCURLY:
                break
            if state == TableParseState.ExpectFieldOrEnd:
                if token.type != TokenKind.IDENTIFIER:
                    raise ValueError(f"expected colon, got {token}")
                field = parser_types.AstIdentifier(token)
                state = TableParseState.ExpectColon
            elif state == TableParseState.ExpectColon:
                if token.type != TokenKind.COLON:
                    raise ValueError(f"expected colon, got {token}")
                state = TableParseState.ExpectExpression
            elif state == TableParseState.ExpectExpression:
                if field is None:
                    raise ValueError("Interal Error or Syntax Error: Field is None")
                stream.insert(0, token)
                expression, stream, symbols_used = Expression.from_stream(
                    stream, syntax, terminating_token=_expression_terminating_token
                )
                all_symbols_used.update(symbols_used)
                assignments.append((field, expression))
                field = None
                state = TableParseState.ExpectCommaOrEnd
            elif state == TableParseState.ExpectCommaOrEnd:
                if token.type == TokenKind.COMMA:
                    state = TableParseState.ExpectFieldOrEnd
                elif token.type == TokenKind.RCURLY:
                    break
                else:
                    raise ValueError(f"expected comma or rcurly, got {token}")
                state = TableParseState.ExpectFieldOrEnd
        if state not in (
            TableParseState.ExpectCommaOrEnd,
            TableParseState.ExpectFieldOrEnd,
        ):
            raise ValueError("Unexpected end of tokenstream")
        assignments = dict(assignments)
        assignments = {
            field.name: expression for field, expression in assignments.items()
        }
        return cls(assignments), stream, all_symbols_used


class FieldAccess(AstRelated, Resolvable):
    """
    A class that represents a field access to a value. Can be stacked by having a FieldAccess as the object field of another FieldAccess

    Attributes:
        object: the resolvable (expression, variable, whatever) to access field from
        field: the field to look for and return

    """

    def __init__(self, object: "Resolvable", field: "parser_types.AstIdentifier"):
        """

        Args:
            object: the resolvable (expression, variable, whatever) to access field from
            field: the field to look for and return

        """
        self.object = object
        self.field = field

    def __str__(self):
        return "AstFieldAccess(" + str(self.object) + ", " + str(self.field) + ")"

    def to_python(self) -> tuple[str, str, str]:
        return (
            "",
            f'{expect_only_expression(self.object.to_python())}["{expect_only_expression(self.field.to_python())}"]',
            "",
        )

    def resolve(self, context: Context) -> Object:
        obj = self.object.resolve(context)
        if isinstance(
            obj, (Gettable, Object)
        ):  # Object is fine if wrapped value is gettable
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

    def __init__(
        self, fuction_resolvable: Resolvable, arguments: List, syntax: Syntax = Syntax()
    ):
        """
        Resolved by spawning a new interpreter_vm_mod.InterpreterVM

        Args:
            function_container: Object[Function]
            arguments: List[Expression]
        """
        self.function_resolvable = fuction_resolvable
        self.arguments = arguments
        self._syntax = syntax

    def __str__(self):
        return (
            "FunctionCall("
            + str(self.function_resolvable)
            + ", "
            + str(self.arguments)
            + ")"
        )

    def to_python(self) -> tuple[str, str, str]:
        f_pre_insert, f_expr, f_post_insert = self.function_resolvable.to_python()
        arg_pre_inserts, arg_exprs, arg_post_inserts = zip(
            *[arg.to_python() for arg in self.arguments]
        )
        pre_insert = "\n".join((*arg_pre_inserts, f_pre_insert))
        post_insert = "\n".join((f_post_insert, *arg_post_inserts))

        # NOTE: this calling method might seem weird, but for the sake of easy currying compilation, functions are transpiled as lambda chains and each arg needs to be provided individually
        f_expr = (
            "(" + f_expr + ")" + "".join("(" + arg_expr + ")" for arg_expr in arg_exprs)
        )
        return (
            pre_insert,
            f_expr,
            post_insert,
        )

    def resolve(self, context: Context) -> Optional[Object]:
        """
        Resolves the value of the function call by spawning a new interpreter_vm_mod.InterpreterVM

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
                result = Function(function.body, remaining_args, bounds, self._syntax)
                return Object(DType.FUNCTION, result)
            else:
                # all args provided -> execute function
                # create local context
                local_context = context.copy()
                # functions can bind to global values, example in docs
                # https://github.com/barnii77/lcaml_py/blob/main/docs/interpreter.md#functions
                non_none_bounds = {
                    k: v for k, v in function.bounds.items() if v is not None
                }
                local_context.update(non_none_bounds)
                # overwrite variables from outer context with local args
                local_context.update(arg_locals)
                # spawn new interpreter vm
                interpreter_vm = interpreter_vm_mod.InterpreterVM(
                    function.body, local_context
                )
                interpreter_vm.execute()
                return interpreter_vm.return_value

        elif isinstance(function, extern_python.ExternPython):
            return function.execute(context, resolved_args)

        elif hasattr(function, "execute"):
            print(
                "Warning: unregistered object type executed (function not registered as ExternPython, but has necessary API). "
                "Please report to developer."
            )
            return function.__class__.execute(function, context, resolved_args)

        else:
            raise TypeError("Cannot call non-function")


class Variable(AstRelated, Resolvable):
    def __init__(self, identifier: "parser_types.AstIdentifier"):
        self.identifier = identifier

    def __str__(self):
        return "Variable(" + str(self.identifier) + ")"

    def to_python(self):
        return "", expect_only_expression(self.identifier.to_python()), ""

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

    def to_python(self):
        if self.value.type == DType.STRING:
            value = '"' + str(self.value.value) + '"'
        elif self.value.type == DType.UNIT:
            value = "None"
        elif self.value.type == DType.BOOL:
            value = str(self.value.value).title()
        else:
            value = str(self.value.value)
        return "", value, ""

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
        return "Expression(" + str(self.expression) + ")"

    def to_python(self):
        return self.expression.to_python()

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
                3.1 . (field access)
                3.2. ! ~
                3.3 **
                3.4. * / % & |
                3.5. + -
                3.6 | &
                3.7. == != < > <= >=
                3.8. || &&

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
            elif token.type == TokenKind.DOT:
                fa = FieldAccess(None, None)  # reuse logic from Operation handling
                first_pass_buffer.append(fa)
            elif token.type in TokenKind._builtin_types:
                first_pass_buffer.append(Constant(token, syntax))
            elif token.type == TokenKind.IDENTIFIER:
                first_pass_buffer.append(Variable(parser_types.AstIdentifier(token)))
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
                    function_call = FunctionCall(thing, [], syntax)
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

        this_pass_buffer, prev_pass_buffer = [], second_pass_buffer
        # third set of passes: fill out operations with their args
        # NOTE: this is a sequence of related passes because ! ~ have to be done before * / %
        # NOTE: and * / % have to be done before + -
        # NOTE and + - have to be done before | & which have to be done before == != < > <= >=
        # NOTE: which have to be done before || &&

        while prev_pass_buffer:
            thing = prev_pass_buffer.pop(0)

            if isinstance(thing, FieldAccess):
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
                # if variable, unwrap to raw identifier
                if isinstance(right, Variable):
                    right = right.identifier
                thing.object = left
                thing.field = right

            this_pass_buffer.append(thing)
        # stage one of third pass (unary): ! ~
        this_pass_buffer, prev_pass_buffer = [], this_pass_buffer
        while prev_pass_buffer:
            thing = prev_pass_buffer.pop(0)

            if isinstance(thing, Operation):
                if thing.is_unary:
                    # NOTE: unary operands are always on the right
                    if not prev_pass_buffer:
                        raise ValueError("Unary operand must have right operand")
                    right = prev_pass_buffer.pop(0)
                    thing.right = right  # thing is unary operation

            this_pass_buffer.append(thing)

        # all the other passes (binary) are kind of the same
        sorted_pass_operations = (
            (OperationKind.POW,),
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
        elif stream[0].type == TokenKind.LCURLY:
            (
                table,
                remaining_stream,
                symbols_used,
            ) = Table.from_stream(stream, syntax)
            return cls(table), remaining_stream, symbols_used
        elif stream[0].type == TokenKind.LSQUARE:
            (
                lst,
                remaining_stream,
                symbols_used,
            ) = LList.from_stream(stream, syntax)
            return cls(lst), remaining_stream, symbols_used
        if terminating_token not in stream:
            raise ValueError(f"Expression must end with a {terminating_token}")

        # exclude terminating and starting token
        # leave terminating token in remaining stream so the parser can be sure syntax is valid
        expression_stream, remaining_stream = split_at_context_end(
            stream, terminating_token
        )
        if not expression_stream:
            raise ValueError("Empty expression")

        expression, symbols_used = cls._build_from(expression_stream, syntax)
        return expression, remaining_stream, symbols_used
