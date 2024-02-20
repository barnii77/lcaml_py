import lcaml_expression
import interpreter_types
from extern_python import ExternPython


class Print(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        print(*[arg.value for arg in args], sep="", end="")
        return interpreter_types.Object(interpreter_types.DType.UNIT, None)

    def __str__(self) -> str:
        return "Print()"


class PrintLn(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        print(*[arg.value for arg in args], sep="")
        return interpreter_types.Object(interpreter_types.DType.UNIT, None)

    def __str__(self) -> str:
        return "PrintLn()"


class Input(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 1:
            raise ValueError("Input takes 1 argument: prompt (string)")
        prompt = args[0]
        Print().execute({}, [prompt])
        return interpreter_types.Object(interpreter_types.DType.STRING, input())

    def __str__(self) -> str:
        return "Input()"


class IsInstance(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 2:
            raise ValueError("is_like takes 2 arguments: struct_instance, struct_type")
        struct_instance, struct_type = args[0].value, args[1].value
        if not isinstance(struct_instance, lcaml_expression.StructInstance):
            raise TypeError(
                f"Expected struct_instance to be a StructInstance, got {type(struct_instance)}"
            )
        if not isinstance(struct_type, lcaml_expression.StructType):
            raise TypeError(
                f"Expected struct_type to be a StructType, got {type(struct_type)}"
            )
        return interpreter_types.Object(
            interpreter_types.DType.BOOL,
            set(struct_type.fields) == set(struct_instance.fields.keys()),
        )

    def __str__(self) -> str:
        return "IsLike()"


class IsLike(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 2:
            raise ValueError("is_like takes 2 arguments: struct_instance, struct_type")
        a, b = args[0].value, args[1].value
        if isinstance(a, lcaml_expression.StructInstance) and isinstance(
                b, lcaml_expression.StructInstance
        ):
            return interpreter_types.Object(
                interpreter_types.DType.BOOL,
                set(a.fields.keys()) == set(b.fields.keys()),
            )
        elif not isinstance(a, lcaml_expression.StructInstance) and not isinstance(
                b, lcaml_expression.StructInstance
        ):
            return interpreter_types.Object(
                interpreter_types.DType.BOOL, a.dtype == b.dtype
            )
        return interpreter_types.Object(interpreter_types.DType.BOOL, False)

    def __str__(self) -> str:
        return "IsLike()"


class Float(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 1:
            raise ValueError("float_from_string takes 1 argument: string")
        try:
            res = interpreter_types.Object(
                interpreter_types.DType.INT, float(args[0].value)
            )
        except ValueError:
            res = interpreter_types.Object(
                interpreter_types.DType.UNIT, None
            )
        return res

    def __str__(self) -> str:
        return "Float()"


class Int(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 1:
            raise ValueError("int_from_float takes 1 argument: float")
        try:
            res = interpreter_types.Object(
                interpreter_types.DType.INT, int(args[0].value)
            )
        except ValueError:
            res = interpreter_types.Object(
                interpreter_types.DType.UNIT, None
            )
        return res

    def __str__(self) -> str:
        return "Int()"


class String(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 1:
            raise ValueError("string_from_float takes 1 argument: float")
        try:
            res = interpreter_types.Object(
                interpreter_types.DType.STRING, str(args[0].value)
            )
        except ValueError:
            res = interpreter_types.Object(
                interpreter_types.DType.UNIT, None
            )
        return res

    def __str__(self) -> str:
        return "String()"


class Bool(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 1:
            raise ValueError("bool_from_string takes 1 argument: string")
        try:
            res = interpreter_types.Object(
                interpreter_types.DType.BOOL, bool(args[0].value)
            )
        except ValueError:
            res = interpreter_types.Object(
                interpreter_types.DType.UNIT, None
            )
        return res

    def __str__(self) -> str:
        return "Bool()"


class Set(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 3:
            raise ValueError("set takes 1 argument: list")
        if not all(isinstance(arg, interpreter_types.Object) for arg in args):
            raise ValueError("set takes 1 argument: list")
        if not args[0].type == interpreter_types.DType.STRUCT_INSTANCE:
            raise ValueError("set takes 1 argument: list")
        table, key, value = args[0].value, args[1], args[2]
        table.fields[key] = value
        return interpreter_types.Object(
            interpreter_types.DType.UNIT, None
        )

    def __str__(self) -> str:
        return "Set()"


class Get(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 2:
            raise ValueError("get takes 2 arguments: table, key")
        if not all(isinstance(arg, interpreter_types.Object) for arg in args):
            raise ValueError("get takes 2 arguments: table, key")
        if not args[0].type == interpreter_types.DType.STRUCT_INSTANCE:
            raise ValueError("get takes 2 arguments: table, key")
        table, key = args[0].value, args[1]
        if key not in table.fields:
            return interpreter_types.Object(lcaml_expression.DType.UNIT, None)
        return table.fields[key]

    def __str__(self) -> str:
        return "Get()"


class List(ExternPython):
    """Under the hood this is just a struct instance with some methods"""

    def execute(self, context, args) -> interpreter_types.Object:
        if not args:
            raise RuntimeError("Internal error: no arguments passed to List() \
            (should not be detected as function call)")
        if not all(isinstance(arg, interpreter_types.Object) for arg in args):
            raise RuntimeError("Internal error: not all object wrapped in Object type")
        # if called only with unit type, just return empty list (call with 2 unit types and pop if you want 1 unit type)
        ret = {} if args[0].type == interpreter_types.DType.UNIT and len(args) == 1 else {
            interpreter_types.Object(interpreter_types.DType.INT, i): arg for i, arg in enumerate(args)
        }
        return interpreter_types.Object(
            interpreter_types.DType.STRUCT_INSTANCE,
            lcaml_expression.StructInstance(ret)
        )

    def __str__(self) -> str:
        return "List()"


class Append(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 2:
            raise ValueError("append takes 2 arguments: list, value")
        if not all(isinstance(arg, interpreter_types.Object) for arg in args):
            raise ValueError("append takes 2 arguments: list, value")
        if not args[0].type == interpreter_types.DType.STRUCT_INSTANCE:
            raise ValueError("append takes 2 arguments: list, value")
        if not all(isinstance(arg, interpreter_types.Object) for arg in args):
            raise RuntimeError("internal error")
        list_, value = args[0].value, args[1]
        list_.fields[len(list_.fields)] = value
        return interpreter_types.Object(
            interpreter_types.DType.UNIT, None
        )

    def __str__(self) -> str:
        return "Append()"


class Pop(ExternPython):
    def execute(self, context, args) -> interpreter_types.Object:
        if len(args) != 1:
            raise ValueError("append takes 2 arguments: list, index")
        if not all(isinstance(arg, interpreter_types.Object) for arg in args):
            raise ValueError("append takes 2 arguments: list, index")
        if not args[0].type == interpreter_types.DType.STRUCT_INSTANCE:
            raise ValueError("append takes 2 arguments: list, index")
        if not all(isinstance(arg, interpreter_types.Object) for arg in args):
            raise RuntimeError("internal error")
        list_ = args[0].value
        ret = list_.fields.pop(len(list_.fields) - 1)
        if not isinstance(ret, interpreter_types.Object):
            raise RuntimeError("internal error")
        return ret

    def __str__(self) -> str:
        return "Append()"


BUILTINS = {
    "print": Print,
    "println": PrintLn,
    "input": Input,
    "isinstance": IsInstance,
    "islike": IsLike,
    "nl": lambda: interpreter_types.Object(interpreter_types.DType.STRING, "\n"),
    "float": Float,
    "int": Int,
    "string": String,
    "bool": Bool,
    "set": Set,
    "get": Get,
    "list": List,
    "append": Append,
    "pop": Pop,
}
