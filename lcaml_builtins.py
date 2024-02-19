import lcaml_expression
import interpreter_types
from extern_python import ExternPython


class Print(ExternPython):
    def execute(self, context, args) -> lcaml_expression.Object:
        print(*[arg.value for arg in args], sep="", end="")
        return lcaml_expression.Object(interpreter_types.DType.UNIT, None)

    def __str__(self) -> str:
        return "Print()"


class PrintLn(ExternPython):
    def execute(self, context, args) -> lcaml_expression.Object:
        print(*[arg.value for arg in args], sep="")
        return lcaml_expression.Object(interpreter_types.DType.UNIT, None)

    def __str__(self) -> str:
        return "PrintLn()"


class Input(ExternPython):
    def execute(self, context, args) -> lcaml_expression.Object:
        if len(args) != 1:
            raise ValueError("Input takes 1 argument: prompt (string)")
        prompt = args[0]
        return lcaml_expression.Object(interpreter_types.DType.STRING, input(prompt))

    def __str__(self) -> str:
        return "Input()"


class IsInstance(ExternPython):
    def execute(self, context, args) -> lcaml_expression.Object:
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
        return lcaml_expression.Object(
            interpreter_types.DType.BOOL,
            set(struct_type.fields) == set(struct_instance.fields.keys()),
        )

    def __str__(self) -> str:
        return "IsLike()"


class IsLike(ExternPython):
    def execute(self, context, args) -> lcaml_expression.Object:
        if len(args) != 2:
            raise ValueError("is_like takes 2 arguments: struct_instance, struct_type")
        a, b = args[0].value, args[1].value
        if isinstance(a, lcaml_expression.StructInstance) and isinstance(
            b, lcaml_expression.StructInstance
        ):
            return lcaml_expression.Object(
                interpreter_types.DType.BOOL,
                set(a.fields.keys()) == set(b.fields.keys()),
            )
        elif not isinstance(a, lcaml_expression.StructInstance) and not isinstance(
            b, lcaml_expression.StructInstance
        ):
            return lcaml_expression.Object(
                interpreter_types.DType.BOOL, a.dtype == b.dtype
            )
        return lcaml_expression.Object(interpreter_types.DType.BOOL, False)

    def __str__(self) -> str:
        return "IsLike()"


BUILTINS = {
    "print": Print,
    "println": PrintLn,
    "input": Input,
    "isinstance": IsInstance,
    "islike": IsLike,
    "nl": lambda: lcaml_expression.Object(interpreter_types.DType.STRING, "\n"),
}
