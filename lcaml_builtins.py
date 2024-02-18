import lcaml_expression
import interpreter_types
from extern_python import ExternPython


class Print(ExternPython):
    def execute(self, context, args) -> lcaml_expression.Object:
        print(*[arg.value for arg in args], sep='', end='')
        return lcaml_expression.Object(interpreter_types.DType.UNIT, None)

    def __str__(self) -> str:
        return "Print()"


class PrintLn(ExternPython):
    def execute(self, context, args) -> lcaml_expression.Object:
        print(*[arg.value for arg in args], sep='')
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


BUILTINS = {"print": Print, "println": PrintLn, "input": Input}
