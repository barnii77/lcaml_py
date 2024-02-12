import expression
import interpreter_types
from extern_python import ExternPython


class Print(ExternPython):
    def execute(self, context, args) -> expression.Object:
        print(*[arg.value for arg in args])
        return expression.Object(interpreter_types.DType.UNIT, None)

    def __str__(self) -> str:
        return "Print()"


class Input(ExternPython):
    def execute(self, context, args) -> expression.Object:
        if len(args) != 1:
            raise ValueError("Input takes 1 argument: prompt (string)")
        prompt = args[0]
        return expression.Object(interpreter_types.DType.STRING, input(prompt))

    def __str__(self) -> str:
        return "Input()"


BUILTINS = {"print": Print, "input": Input}
