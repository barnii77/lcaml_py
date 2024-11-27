import sys
from typing import Optional
from .lcaml_utils import LCAML_RECURSION_LIMIT
from . import lcaml_lexer
from . import lcaml_parser
from . import lcaml_builtins
from . import interpreter_vm
from . import interpreter_types
from .interpreter_types import Object, DType
from . import lcaml_expression


def get_builtins():
    b = lcaml_builtins.module({})
    assert isinstance(b, interpreter_types.Object)
    assert isinstance(b.value, lcaml_expression.Table)
    return b.value.fields


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

    def __init__(
        self,
        code: str,
        syntax=None,
        file="<unknown>",
        line_callbacks=None,
        next_step_callbacks=None,
        enable_vm_callbacks=True,
        parent=None,
    ):
        if syntax is None:
            syntax = lcaml_lexer.Syntax()
        self.syntax = syntax
        self.code = code
        self.tokens = lcaml_lexer.Lexer(code, self.syntax, file)()
        self.ast = lcaml_parser.Parser(self.tokens, self.syntax, file, code)()
        self.line_callbacks = {} if line_callbacks is None else line_callbacks
        self.next_step_callbacks = (
            [] if next_step_callbacks is None else next_step_callbacks
        )
        self.parent = parent
        self.enable_vm_callbacks = enable_vm_callbacks
        self.vm = interpreter_vm.InterpreterVM(
            self.ast,
            parent=self,
            line_callbacks=self.line_callbacks,
            next_step_callbacks=self.next_step_callbacks,
            file=file,
            _causes_traceback_entry=True,
            _enable_vm_callbacks=enable_vm_callbacks,
        )

    def execute(self, variables: Optional[dict] = None):
        """
        Returns:
            Any: The return value of the code
        """
        if variables is None:
            self.vm.context = get_builtins()
        else:
            self.vm.context = variables
        self.vm.context[self.syntax._interpreter_intrinsic] = Object(DType.PY_OBJ, self)
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(LCAML_RECURSION_LIMIT)

        try:
            self.vm.execute()
        except Exception as e:
            if not hasattr(e, "__lcaml_traceback_info"):
                setattr(e, "__lcaml_traceback_info", [])
            getattr(e, "__lcaml_traceback_info").append(self)
            raise e

        sys.setrecursionlimit(recursion_limit)
        return self.vm.return_value
