import sys
from lcaml_utils import LCAML_RECURSION_LIMIT
import lcaml_lexer
import lcaml_parser
import lcaml_builtins
import interpreter_vm
from parser_types import AstIdentifier
from token_type import Token, TokenKind


def lcamlify_vars(variables: dict[str, object]) -> dict[AstIdentifier, object]:
    result = {}
    for name, value in variables.items():
        name_ast_id = AstIdentifier(Token(TokenKind.IDENTIFIER, name))
        result[name_ast_id] = lcamlify_vars(value) if isinstance(value, dict) else value
    return result


def get_builtins():
    return lcamlify_vars(lcaml_builtins.LML_EXPORTS)


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
    def __init__(self, code: str, syntax=None):
        if syntax is None:
            syntax = lcaml_lexer.Syntax()
        self.syntax = syntax
        self.tokens = lcaml_lexer.Lexer(code, self.syntax)()
        self.ast = lcaml_parser.Parser(self.tokens, self.syntax)()
        self.vm = interpreter_vm.InterpreterVM(self.ast)

    def execute(self, variables: dict = None):
        """

        Returns:
            Any: The return value of the code

        """
        if variables is None:
            self.vm.variables = get_builtins()
        else:
            self.vm.variables = variables
        recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(LCAML_RECURSION_LIMIT)
        try:
            self.vm.execute()
        except Exception as e:
            sys.setrecursionlimit(recursion_limit)
            raise e
        else:
            sys.setrecursionlimit(recursion_limit)
        return self.vm.return_value
