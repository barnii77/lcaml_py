import lcaml_lexer
import lcaml_parser
import lcaml_builtins
import interpreter_vm
from parser_types import AstIdentifier
from token_type import Token, TokenKind


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

    def execute(self):
        """

        Returns:
            Any: The return value of the code

        """
        self.vm.variables = {}
        for name, value in lcaml_builtins.BUILTINS.items():
            # construct the value (which is a class)
            name_ast_id = AstIdentifier(Token(TokenKind.IDENTIFIER, name))
            self.vm.variables[name_ast_id] = value()
        self.vm.execute()
        return self.vm.return_value
