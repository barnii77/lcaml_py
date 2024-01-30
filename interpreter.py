import lcaml_lexer
import lcaml_parser
import interpreter_vm


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
    def __init__(self, code: str):
        self.syntax = lcaml_lexer.Syntax()
        self.tokens = lcaml_lexer.Lexer(code, self.syntax)()
        self.ast = lcaml_parser.Parser(self.tokens, self.syntax)()
        self.vm = interpreter_vm.InterpreterVM(self.ast)

    def execute(self):
        """

        Returns:
            Any: The return value of the code

        """
        self.vm.execute()
        return self.vm.return_value
