from lcaml_lexer import Lexer, Syntax
from lcaml_parser import Parser
from interpreter import InterpreterVM


if __name__ == '__main__':
    code = """
    let x = 10; -- x y z
    let y = 20;
    let z = x + y;
    """
    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    parser = Parser(tokens)
    ast = parser()
    variables = {}
    interpreter = InterpreterVM(ast, variables)
    interpreter.execute()
    print(variables)
