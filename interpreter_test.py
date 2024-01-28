from lcaml_lexer import Lexer, Syntax
from lcaml_parser import Parser
from interpreter import InterpreterVM


if __name__ == '__main__':
    code = """
let x = 11; -- x y z
let a = 5;
let y = x + a;
let abc = ((x - (3 | x * 2)) % a) * (y + 1);
let z = abc & 1;"""
    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    parser = Parser(tokens)
    ast = parser()
    interpreter = InterpreterVM(ast, {})
    interpreter.execute()
    print(code)
    print()
    for k, v in interpreter.variables.items():
        print(f"{k.name} = {v}")
