from lcaml_lexer import Lexer, Syntax
from lcaml_parser import Parser
from interpreter import InterpreterVM


if __name__ == '__main__':
    code = """
let f = |x| {
    let y = x + 1;
    return y;
};

let x = 10;
let y = f (x + 4);
"""
    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    parser = Parser(tokens, syntax)
    ast = parser()
    interpreter = InterpreterVM(ast, {})
    interpreter.execute()
    print(code)
    print()
    for k, v in interpreter.variables.items():
        print(f"{k.name} = {v}")
