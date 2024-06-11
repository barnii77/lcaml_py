from pyffi import pyffi

import os
import json
import sys
from lcaml_lexer import Syntax
import interpreter as interpreter_mod

# TODO add hidden py object to Object class so that python objects like file handles can be stored (but not directly manipulated) in lcaml


@pyffi
def test_pyffi(n: int):
    print("wow call to python", n)
    return n + 99


def get_variables(others: dict):
    variables = interpreter_mod.lcamlify_vars({"test_pyffi": test_pyffi})
    variables.update(others)
    return variables


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        raise Exception("Please provide a file to run.")
    if len(sys.argv) > 2:
        syntax_file = sys.argv[2]
    else:
        syntax_file = None
    if not file.endswith(".lml"):
        raise Exception("Please provide a .lml file to run.")
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} not found.")
    syntax = None
    if syntax_file is not None:
        if not os.path.exists(syntax_file):
            raise FileNotFoundError(f"Syntax file {syntax_file} not found.")
        if not syntax_file.endswith(".json"):
            raise Exception("Please provide a .json file for syntax.")
        with open(syntax_file, 'r') as f:
            syntax = json.load(f)
    with open(file, 'r') as f:
        code = f.read()
    if syntax is not None:
        syntax = Syntax(**syntax)
    interpreter = interpreter_mod.Interpreter(code, syntax)
    variables = get_variables(interpreter_mod.get_builtins())
    ret = interpreter.execute(variables)
    print("---------------------")
    print("Interpreter returned:", ret)
