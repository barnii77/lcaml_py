import os
import json
import sys
from interpreter import Interpreter


def run():
    interpreter.vm.variables = {}
    result = interpreter.execute()
    print("interpreter returned: ", result)


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
    interpreter = Interpreter(code, syntax)
    interpreter.execute()
