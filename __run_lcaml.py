import os
import json
from lcaml_lexer import Syntax
import interpreter as interpreter_mod

# lcaml ffi module imports
import lcaml_builtins


def run(file, syntax_file, print_ret=False):
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
        with open(syntax_file, "r") as f:
            syntax = json.load(f)
    with open(file, "r") as f:
        code = f.read()
    if syntax is not None:
        syntax = Syntax(**syntax)
    interpreter = interpreter_mod.Interpreter(code, syntax)
    # fuse all module exports into one dict
    mod_vars = {
        name: thing
        for mod in map(lambda mod: mod.LML_EXPORTS, [])
        for name, thing in mod.items()
    }
    # put builtins directly into context and external python functions into __extern_py intrinsic
    variables = interpreter_mod.lcamlify_vars(
        {**lcaml_builtins.LML_EXPORTS, "__extern_py": mod_vars}
    )
    result = interpreter.execute(variables)
    if print_ret and result is not None:
        print("\n", result, sep='')


def main(file, syntax_file=None, print_ret=False):
    if os.path.isdir(file):
        # run all files in dir
        for f in os.listdir(file):
            if f.startswith("_"):
                continue
            print("\n----------------------\n")
            print(f"Running {f}")
            if f.endswith(".lml"):
                run(os.path.join(file, f), syntax_file, print_ret)
            print("\n----------------------")
    else:
        run(file, syntax_file, print_ret)


main("/home/david/projects/lcaml/lcaml_py/tests/end_to_end/a.lml", None, True)
