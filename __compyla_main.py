import sys
import subprocess
import warnings
import os

from core.lcaml_lexer import Lexer as LCamlLexer, Syntax
from core.lcaml_parser import Parser as LCamlParser
from core.lcaml_utils import expect_only_expression


DUMP_TO_FILE = False


def main():
    # FIXME this is hacky for debugging
    with open("__test.lml", "r") as file:
        lcaml_code = file.read()
    # super_path = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    super_path = "/home/david/projects/lcaml/lcaml_py"
    with open(os.path.join(super_path, "core", "lpy_runtime.py")) as runtime:
        lpy_runtime = runtime.read() + "\n\n"
    syntax = Syntax()
    tokens = LCamlLexer(lcaml_code, syntax)()
    ast = LCamlParser(tokens, syntax)()
    python_code = lpy_runtime + "\n\n" + expect_only_expression(ast.to_python())
    python_code = "\n".join(
        filter(lambda x: bool(x.rstrip()), python_code.split("\n"))
    )  # remove empty lines
    print(python_code)
    if DUMP_TO_FILE:
        with open("__test.py", "w") as file:
            file.write(python_code)


if __name__ == "__main__":
    main()
