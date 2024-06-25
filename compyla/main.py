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
    if os.path.split(sys.argv[1])[1] != "__compyla_main.py":
        with open(sys.argv[1], "r") as this_file:
            code = this_file.read()
        super_path = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1])))
        with open(super_path + "__compyla_main.py", "w") as file:
            file.write(code)
        warnings.warn("copied file to __compyla_main.py")
        subprocess.run(["python", super_path + "__compyla_main.py"])
    else:
        with open("__test.lml", "r") as file:
            lcaml_code = file.read()
        syntax = Syntax()
        tokens = LCamlLexer(lcaml_code, syntax)()
        ast = LCamlParser(tokens, syntax)()
        python_code = expect_only_expression(ast.to_python())
        print(python_code)
        if DUMP_TO_FILE:
            with open("__test.py", "w") as file:
                file.write(python_code)


if __name__ == "__main__":
    main()
