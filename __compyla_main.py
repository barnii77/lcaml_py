import sys
import subprocess
import warnings
import os


# TODO replace usages of lcaml builtins with calls to lpy_runtime functions
# TODO support import and import_glob better:
# - currently, if import is referenced via a variable and not import builtin directly, it will not work
# - import_glob is not supported at all
# TODO support automatic detection and compilation of dependencies
# TODO add mapping from lcaml builtins to python lcaml runtime functions
# TODO figure out how to transpile __extern_py accesses


if os.path.split(sys.argv[0])[1] != "__compyla_main.py":
    with open(sys.argv[0], "r") as this_file:
        code = this_file.read()
    super_path = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    new_fp = os.path.join(super_path, "__compyla_main.py")
    print(new_fp)
    with open(new_fp, "w") as file:
        file.write(code)
    warnings.warn("copied file to __compyla_main.py")
    subprocess.run(f"cd {super_path} && python {new_fp}", shell=True)
    sys.exit()

from core.lcaml_lexer import Lexer as LCamlLexer, Syntax
from core.lcaml_parser import Parser as LCamlParser
from core.lcaml_utils import expect_only_expression, indent


DUMP_TO_FILE = True


def main():
    # FIXME this is hacky for debugging
    super_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    with open(os.path.join(super_path, "__test.lml"), "r") as file:
        lcaml_code = file.read()
    with open(os.path.join(super_path, "core", "lpy_runtime.py")) as runtime:
        lpy_runtime = runtime.read() + "\n\n"
    syntax = Syntax()
    tokens = LCamlLexer(lcaml_code, syntax)()
    ast = LCamlParser(tokens, syntax)()
    python_code = (
        lpy_runtime + "\n"
        + "def module():\n"
        + indent(expect_only_expression(ast.to_python()))
        + '\nif __name__ == "__main__":\n'
        + indent("module()")
    )
    python_code = "\n".join(
        filter(lambda x: bool(x.rstrip()), python_code.split("\n"))
    )  # remove empty lines
    print(python_code)
    if DUMP_TO_FILE:
        with open(os.path.join(super_path, "__test.py"), "w") as file:
            file.write(python_code)


if __name__ == "__main__":
    main()
