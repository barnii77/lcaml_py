import argparse
import shutil
import multiprocessing as mp
import os
import sys
from typing import Optional

from lcaml_py.core.lcaml_lexer import Lexer as LCamlLexer, Syntax
from lcaml_py.core.lcaml_parser import Parser as LCamlParser
from lcaml_py.core.lcaml_utils import expect_only_expression, indent
from lcaml_py.core import lcaml_expression, lcaml_utils


format_with_black = False
format_at_all = False
try:
    import black

    format_with_black = True
except ImportError:
    try:
        import autopep8

        format_at_all = True
    except ImportError:
        pass

# TODO support import and import_glob better:
# - currently, if import is referenced via a variable and not import builtin directly, it will not work
# - import_glob is not supported at all
# TODO support automatic detection and compilation of dependencies


lcaml_module = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
lcaml_root = os.path.dirname(lcaml_module)
with open(os.path.join(lcaml_module, "core", "lpy_runtime.py")) as runtime:
    lpy_runtime = runtime.read()


def compile_lcaml(
    code: str,
    syntax: Optional["Syntax"] = None,
    format_output: Optional[str] = None,
) -> str:
    """
    Takes lcaml code and an optional syntax object and transpiles it to python code

    Args:
        code: the lcaml code
        syntax: optional Syntax object for the lexer
        format_output: optionally specifies what formatter to format result with, can be one of (None, "autopep8", "black")

    Returns:
        str: python code
    """
    if format_output not in (None, "autopep8", "black"):
        raise ValueError(f"Invalid value of format_output: {format_output}")
    if syntax is None:
        syntax = Syntax()
    tokens = LCamlLexer(code, syntax)()
    ast = LCamlParser(tokens, syntax)()
    template_params = (
        f"COMPILE_WITH_CONTEXT_LEAKING = {lcaml_expression.COMPILE_WITH_CONTEXT_LEAKING}\n"
        f"LCAML_RECURSION_LIMIT = {lcaml_utils.LCAML_RECURSION_LIMIT}\n"
    )
    generated_code = expect_only_expression(ast.to_python())
    python_code = (
        template_params
        + lpy_runtime
        + "\ndef module(_4083bec4fe9f2fc73612f03b):\n"
        + indent("_ad7aaf167f237a94dc2c3ad2.update(_4083bec4fe9f2fc73612f03b)\n\n")
        + indent(generated_code)
        + '\nif __name__ == "__main__":\n'
        + indent("module({})")
    )
    python_code = "\n".join(
        filter(lambda x: bool(x.rstrip()), python_code.split("\n"))
    )  # remove empty lines
    if format_output is not None:
        if format_with_black:
            python_code = black.format_str(
                python_code, mode=black.Mode({black.TargetVersion.PY311})
            )
        else:
            python_code = autopep8.fix_code(python_code)
    return python_code


def compile_lcaml_io_handled(
    lcaml_input_filepath: str,
    build_folder: str,
    syntax: Optional["Syntax"] = None,
    format_output: Optional[str] = None,
):
    _, lcaml_fn = os.path.split(lcaml_input_filepath)
    lcaml_fn_wo_ext, _ = os.path.splitext(lcaml_fn)
    py_filepath = os.path.join(build_folder, lcaml_fn_wo_ext + ".py")
    with open(lcaml_input_filepath, "r") as file:
        lcaml_code = file.read()

    python_code = compile_lcaml(lcaml_code, syntax, format_output)

    with open(py_filepath, "w") as file:
        file.write(python_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Files to transpile to python")
    parser.add_argument("-s", "--syntax", default=None, help="Syntax file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--format-autopep",
        action="store_true",
        help="If set, format the generated python code using autopep8",
    )
    group.add_argument(
        "--format",
        action="store_true",
        dest="format_advanced",
        help="If set, format the generated python code using black if it is installed, otherwise autopep8",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=None,
        help="Optionally make compiler use a threadpool (using multiprocessing) to compile multiple files at once",
    )
    parser.add_argument(
        "--no-context-leaking",
        action="store_true",
        help="If set, disable context leaking in generated python (better performance at the cost of discompatibility with some of lcamls quirks)",
    )
    args = parser.parse_args()
    if args.no_context_leaking:
        lcaml_expression.COMPILE_WITH_CONTEXT_LEAKING = False
    format_with_black = format_with_black and args.format_advanced
    format_at_all = format_at_all and (args.format_advanced or args.format_autopep)
    output_formatter = (
        "black" if format_with_black else "autopep8" if format_at_all else None
    )
    build_dir = os.path.join(lcaml_root, "build")
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    build_dir = os.path.join(build_dir, "compiled_lcaml")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)

    for file in args.files:
        if not os.path.exists(os.path.abspath(file)):
            raise FileNotFoundError(f"file {file} does not exist")
    if args.n_parallel and args.n_parallel > 1 and len(args.files) > 1:
        with mp.Pool(args.n_parallel) as pool:
            pool.starmap(
                compile_lcaml_io_handled,
                [
                    (os.path.abspath(file), build_dir, None, output_formatter)
                    for file in args.files
                ],
            )
    else:
        for file in args.files:
            compile_lcaml_io_handled(
                os.path.abspath(file),
                build_dir,
                None,
                output_formatter,
            )
