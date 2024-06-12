import argparse
import subprocess
import os
import sys
import package_manager as pm
import warnings


RUN_TEMPLATE = """\
import os
import json
from lcaml_lexer import Syntax
import interpreter as interpreter_mod

# lcaml ffi module imports
import lcaml_builtins
{{{0}}}

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
        for mod in map(lambda mod: mod.LML_EXPORTS, [{{{1}}}])
        for name, thing in mod.items()
    }
    # put builtins directly into context and external python functions into __extern_py intrinsic
    variables = interpreter_mod.lcamlify_vars(
        {**lcaml_builtins.LML_EXPORTS, "__extern_py": mod_vars}
    )
    result = interpreter.execute(variables)
    if print_ret and result is not None:
        print("\\n", result, sep='')


def main(file, syntax_file=None, print_ret=False):
    if os.path.isdir(file):
        # run all files in dir
        for f in os.listdir(file):
            if f.startswith("_"):
                continue
            print("\\n----------------------\\n")
            print(f"Running {f}")
            if f.endswith(".lml"):
                run(os.path.join(file, f), syntax_file, print_ret)
            print("\\n----------------------")
    else:
        run(file, syntax_file, print_ret)


main({{{2}}}, {{{3}}}, True)
"""


def format(string: str, *args):
    for i, a in enumerate(args):
        string = string.replace("{{{" + str(i) + "}}}", str(a))
    return string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", nargs="?", default="tests/end_to_end", help="File to run"
    )
    parser.add_argument(
        "-f", "--file", dest="file", default="tests/end_to_end", help="File to run"
    )
    # parser.add_argument("-d", "--deps", default=None, help="Python dependencies")
    parser.add_argument("-s", "--syntax", default=None, help="Syntax file")
    args = parser.parse_args()
    # get absolute paths of syntax, code and deps files
    syntax_path = os.path.abspath(args.syntax) if args.syntax else None
    # deps_path = os.path.abspath(args.deps) if args.deps else None
    file_path = os.path.abspath(args.file)
    package_path = os.path.dirname(file_path)
    requirements_path = os.path.join(package_path, "requirements.json")
    root_dir = os.path.abspath(os.sep)
    while not os.path.exists(requirements_path) and package_path != root_dir:
        package_path = os.path.dirname(package_path)
        requirements_path = os.path.join(package_path, "requirements.json")
    if not os.path.exists(requirements_path):
        raise FileNotFoundError("requirements.json not found in any parent directory.")
    elif os.getcwd() != package_path:
        warnings.warn(
            f"requirements.json not found in current working directory. Using one in {'child' if len(os.getcwd()) > package_path else 'parent'} dir instead."
        )

    # if deps_path:
    #     with open(deps_path, "r") as f:
    #         deps = ", ".join(filter(lambda x: bool(x), f.read().split("\n")))
    # else:
    #     deps = ""
    # run everything in special lcaml virtual env

    venv_path = pm.get_lcaml_virtual_env_dir()
    # if unix-based, source the activate script, else use ./activate
    venv_activate_path = pm.get_venv_activate_path(venv_path)
    venv_source_cmd = (
        "source " if not sys.platform.startswith("win") else ""
    ) + venv_activate_path

    deps_set = pm.install_dependencies(package_path)
    deps = ", ".join(map(lambda x: x[0].replace("-", "_"), deps_set))

    code = format(
        RUN_TEMPLATE,
        f"import {deps}\n" if deps else "",
        deps,
        f'"{file_path}"',
        f'"{syntax_path}"' if syntax_path else None,
    )
    run_lcaml = os.path.join(pm.get_lcaml_install_dir(), "__run_lcaml.py")
    with open(run_lcaml, "w") as f:
        f.write(code)
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    exit()
    subprocess.run(
        f"{venv_source_cmd} && python {run_lcaml}",
        shell=True,
        executable="/bin/bash" if not sys.platform.startswith("win") else None,
    )
