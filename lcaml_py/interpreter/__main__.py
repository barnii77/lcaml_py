import argparse
import json
import os
import sys
from typing import Set
import warnings

from lcaml_py.core import interpreter as interpreter_mod
from lcaml_py.core import interpreter_vm as interpreter_vm_mod
from lcaml_py.core import lcaml_expression
from lcaml_py.core.lcaml_lexer import Syntax
from lcaml_py.core.lcaml_utils import get_marked_code_snippet
from . import package_manager as pm


def get_lcaml_traceback(exc: Exception) -> str:
    tb_lines = []
    tb_lines.append("LCaml Traceback (most recent call last):")
    if not hasattr(exc, "__lcaml_traceback_info"):
        print("Unable to construct lcaml traceback.")
    tb_info = getattr(exc, "__lcaml_traceback_info")
    code_lines = None
    for loc in reversed(tb_info):
        if isinstance(loc, interpreter_mod.Interpreter):
            tb_lines.append(f"In file {loc.vm.file}:\n")
            code_lines = loc.code.splitlines()
        elif isinstance(loc, interpreter_vm_mod.InterpreterVM):
            tb_lines.append(f"On line {loc.statement_line}:")
            tb_lines.append(
                get_marked_code_snippet(code_lines, loc.statement_line - 1, 3)
                if code_lines is not None
                else "<code unavailable>"
            )
            tb_lines.append("")
        elif isinstance(loc, int):
            tb_lines.append(f"On line {loc}:")
            tb_lines.append(
                get_marked_code_snippet(code_lines, loc - 1, 3)
                if code_lines is not None
                else "<code unavailable>"
            )
            tb_lines.append("")
        elif isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], str) and isinstance(loc[1], str):
            file, code = loc
            tb_lines.append(f"In file {file}:\n")
            code_lines = code.splitlines()
        elif isinstance(loc, str):
            tb_lines.append("Note: " + loc + "\n")
        else:
            raise TypeError("Invalid traceback entry encountered.")

    # this chaos below improves error output format by trying to parse `repr(exc)` to some degree
    r = repr(exc)
    exc_name = ""
    while r[0].isalnum():
        exc_name += r[0]
        r = r[1:]
    if r.startswith("('") or r.startswith('("'):
        r = r[2:]
    elif r.startswith('('):
        r = r[1:]
    if r.endswith("')") or r.endswith('")'):
        r = r[:-2]
    elif r.endswith(')'):
        r = r[:-1]
    if r:
        r = ": " + r
    out = exc_name + r
    if out:
        out = "Raised " + out
    tb_lines.append(out)
    return "\n".join(tb_lines)


def run(
    file, syntax_file, print_ret=False, enable_vm_callbacks=True, lcaml_tracebacks=True
):
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
    try:
        interpreter = interpreter_mod.Interpreter(
            code, syntax, file, enable_vm_callbacks=enable_vm_callbacks
        )
        variables = interpreter_mod.get_builtins()
        result = interpreter.execute(variables)
    except Exception as e:
        if lcaml_tracebacks:
            print(get_lcaml_traceback(e))
        else:
            raise e
    else:
        if print_ret and result is not None:
            print("\n", result, sep="")


def main(
    file,
    syntax_file=None,
    print_ret=False,
    enable_vm_callbacks=True,
    lcaml_tracebacks=True,
):
    if os.path.isdir(file):
        # run all files in dir
        for f in os.listdir(file):
            if f.startswith("_"):
                continue
            print("\n----------------------\n")
            print(f"Running {f}")
            if f.endswith(".lml"):
                run(
                    os.path.join(file, f),
                    syntax_file,
                    print_ret,
                    enable_vm_callbacks,
                    lcaml_tracebacks,
                )
            print("\n----------------------")
    else:
        run(file, syntax_file, print_ret, enable_vm_callbacks, lcaml_tracebacks)


def has_module_py(dep):
    package_dir = pm.get_lcaml_package_dir(dep[0])
    assert (
        package_dir
    ), "Bug in package manager. Please report. Directory missing but not detected as missing in checks."
    return os.path.exists(os.path.join(package_dir, "_module.py"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", default=None, help="File to run")
    parser.add_argument("-s", "--syntax", default=None, help="Syntax file")
    group.add_argument(
        "-i", "--install-deps", default=None, help="Install all dependencies"
    )
    group.add_argument("-a", "--add-dep", default=None, help="Add dependency")
    group.add_argument("-r", "--remove-dep", default=None, help="Remove dependency")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, print return value of file",
    )
    parser.add_argument(
        "--no-vm-callbacks",
        action="store_true",
        help="If set, turns off vm callbacks, which shaves off some overhead but removes the underlying mechanism used by the breakpoint() function builtin",
    )
    parser.add_argument(
        "--no-lcaml-tracebacks",
        action="store_true",
        help="If set, turns off lcaml tracebacks and instead propagates through the python exceptions",
    )
    parser.add_argument(
        "--jit-by-default",
        action="store_true",
        help="If set, enables jit compilation by default and uses interpreter as fallback if jitting fails",
    )
    parser.add_argument(
        "--suppress-jit",
        action="store_true",
        help="If set, suppresses the jit compiler completely and prevents jit compilation even of functions where it is forced (using the `jit` builtin)",
    )
    parser.add_argument(
        "--debug-print-unoptimized-llvm-ir",
        action="store_true",
        help="If set, the pre-optimization LLVM IR generated by the jit compiler is printed",
    )
    parser.add_argument(
        "--debug-print-optimized-llvm-ir",
        action="store_true",
        help="If set, the optimized LLVM IR generated by the jit compiler is printed",
    )
    parser.add_argument(
        "--jit-opt-level",
        type=int,
        help="Allows you to set the LLVM opt level used by the JIT compiler's optimization pipeline. Defaults to `2`, which should basically always be enough since the JIT compiler produces very simple IR where no super advanced passes should be needed.",
        default=2,
    )
    args = parser.parse_args()

    if args.file:
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
            raise FileNotFoundError(
                "requirements.json not found in any parent directory."
            )
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

        modules_before_install = set(os.listdir(pm.get_lcaml_modules_dir()))
        deps_set: Set[tuple[str, str]] = pm.install_dependencies(package_path)
        if not all(map(lambda dep: pm.get_lcaml_package_dir(dep[0]), deps_set)):
            raise RuntimeError("One or more dependencies failed to install.")
        modules_after_install = set(os.listdir(pm.get_lcaml_modules_dir()))

        py_deps_set = filter(has_module_py, deps_set)

        if modules_after_install != modules_before_install:
            # probably some logs, separate from the main output
            print("+----------------------+ +-+ +----------------------+")
        lcaml_expression.JIT_BY_DEFAULT = (
            lcaml_expression.JIT_BY_DEFAULT or args.jit_by_default
        )
        lcaml_expression.SUPPRESS_JIT = args.suppress_jit
        lcaml_expression.JIT_OPT_LEVEL = args.jit_opt_level
        lcaml_expression.DEBUG_PRINT_OPTIMIZED_LLVM_IR = args.debug_print_optimized_llvm_ir
        lcaml_expression.DEBUG_PRINT_UNOPTIMIZED_LLVM_IR = args.debug_print_unoptimized_llvm_ir
        main(
            file_path,
            syntax_path,
            args.debug,
            not args.no_vm_callbacks,
            not args.no_lcaml_tracebacks,
        )

    elif args.install_deps:
        # install dependencies listed in args.install_deps (file path)
        if not os.path.exists(args.install_deps):
            raise FileNotFoundError(f"File {args.install_deps} not found.")
        if not args.install_deps.endswith("requirements.json"):
            raise Exception("Please provide a .json file for dependencies.")
        package_path = os.path.dirname(args.install_deps)
        venv_path = pm.get_lcaml_virtual_env_dir()
        # if unix-based, source the activate script, else use ./activate
        venv_activate_path = pm.get_venv_activate_path(venv_path)
        venv_source_cmd = (
            "source " if not sys.platform.startswith("win") else ""
        ) + venv_activate_path

        modules_before_install = set(os.listdir(pm.get_lcaml_modules_dir()))
        deps_set: Set[tuple[str, str]] = pm.install_dependencies(package_path)

    elif args.add_dep:
        # edit requirements.json to add dep
        requirements_path = os.path.join(os.getcwd(), "requirements.json")
        if not os.path.exists(requirements_path):
            raise FileNotFoundError("requirements.json not found in current directory.")
        with open(requirements_path, "r") as f:
            config = json.load(f)
            deps = config.get("dependencies", [])
        dep = args.add_dep.split(" ")
        if len(dep) not in (1, 2):
            raise ValueError(
                "Please provide a url and optionally a name for the dependency."
            )
        url = dep[0]
        if len(dep) == 1:
            # name of package is the same as the url ending (without .git)
            name = url.split("/")[-1].split(".")[0]
            dep = (url, name)
        deps.append({"name": dep[1], "url": dep[0]})
        with open(requirements_path, "w") as f:
            config["dependencies"] = deps
            json.dump(config, f, indent=2)

    elif args.remove_dep:
        # similar to add_dep but remove
        requirements_path = os.path.join(os.getcwd(), "requirements.json")
        if not os.path.exists(requirements_path):
            raise FileNotFoundError("requirements.json not found in current directory.")
        with open(requirements_path, "r") as f:
            config = json.load(f)
            deps = config.get("dependencies", [])
        dep = args.remove_dep.split(" ")
        if len(dep) != 1:
            raise ValueError("Please provide a url or a name for the dependency.")
        dep_data = {"name": None, "url": None}
        if "/" in dep[0]:
            dep_data["url"] = dep[0]
        else:
            dep_data["name"] = dep[0]

        def url_is_same(x):
            return dep_data["url"] is None or x["url"] == dep_data["url"]

        def name_is_same(x):
            return dep_data["name"] is None or x["name"] == dep_data["name"]

        matching_dep = list(filter(lambda x: url_is_same(x) and name_is_same(x), deps))
        if len(matching_dep) == 0:
            raise ValueError("No matching dependency found.")
        elif len(matching_dep) > 1:
            raise ValueError(
                "Multiple matching dependencies found. Please remove manually or provide url instead if you provided name."
            )
        # remove the first matching dep
        for i, d in enumerate(deps):
            if d == matching_dep[0]:
                deps.pop(i)
                break
        with open(requirements_path, "w") as f:
            config["dependencies"] = deps
            json.dump(config, f, indent=2)
