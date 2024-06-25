import argparse
import subprocess
import json
import os
import sys
import package_manager as pm
import warnings
from typing import Set


RUN_TEMPLATE = """\
import os
import json
from core.lcaml_lexer import Syntax
import core.interpreter as interpreter_mod

# lcaml ffi module imports
import core.lcaml_builtins
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
        for name, thing in (mod.items() if isinstance(mod, dict) else mod.fields.items())
    }
    # put builtins directly into context and external python functions into __extern_py intrinsic
    variables = interpreter_mod.lcamlify_vars(
        {**lcaml_builtins.LML_EXPORTS, "__extern_py": mod_vars}
    ).fields
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


main(r{{{2}}}, {{{3}}}, True)
"""


def format(string: str, *args):
    for i, a in enumerate(args):
        string = string.replace("{{{" + str(i) + "}}}", str(a))
    return string


def has_module_py(dep):
    package_dir = pm.get_lcaml_package_dir(dep[0])
    assert (
        package_dir
    ), "Bug in package manager. Please report. Directory missing but not detected as missing in checks."
    return os.path.exists(os.path.join(package_dir, "module.py"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    inner_group = group.add_mutually_exclusive_group()
    group.add_argument("file", nargs="?", default=None, help="File to run")
    inner_group.add_argument(
        "-f", "--file", dest="file", default=None, help="File to run"
    )
    parser.add_argument("-s", "--syntax", default=None, help="Syntax file")
    inner_group.add_argument(
        "-i", "--install-deps", default=None, help="Install all dependencies"
    )
    inner_group.add_argument("-a", "--add-dep", default=None, help="Add dependency")
    inner_group.add_argument(
        "-r", "--remove-dep", default=None, help="Remove dependency"
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
        # if there is a python component in the module, the exported stuff has to be in module.py, hence name + ".module"
        py_deps = ", ".join(
            map(lambda x: "__modules." + x[0].replace("-", "_") + ".module", py_deps_set)
        )

        code = format(
            RUN_TEMPLATE,
            f"import {py_deps}\n" if py_deps else "",
            py_deps,
            f'"{file_path}"',
            f'"{syntax_path}"' if syntax_path else None,
        )
        run_lcaml = os.path.join(pm.get_lcaml_install_dir(), "__run_lcaml.py")
        with open(run_lcaml, "w") as f:
            f.write(code)

        if modules_after_install != modules_before_install:
            # probably some logs, separate from the main output
            print("+----------------------+ +-+ +----------------------+")
        subprocess.run(
            f"{venv_source_cmd} && python {run_lcaml}",
            shell=True,
            executable="/bin/bash" if not sys.platform.startswith("win") else None,
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
        url_is_same = lambda x: dep_data["url"] is None or x["url"] == dep_data["url"]
        name_is_same = lambda x: dep_data["name"] is None or x["name"] == dep_data["name"]
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
