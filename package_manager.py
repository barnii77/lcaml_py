"""
this file contains the lcaml package manager built into the main interpreter script main.py
it uses the storage path of the main.py file to determine the installation path of lcaml
then, it creates a __modules folder if it doesnt already exist
there, the packages are installed
every lcaml project has to contain a json config file for the lcaml package manager
the file will be read, and the dependencies will be installed in __modules if they aren't already there.
then, the dependencies of the dependencies will be installed, and so on
this will be called from the main.py script
"""

import os
import subprocess
import warnings
import shutil
import json
import sys
from typing import List, Set


def get_lcaml_install_dir():
    path = os.path.abspath(sys.argv[0])  # __path__ ?
    path = os.path.dirname(path)
    return path


def get_lcaml_modules_dir():
    path = os.path.join(get_lcaml_install_dir(), "__modules")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_lcaml_package_dir(package_name):
    path = os.path.join(get_lcaml_modules_dir(), package_name)
    if not os.path.exists(path):
        return None
    return path


def get_lcaml_virtual_env_dir():
    path = os.path.join(get_lcaml_install_dir(), ".venv")
    if not os.path.exists(path):
        subprocess.run(["python", "-m", "venv", path])
    return path


def get_venv_activate_path(venv_path):
    if sys.platform.startswith("win"):
        return os.path.join(venv_path, "Scripts", "activate")
    return os.path.join(venv_path, "bin", "activate")


def install_package_flat(package_name, package_url, force_reinstall=False):
    """runs git clone using the package url and clones to the package_name's associated directory"""
    install_dir = get_lcaml_package_dir(package_name)
    if install_dir is None or force_reinstall:
        install_dir = os.path.join(get_lcaml_modules_dir(), package_name)
        if os.path.exists(install_dir):
            shutil.rmtree(install_dir, ignore_errors=False)
        print(
            "\033[38;5;10m",
            "Installing ",
            '"\033[31;1m',
            package_name,
            '\033[38;5;10m"',
            "...",
            "\033[0m",
            sep="",
        )
        subprocess.run(
            ["git", "clone", package_url, install_dir], cwd=get_lcaml_modules_dir()
        )
        if os.path.exists(os.path.join(install_dir, "requirements.txt")):
            # install python dependencies
            venv_path = get_lcaml_virtual_env_dir()
            # if unix-based, source the activate script, else use ./activate
            venv_activate_path = get_venv_activate_path(venv_path)
            venv_source_cmd = (
                "source " if not sys.platform.startswith("win") else ""
            ) + venv_activate_path
            requirements_file = os.path.join(install_dir, "requirements.txt")
            subprocess.run(
                f"{venv_source_cmd} && python -m pip install -r {requirements_file}",
                shell=True,
                executable="/bin/bash" if not sys.platform.startswith("win") else None,
            )
    if not os.path.exists(os.path.join(install_dir, "module.lml")):
        raise Exception(f"Package {package_name} does not contain a module.lml file.")
    if not os.path.exists(os.path.join(install_dir, "requirements.json")):
        raise Exception(
            f"Package {package_name} does not contain a requirements.json file."
        )


def install_package_recursive(package_name, package_url, force_reinstall=False):
    """runs git clone using the package url and clones to the package_name's associated directory"""
    install_dir = get_lcaml_package_dir(package_name)
    if install_dir is None or force_reinstall:
        install_dir = os.path.join(get_lcaml_modules_dir(), package_name)
        shutil.rmtree(install_dir, ignore_errors=False)
        subprocess.run(
            ["git", "clone", package_url, install_dir], cwd=get_lcaml_modules_dir()
        )

    if not os.path.exists(os.path.join(install_dir, "module.lml")):
        raise Exception(f"Package {package_name} does not contain a module.lml file.")
    if not os.path.exists(os.path.join(install_dir, "requirements.json")):
        raise Exception(
            f"Package {package_name} does not contain a requirements.json file."
        )

    # install dependencies of the package
    with open(os.path.join(install_dir, "requirements.json"), "r") as f:
        config = json.load(f)
        deps = config.get("dependencies", [])
    for dep in deps:
        install_package_recursive(dep["name"], dep["url"])


def install_dependencies(package_path: str) -> Set[tuple[str, str]]:
    with open(os.path.join(package_path, "requirements.json"), "r") as f:
        config = json.load(f)
        deps_json = config.get("dependencies", [])
    deps = [(d["name"], d["url"]) for d in deps_json]
    deps = recursive_dependencies(deps)
    for dep in deps:
        install_package_flat(dep[0], dep[1])
    return deps


def recursive_dependencies(
    deps: List[tuple[str, str]], _result=None
) -> Set[tuple[str, str]]:
    """Builds a dependency set of the dependencies of dependencies of ... recursively.
    Uses DFS to traverse the dependency tree and add to a shared set of dependencies."""
    if _result is None:
        _result = set()
    for dep in deps:
        if dep not in _result:
            _result.add(dep)
            package_path = get_lcaml_package_dir(dep[0])
            if not package_path:
                # try installing it
                warnings.warn(f"Package {dep[0]} not found. Attempting to install it.")
                install_package_flat(dep[0], dep[1])
                package_path = get_lcaml_package_dir(dep[0])
                if not package_path:
                    raise Exception(
                        f"Package {dep[0]} not found and could not be installed."
                    )
            req_path = os.path.join(package_path, "requirements.json")
            with open(req_path) as f:
                config = json.load(f)
            deps_json = config.get("dependencies", [])
            deps = [(d["name"], d["url"]) for d in deps_json]
            recursive_dependencies(deps, _result)
    return _result
