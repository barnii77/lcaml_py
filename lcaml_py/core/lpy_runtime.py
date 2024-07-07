import sys as _518b67e652531c5fe7e25d6b
import os as _840a8dcfeae95966a870b0b5

_c96c6d5be8d08a12e7b5cdc1 = input
_ce953a0eb08246617b7f8494 = print
_6ee0eb490ff832101cf82a3d = set
_a330395cc0a53ad120773654 = list
_8c25cb3686462e9a86d2883c = str
_6ab47d70854a8c690a0c2035 = dict
_6da88c34ba124c41f977db66 = int
_76a7e234dc54e9c605b2cc9f = float
_b760f44fa5965c2474a3b471 = bool
_1303c06b0b014d0ce7b988ab = type
_4a11dbf5131539804348ceb5 = isinstance
_2269c0be009b610cfdbb8cfe = range
_71fa9faaa6f884aa11f4cea2 = len
_fb2c3dcbe52aa287c413879a = tuple
_2706c619fe73f0cf112473c6 = exec


def print(*args):
    _ce953a0eb08246617b7f8494(*args, sep="", end="")


def println(*args):
    _ce953a0eb08246617b7f8494(*args, sep="")


def input(prompt):
    return _c96c6d5be8d08a12e7b5cdc1(prompt)


def is_instance(table, struct_type):
    if not _4a11dbf5131539804348ceb5(table, _6ab47d70854a8c690a0c2035):
        raise TypeError(
            f"Expected table to be a dict, got {_1303c06b0b014d0ce7b988ab(table)}"
        )
    if not _4a11dbf5131539804348ceb5(struct_type, _6ee0eb490ff832101cf82a3d):
        raise TypeError(
            f"Expected struct_type to be a StructType, got {_1303c06b0b014d0ce7b988ab(struct_type)}"
        )
    return _6ee0eb490ff832101cf82a3d(table.keys()) == struct_type


def is_like(a, b):
    if _4a11dbf5131539804348ceb5(
        a, _6ab47d70854a8c690a0c2035
    ) and _4a11dbf5131539804348ceb5(b, _6ab47d70854a8c690a0c2035):
        return a.keys() == b.keys()
    elif not _4a11dbf5131539804348ceb5(
        a, _6ab47d70854a8c690a0c2035
    ) and not _4a11dbf5131539804348ceb5(b, _6ab47d70854a8c690a0c2035):
        return _1303c06b0b014d0ce7b988ab(a) == _1303c06b0b014d0ce7b988ab(b)
    return False


def string(x):
    try:
        return _8c25cb3686462e9a86d2883c(x)
    except ValueError:
        pass


def float(x):
    try:
        return _76a7e234dc54e9c605b2cc9f(x)
    except ValueError:
        pass


def int(x):
    try:
        return _6da88c34ba124c41f977db66(x)
    except ValueError:
        pass


def bool(x):
    try:
        return _b760f44fa5965c2474a3b471(x)
    except ValueError:
        pass


def get(thing, key):
    return thing[key] if key in thing else None


def set(thing, key, value):
    thing[key] = value


def list(thing):
    return _a330395cc0a53ad120773654(thing)


def join(list_of_strings, join_elem):
    return join_elem.join(list_of_strings)


def len(thing):
    return _71fa9faaa6f884aa11f4cea2(thing)


def keys(thing):
    return thing.keys()


def values(thing):
    return thing.values()


def append(list, thing):
    list.append(thing)


def pop(list, index):
    list.pop(index)


def fuse(thing1, thing2):
    return {**thing1, **thing2}


def import_lcaml(path: str):
    path_without_ext, ext = _840a8dcfeae95966a870b0b5.path.splitext(path)
    if ext != "lml":
        raise ValueError("Invalid path: file must have .lml extension")
    return import_py(path_without_ext + ".py")


def import_py(path: str):
    if not _840a8dcfeae95966a870b0b5.path.exists(path):
        raise ValueError(f"Invalid path: path doesn't exist")
    path_without_ext, ext = _840a8dcfeae95966a870b0b5.path.splitext(path)
    if ext != "py":
        raise ValueError(f"Invalid path: file must have .py extension, but has .{ext}")
    if "." in path_without_ext:
        raise ValueError(
            "Disallowed path: path must not contain any dots (except before extension of file)"
        )
    python_path = (
        path_without_ext.replace("\\\\", ".").replace("\\", ".").replace("/", ".")
    )
    _2706c619fe73f0cf112473c6(f"import {python_path} as mod")
    return mod.module()


def exit(code):
    _518b67e652531c5fe7e25d6b.exit(code)
