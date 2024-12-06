# this is a template, undefined values will be inserted above automatically during compilation

from inspect import signature as _1a2fc26dc7ea5a2a4748b7cb
import sys as _518b67e652531c5fe7e25d6b
import os as _840a8dcfeae95966a870b0b5
import time as _336074805fc853987abe6f7f
from copy import deepcopy
from lcaml_py.core import pyffi as lcaml_pyffi_mod

_518b67e652531c5fe7e25d6b.setrecursionlimit(LCAML_RECURSION_LIMIT)
lcaml_pyffi_mod.COMPILE_WITH_CONTEXT_LEAKING = COMPILE_WITH_CONTEXT_LEAKING

_f676713508a2d8f20081f0fa = deepcopy
_2706c619fe73f0cf112473c6 = exec
_091d41be4c916bd540e8b292 = id
_78da4a596a88bc5114f071ba = abs
_c96c6d5be8d08a12e7b5cdc1 = input
_ce953a0eb08246617b7f8494 = print
_6ee0eb490ff832101cf82a3d = set
_a330395cc0a53ad120773654 = list
_6ab47d70854a8c690a0c2035 = dict
_6da88c34ba124c41f977db66 = int
_76a7e234dc54e9c605b2cc9f = float
_b760f44fa5965c2474a3b471 = bool
_4a11dbf5131539804348ceb5 = isinstance
_71fa9faaa6f884aa11f4cea2 = len
_dc937b59892604f5a86ac969 = None
_8c25cb3686462e9a86d2883c = str
_55df18d062878fb6d3f4a6d1 = ord
_943723cd5955a5316f4364f7 = chr
_252813b4c8c5b033423ac5ab = breakpoint

__compiled = True


def _cf6a7c2c6ee4aefed721396c(lamb, func):
    lamb.bounds = func.bounds
    return lamb


def _d6b5f6e2e1a192b4fd7cccb6(bounds):
    def decorator(func):
        class Wrapper:
            def __init__(self):
                self.bounds = bounds

            def __call__(self, context, *args):
                for k, v in self.bounds.items():
                    if v is not None:
                        context[k] = v
                return func(context, *args)

        return Wrapper()

    return decorator


def _6e8be7d82f8a724d77d4d12c(func):  # mark function as pyffi function
    setattr(func, "_e6c50da35e8f9284c183e69b", 0)
    return func


def _881ecbfb15f7e6881a337113(func):
    class BuiltinWrapper:
        def __init__(self):
            setattr(self, "_e6c50da35e8f9284c183e69b", 0)

        @staticmethod
        def __call__(*args):
            """
            Args:
                _: context that is ignored in builtins
                *args: args provided to the function
            Returns:
                Any: whatever the function returns

            """
            if COMPILE_WITH_CONTEXT_LEAKING:
                args = args[1:]
            return func(*args)

    return BuiltinWrapper()


class _8f43c264c756af91f5eff200:
    def __init__(self, func):
        self.func = func
        if hasattr(func, "_e6c50da35e8f9284c183e69b"):
            self.n_args = 0
        else:
            self.n_args = _71fa9faaa6f884aa11f4cea2(
                _1a2fc26dc7ea5a2a4748b7cb(func).parameters.keys()
            )
        self.curried_args = []

    def __call__(self, *args):
        if hasattr(self.func, "_e6c50da35e8f9284c183e69b"):
            # pyffi function
            return self.func(*args)
        assert (
            self.n_args > 0
        ), "invalid compile output; bug in compyla... please raise issue in https://github.com/barnii77/lcaml_py"
        result = _dc937b59892604f5a86ac969
        for i, arg in enumerate(args):
            result = self
            self.curried_args.append(arg)
            if _71fa9faaa6f884aa11f4cea2(self.curried_args) == self.n_args:
                # call function and make the result the new function object
                out = self.func(*self.curried_args)
                if _4a11dbf5131539804348ceb5(
                    out, _8f43c264c756af91f5eff200
                ):  # isinstance(out, ThisClass)
                    self.func = out.func
                    self.n_args = out.n_args
                    self.curried_args = out.curried_args
                elif callable(out):
                    self.__init__(out)
                elif i != _71fa9faaa6f884aa11f4cea2(args) - 1:
                    raise TypeError(f"Cannot call object {out} of type {type(out)}")
                result = out
        return result


@_881ecbfb15f7e6881a337113
def print(*args):
    _ce953a0eb08246617b7f8494(*args, sep="", end="")


@_881ecbfb15f7e6881a337113
def println(*args):
    _ce953a0eb08246617b7f8494(*args, sep="")


@_881ecbfb15f7e6881a337113
def input(prompt):
    return _c96c6d5be8d08a12e7b5cdc1(prompt)


@_881ecbfb15f7e6881a337113
def isinstance(table, struct_type):
    if not _4a11dbf5131539804348ceb5(table, _6ab47d70854a8c690a0c2035):
        raise TypeError(f"Expected table to be a dict, got {type(table)}")
    if not _4a11dbf5131539804348ceb5(struct_type, _6ee0eb490ff832101cf82a3d):
        raise TypeError(
            f"Expected struct_type to be a StructType, got {type(struct_type)}"
        )
    return _6ee0eb490ff832101cf82a3d(table.keys()) == struct_type


@_881ecbfb15f7e6881a337113
def islike(a, b):
    if _4a11dbf5131539804348ceb5(
        a, _6ab47d70854a8c690a0c2035
    ) and _4a11dbf5131539804348ceb5(b, _6ab47d70854a8c690a0c2035):
        return a.keys() == b.keys()
    elif not _4a11dbf5131539804348ceb5(
        a, _6ab47d70854a8c690a0c2035
    ) and not _4a11dbf5131539804348ceb5(b, _6ab47d70854a8c690a0c2035):
        return type(a) == type(b)
    return False


@_881ecbfb15f7e6881a337113
def string(x):
    try:
        return str(x)
    except ValueError:
        pass


@_881ecbfb15f7e6881a337113
def float(x):
    try:
        return _76a7e234dc54e9c605b2cc9f(x)
    except ValueError:
        pass


@_881ecbfb15f7e6881a337113
def int(x):
    try:
        return _6da88c34ba124c41f977db66(x)
    except ValueError:
        pass


@_881ecbfb15f7e6881a337113
def bool(x):
    try:
        return _b760f44fa5965c2474a3b471(x)
    except ValueError:
        pass


@_881ecbfb15f7e6881a337113
def get(thing, key):
    if _4a11dbf5131539804348ceb5(thing, _6ab47d70854a8c690a0c2035):
        return thing[key] if key in thing else _dc937b59892604f5a86ac969
    elif _4a11dbf5131539804348ceb5(
        thing, (_a330395cc0a53ad120773654, _8c25cb3686462e9a86d2883c)
    ):
        return thing[key]
    raise TypeError(
        f"unsupported type {type(thing)}: supports only list, string and dict"
    )


@_881ecbfb15f7e6881a337113
def set(thing, key, value):
    thing[key] = value


@_881ecbfb15f7e6881a337113
def list(thing):
    return _a330395cc0a53ad120773654(thing)


@_881ecbfb15f7e6881a337113
def join(list_of_strings, join_elem=""):
    return join_elem.join(list_of_strings)


@_881ecbfb15f7e6881a337113
def len(thing):
    return _71fa9faaa6f884aa11f4cea2(thing)


@_881ecbfb15f7e6881a337113
def keys(thing):
    return thing.keys()


@_881ecbfb15f7e6881a337113
def values(thing):
    return thing.values()


@_881ecbfb15f7e6881a337113
def append(list, thing):
    list.append(thing)


@_881ecbfb15f7e6881a337113
def insert(list, index, thing):
    list.insert(index, thing)


@_881ecbfb15f7e6881a337113
def pop(list, index=-1):
    list.pop(index)


@_881ecbfb15f7e6881a337113
def fuse(thing1, thing2, inplace=False):
    if inplace:
        thing1.update(thing2)
        return thing1
    return {**thing1, **thing2}


@_6e8be7d82f8a724d77d4d12c
def import_lcaml(*args):
    context = {}
    if COMPILE_WITH_CONTEXT_LEAKING:
        context, path, *_ = args
    else:
        path, *_ = args
    path_without_ext, ext = _840a8dcfeae95966a870b0b5.path.splitext(path)
    if ext != "lml":
        raise ValueError("Invalid path: file must have .lml extension")
    return import_py(context, path_without_ext + ".py")


@_6e8be7d82f8a724d77d4d12c
def import_glob(*args):
    context = {}
    if COMPILE_WITH_CONTEXT_LEAKING:
        context, path, *_ = args
    else:
        path, *_ = args
    if not _4a11dbf5131539804348ceb5(path, _8c25cb3686462e9a86d2883c):
        raise ValueError("argument 1 (filepath) must be of type string")
    dirpath = _840a8dcfeae95966a870b0b5.path.join(
        _840a8dcfeae95966a870b0b5.path.dirname(_518b67e652531c5fe7e25d6b.argv[0]),
        "__modules",
        path,
    )
    if not _840a8dcfeae95966a870b0b5.path.exists(dirpath):
        return 1
    filepath = _840a8dcfeae95966a870b0b5.path.join(dirpath, "module.lml")
    args = (filepath,)
    return import_lcaml(context, args)


@_6e8be7d82f8a724d77d4d12c
def import_py(*args):
    context = {}
    if COMPILE_WITH_CONTEXT_LEAKING:
        context, path = args
    else:
        (path,) = args
    if not _840a8dcfeae95966a870b0b5.path.exists(path):
        raise ValueError("Invalid path: path doesn't exist")
    path_without_ext, ext = _840a8dcfeae95966a870b0b5.path.splitext(path)
    ext = ext[1:]  # remove .
    if ext != "py":
        raise ValueError(f"Invalid path: file must have .py extension, but has .{ext}")
    if "." in path_without_ext:
        raise ValueError(
            "Disallowed path: path must not contain any dots (except before extension of file)"
        )
    python_path = (
        path_without_ext.replace("\\\\", ".").replace("\\", ".").replace("/", ".")
    )
    g = {}
    _2706c619fe73f0cf112473c6(f"import {python_path} as mod", g)
    return g["mod"].module(context)


@_881ecbfb15f7e6881a337113
def exit(code=0):
    _518b67e652531c5fe7e25d6b.exit(code)


@_881ecbfb15f7e6881a337113
def panic(msg):
    if not _4a11dbf5131539804348ceb5(msg, _8c25cb3686462e9a86d2883c):
        raise TypeError("argument 1 (msg) of panic must be of type str")
    raise RuntimeError("panic: " + msg)


@_881ecbfb15f7e6881a337113
def ord(c):
    if not _4a11dbf5131539804348ceb5(c, str):
        raise TypeError("argument 1 (c) must be of type string")
    if _71fa9faaa6f884aa11f4cea2(c) > 1:
        raise ValueError(
            "ord expects a single-character string, but got multi-character string"
        )
    return _55df18d062878fb6d3f4a6d1(c)


@_881ecbfb15f7e6881a337113
def chr(o):
    if not _4a11dbf5131539804348ceb5(o, _6da88c34ba124c41f977db66):
        raise TypeError("argument 1 (o) must be of type int")
    return _943723cd5955a5316f4364f7(o)


@_881ecbfb15f7e6881a337113
def time():
    return _336074805fc853987abe6f7f.time()


@_881ecbfb15f7e6881a337113
def sleep(s):
    if not _4a11dbf5131539804348ceb5(
        s, (_6da88c34ba124c41f977db66, _76a7e234dc54e9c605b2cc9f)
    ):
        raise TypeError("argument 1 (s) must be of type int or float")
    if s < 0:
        raise ValueError("cannot sleep for negative period")
    return _336074805fc853987abe6f7f.sleep(s)


@_881ecbfb15f7e6881a337113
def breakpoint():
    _252813b4c8c5b033423ac5ab()


@_881ecbfb15f7e6881a337113
def jit(func):
    # python doesn't have a JIT compiler
    return func


@_6e8be7d82f8a724d77d4d12c
def is_defined(*args):
    context = _ad7aaf167f237a94dc2c3ad2
    if COMPILE_WITH_CONTEXT_LEAKING:
        context, var_name, *_ = args
    else:
        var_name, *_ = args
    return var_name in context


@_881ecbfb15f7e6881a337113
def abs(x):
    return _78da4a596a88bc5114f071ba(x)


@_881ecbfb15f7e6881a337113
def py_hasattr(obj, attr):
    return hasattr(obj, attr)


@_881ecbfb15f7e6881a337113
def py_getattr(obj, attr, default_value=None):
    return getattr(obj, attr, default_value)


@_881ecbfb15f7e6881a337113
def py_setattr(obj, attr, value):
    setattr(obj, attr, value)


@_881ecbfb15f7e6881a337113
def py_getattr_exec(obj, attr):
    g = {"obj": obj}
    _2706c619fe73f0cf112473c6(
        f"x = obj.{attr}",
        g,
    )
    return g["x"]


@_881ecbfb15f7e6881a337113
def py_setattr_exec(obj, attr, value):
    _2706c619fe73f0cf112473c6(
        f"obj.{attr} = value",
        {"obj": obj, "value": value},
    )


@_881ecbfb15f7e6881a337113
def py_exec(py_code, py_globs):
    _2706c619fe73f0cf112473c6(py_code, py_globs)


@_881ecbfb15f7e6881a337113
def exec(l_code, l_globs):
    raise RuntimeError(
        "the lcaml exec function is not available in lcaml that has been compiled to python"
    )


@_6e8be7d82f8a724d77d4d12c
def locals(*args):
    if COMPILE_WITH_CONTEXT_LEAKING:
        return args[0]
    else:
        return {}


@_881ecbfb15f7e6881a337113
def slice(iterable, start, end=None, step=1):
    return iterable[start:end:step]


@_881ecbfb15f7e6881a337113
def id(obj):
    return _091d41be4c916bd540e8b292(obj)


@_881ecbfb15f7e6881a337113
def copy(thing):
    if _4a11dbf5131539804348ceb5(thing, _6ab47d70854a8c690a0c2035):
        return {k: v for k, v in thing.items()}
    elif _4a11dbf5131539804348ceb5(thing, _a330395cc0a53ad120773654):
        return [v for v in thing]
    elif _4a11dbf5131539804348ceb5(thing, _6ee0eb490ff832101cf82a3d):
        return {v for v in thing}
    return thing


@_881ecbfb15f7e6881a337113
def deep_copy(thing):
    return _f676713508a2d8f20081f0fa(thing)


@_6e8be7d82f8a724d77d4d12c
def update_bounds(*args):
    if COMPILE_WITH_CONTEXT_LEAKING:
        context, func = args
        for k, v in func.bounds.items():
            if v is None:
                func.bounds[k] = context.get(k)
    else:
        (func,) = args
    return func


@_881ecbfb15f7e6881a337113
def openf(filepath, mode="r"):
    return open(filepath, mode)


@_881ecbfb15f7e6881a337113
def closef(file):
    return file.close()


@_881ecbfb15f7e6881a337113
def readablef(file):
    return file.readable()


@_881ecbfb15f7e6881a337113
def readf(file, n=None):
    return file.read(n)


@_881ecbfb15f7e6881a337113
def readlinef(file):
    return file.readline()


@_881ecbfb15f7e6881a337113
def writef(file, x):
    return file.write(x)


@_881ecbfb15f7e6881a337113
def seekf(file, position):
    return file.seek(position)


_ad7aaf167f237a94dc2c3ad2 = globals()
