COMPILE_WITH_CONTEXT_LEAKING = True
# this is a template, undefined values will be inserted above automatically during compilation
from inspect import signature as _1a2fc26dc7ea5a2a4748b7cb
import sys as _518b67e652531c5fe7e25d6b
import os as _840a8dcfeae95966a870b0b5
from lcaml_py.core import pyffi

pyffi.COMPILE_WITH_CONTEXT_LEAKING = COMPILE_WITH_CONTEXT_LEAKING
_c96c6d5be8d08a12e7b5cdc1 = input
py_print = print
_6ee0eb490ff832101cf82a3d = set
_a330395cc0a53ad120773654 = list
_6ab47d70854a8c690a0c2035 = dict
_6da88c34ba124c41f977db66 = int
_76a7e234dc54e9c605b2cc9f = float
_b760f44fa5965c2474a3b471 = bool
py_isinstance = isinstance
_71fa9faaa6f884aa11f4cea2 = len
_dc937b59892604f5a86ac969 = None
__compiled = True


def builtin(func):
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


class CallWrapper:
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
                if py_isinstance(
                    out, CallWrapper
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


@builtin
def print(*args):
    py_print(*args, sep="", end="")


@builtin
def println(*args):
    py_print(*args, sep="")


@builtin
def input(prompt):
    return _c96c6d5be8d08a12e7b5cdc1(prompt)


@builtin
def isinstance(table, struct_type):
    if not py_isinstance(table, _6ab47d70854a8c690a0c2035):
        raise TypeError(f"Expected table to be a dict, got {type(table)}")
    if not py_isinstance(struct_type, _6ee0eb490ff832101cf82a3d):
        raise TypeError(
            f"Expected struct_type to be a StructType, got {type(struct_type)}"
        )
    return _6ee0eb490ff832101cf82a3d(table.keys()) == struct_type


@builtin
def islike(a, b):
    if py_isinstance(
        a, _6ab47d70854a8c690a0c2035
    ) and py_isinstance(b, _6ab47d70854a8c690a0c2035):
        return a.keys() == b.keys()
    elif not py_isinstance(
        a, _6ab47d70854a8c690a0c2035
    ) and not py_isinstance(b, _6ab47d70854a8c690a0c2035):
        return type(a) == type(b)
    return False


@builtin
def string(x):
    try:
        return str(x)
    except ValueError:
        pass


@builtin
def float(x):
    try:
        return _76a7e234dc54e9c605b2cc9f(x)
    except ValueError:
        pass


@builtin
def int(x):
    try:
        return _6da88c34ba124c41f977db66(x)
    except ValueError:
        pass


@builtin
def bool(x):
    try:
        return _b760f44fa5965c2474a3b471(x)
    except ValueError:
        pass


@builtin
def get(thing, key):
    return thing[key] if key in thing else _dc937b59892604f5a86ac969


@builtin
def set(thing, key, value):
    thing[key] = value


@builtin
def list(thing):
    return _a330395cc0a53ad120773654(thing)


@builtin
def join(list_of_strings, join_elem):
    return join_elem.join(list_of_strings)


@builtin
def len(thing):
    return _71fa9faaa6f884aa11f4cea2(thing)


@builtin
def keys(thing):
    return thing.keys()


@builtin
def values(thing):
    return thing.values()


@builtin
def append(list, thing):
    list.append(thing)


@builtin
def pop(list, index):
    list.pop(index)


@builtin
def fuse(thing1, thing2):
    return {**thing1, **thing2}


@builtin
def import_lcaml(path: str):
    path_without_ext, ext = _840a8dcfeae95966a870b0b5.path.splitext(path)
    if ext != "lml":
        raise ValueError("Invalid path: file must have .lml extension")
    return import_py(path_without_ext + ".py")


@builtin
def import_py(path: str):
    if not _840a8dcfeae95966a870b0b5.path.exists(path):
        raise ValueError(f"Invalid path: path doesn't exist")
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
    exec(f"import {python_path} as mod", g)
    return g["mod"].module()


@builtin
def exit(code):
    _518b67e652531c5fe7e25d6b.exit(code)


context = globals()


def module():
    def factorial(context, n, __this):
        context["n"] = n
        context["__this"] = __this
        if context["n"] <= 1:
            return 1
        elif True:
            return context["n"] * (
                context["__this"]
                if py_isinstance(
                    context["__this"], CallWrapper
                )
                else CallWrapper(context["__this"])
            )(context, context["n"] - 1)

    factorial_self_referral_list = [0]
    context["factorial"] = (
        lambda context, n: factorial(
            context.copy(),
            n,
            factorial_self_referral_list[0],
        )
    )
    factorial_self_referral_list[0] = (
        lambda context, n: factorial(
            context.copy(),
            n,
            factorial_self_referral_list[0],
        )
    )
    return (
        context["factorial"]
        if py_isinstance(
            context["factorial"], CallWrapper
        )
        else CallWrapper(context["factorial"])
    )(context, 200)


if __name__ == "__main__":
    py_print(module())
