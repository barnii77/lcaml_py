import json
import time
import sys
import os
from copy import deepcopy
from . import interpreter as interpreter_mod
from . import lcaml_expression
from . import interpreter_types
from . import lcaml_lexer
from . import pyffi
from . import lcaml_debugger


@pyffi.interface(name="print")
def l_print(*args):
    print(*(arg if arg is not None else "()" for arg in args), sep="", end="")


@pyffi.interface(name="println")
def l_println(*args):
    print(*(arg if arg is not None else "()" for arg in args), sep="")


@pyffi.interface(name="input")
def l_input(prompt):
    return input(prompt)


@pyffi.interface(name="isinstance")
def l_is_instance(table, struct_type):
    if not isinstance(table, dict):
        raise TypeError(f"Expected table to be a dict, got {type(table)}")
    if not isinstance(struct_type, set):
        raise TypeError(
            f"Expected struct_type to be a StructType, got {type(struct_type)}"
        )
    return set(table.keys()) == struct_type


@pyffi.interface(name="islike")
def l_is_like(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys()
    elif not isinstance(a, dict) and not isinstance(b, dict):
        return type(a) == type(b)
    return False


@pyffi.interface(name="float")
def l_float(x):
    try:
        return float(x)
    except ValueError:
        pass


@pyffi.interface(name="int")
def l_int(x):
    try:
        return int(x)
    except ValueError:
        pass


@pyffi.interface(name="string")
def l_string(x):
    try:
        return str(x)
    except ValueError:
        pass


@pyffi.interface(name="bool")
def l_bool(x):
    try:
        return bool(x)
    except ValueError:
        pass


@pyffi.raw(name="set")
def l_set(context, args):
    if len(args) != 3:
        raise ValueError("set takes 3 arguments: table, key, value")
    if args[0].type not in (
        interpreter_types.DType.TABLE,
        interpreter_types.DType.LIST,
    ):
        raise TypeError("argument 1 (iter) must be of type table or list")
    iterable, key, value = args[0].value, args[1], args[2]
    if args[0].type == interpreter_types.DType.LIST:
        if key.type != interpreter_types.DType.INT:
            raise TypeError("argument 2 (key) must be of type int")
        index = key.value
        if index >= len(iterable):
            raise IndexError(f"index {index} out of range")
        iterable[index] = value
    else:
        iterable.fields[
            key.value if key.type == interpreter_types.DType.STRING else key
        ] = value
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.raw(name="list")
def l_list(context, args):
    if len(args) != 1:
        raise ValueError("list takes 1 argument: thing")
    if args[0].type not in (
        interpreter_types.DType.LIST,
        interpreter_types.DType.STRUCT_TYPE,
        interpreter_types.DType.STRING,
    ):
        raise TypeError("argument 1 (iter) must be of type list, struct_type, string")
    if args[0].type == interpreter_types.DType.LIST:
        return args[0]
    elif args[0].type == interpreter_types.DType.STRUCT_TYPE:
        return interpreter_types.Object(
            interpreter_types.DType.LIST, list(args[0].value.fields)
        )
    else:
        return interpreter_types.Object(
            interpreter_types.DType.LIST,
            [
                interpreter_types.Object(interpreter_types.DType.STRING, c)
                for c in list(args[0].value)
            ],
        )


@pyffi.raw(name="copy")
def l_copy(context, args):
    if len(args) != 1:
        raise ValueError("copy takes 1 argument: thing")
    ty = args[0].type
    v = args[0].value
    if ty == interpreter_types.DType.LIST:
        return interpreter_types.Object(ty, v.copy())
    elif ty in (interpreter_types.DType.TABLE, interpreter_types.DType.STRUCT_TYPE):
        return interpreter_types.Object(ty, v.fields.copy())
    else:
        return interpreter_types.Object(ty, v)


@pyffi.interface(name="deep_copy")
def l_deep_copy(thing):  # pyffi does deep copy when converting to/from python values
    return thing


@pyffi.interface(name="join")
def l_join(list_of_strings: list[str], join_elem: str = ""):
    return join_elem.join(list_of_strings)


@pyffi.raw(name="get")
def l_get(context, args):
    if len(args) != 2:
        raise ValueError("get takes 2 arguments: table, key")
    ty = args[0].type
    if ty not in (
        interpreter_types.DType.TABLE,
        interpreter_types.DType.LIST,
        interpreter_types.DType.STRING,
    ):
        raise ValueError("argument 1 must be of type table, list or string")
    iterable, key = args[0].value, args[1]
    if ty == interpreter_types.DType.LIST:
        if key.type != interpreter_types.DType.INT:
            raise TypeError("argument 2 (key) must be of type int")
        index = key.value
        if index >= len(iterable):
            raise IndexError(f"index {index} out of range")
        return iterable[index]
    elif ty == interpreter_types.DType.STRING:
        if key.type != interpreter_types.DType.INT:
            raise TypeError("argument 2 (key) must be of type int")
        index = key.value
        if index >= len(iterable):
            raise IndexError(f"index {index} out of range")
        return interpreter_types.Object(interpreter_types.DType.STRING, iterable[index])
    else:
        key = key.value if key.type == interpreter_types.DType.STRING else key
        if key not in iterable.fields:
            return interpreter_types.Object(lcaml_expression.DType.UNIT, None)
        return iterable.fields[key]


@pyffi.raw(name="slice")
def l_slice(context, args):
    if len(args) not in (2, 3, 4):
        raise RuntimeError(
            f"slice takes 2, 3 or 4 arguments (iterable, start, end?, step?), but got {len(args)}"
        )
    if not all(arg.type in (interpreter_types.DType.INT, interpreter_types.DType.UNIT) for arg in args[1:]):
        raise TypeError("all index arguments (arg 2, 3, 4) must be integer or unit type")
    if len(args) == 2:
        iterable, start, end, step = *args, None, 1
        start = start.value
    elif len(args) == 3:
        iterable, start, end, step = *args, 1
        start, end = start.value, end.value
    else:
        iterable, start, end, step = args
        start, end, step = start.value, end.value, step.value
    if iterable.type in (interpreter_types.DType.LIST, interpreter_types.DType.STRING):
        return interpreter_types.Object(iterable.type, iterable.value[start:end:step])
    else:
        raise TypeError(
            f"expected iterable (list or string), got {interpreter_types.DType.name(iterable.type)}"
        )


@pyffi.raw(name="len")
def l_len(context, args):
    if len(args) != 1:
        raise ValueError("len takes 1 argument: iterable")
    if args[0].type not in (
        interpreter_types.DType.TABLE,
        interpreter_types.DType.LIST,
        interpreter_types.DType.STRING,
    ):
        raise ValueError("argument 1 must be of type table, list or string")
    iterable = args[0].value
    ty = args[0].type
    if ty in (interpreter_types.DType.LIST, interpreter_types.DType.STRING):
        length = len(iterable)
    else:
        length = len(iterable.fields)
    return interpreter_types.Object(interpreter_types.DType.INT, length)


@pyffi.raw(name="keys")
def l_keys(context, args):
    if len(args) != 1:
        raise ValueError("keys takes 1 argument: table")
    if not args[0].type == interpreter_types.DType.TABLE:
        raise ValueError("argument 1 (table) must be of type table")
    table = args[0].value
    return interpreter_types.Object(
        interpreter_types.DType.LIST,
        lcaml_expression.LList(list(table.fields.keys())),
    )


@pyffi.raw(name="values")
def l_values(context, args):
    if len(args) != 1:
        raise ValueError("keys takes 1 argument: table")
    if not args[0].type == interpreter_types.DType.TABLE:
        raise ValueError("argument 1 (table) must be of type table")
    table = args[0].value
    return interpreter_types.Object(
        interpreter_types.DType.LIST,
        lcaml_expression.LList(list(table.fields.values())),
    )


@pyffi.raw(name="append")
def l_append(context, args):
    if len(args) != 2:
        raise ValueError("append takes 2 arguments: list, value")
    if not args[0].type == interpreter_types.DType.LIST:
        raise ValueError("argument 1 (list) must be of type list")
    list_, value = args[0].value, args[1]
    list_.append(value)
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.raw(name="insert")
def l_insert(context, args):
    if len(args) != 3:
        raise ValueError("append takes 3 arguments: list, index, value")
    if not args[0].type == interpreter_types.DType.LIST:
        raise ValueError("argument 1 (list) must be of type list")
    if not args[1].type == interpreter_types.DType.INT:
        raise ValueError("argument 2 (index) must be of type int")
    list_, index, value = args[0].value, args[1].value, args[2]
    list_.insert(index, value)
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.raw(name="pop")
def l_pop(context, args):
    if len(args) not in (1, 2):
        raise ValueError("append takes 2 arguments: list, index")
    if args[0].type != interpreter_types.DType.LIST:
        raise ValueError("argument 1 (list) must be of type list")
    list_ = args[0].value
    if len(args) == 1:
        ret = list_.pop()
    else:
        if args[1].type != interpreter_types.DType.INT:
            raise ValueError("argument 2 (index) must be of type int")
        idx = args[1].value
        ret = list_.pop(idx)
    if not isinstance(ret, interpreter_types.Object):
        raise RuntimeError("internal error")
    return ret


@pyffi.raw(name="import_lcaml")
def l_import_lcaml(context, args):
    if len(args) not in (1, 2):
        raise ValueError("import takes 1 or 2 arguments: filepath, Optional[syntax]")
    if not args[0].type == interpreter_types.DType.STRING:
        raise ValueError("argument 1 (filepath) must be of type string")
    filepath = args[0].value
    if not os.path.exists(filepath):
        return interpreter_types.Object(interpreter_types.DType.INT, 1)
    with open(filepath, "r") as f:
        code = f.read()
    syntax = None
    if len(args) == 2:
        if args[1].type != interpreter_types.DType.STRING:
            raise ValueError("argument 2 (syntax_filepath) must be of type string")
        syntax_filepath = args[1].value
        if not os.path.exists(syntax_filepath):
            return interpreter_types.Object(interpreter_types.DType.INT, 2)
        with open(syntax_filepath, "r") as f:
            syntax_raw = f.read()
        syntax_dict = json.loads(syntax_raw)
        syntax = lcaml_lexer.Syntax(**syntax_dict)

    base_interpreter_obj: "interpreter_types.Object" = context.get(
        lcaml_lexer.Syntax._interpreter_intrinsic
    )
    if base_interpreter_obj is None:
        raise RuntimeError(
            f"{lcaml_lexer.Syntax._interpreter_intrinsic} intrinsic is not set: Illegal state"
        )
    if base_interpreter_obj.type != interpreter_types.DType.PY_OBJ or not isinstance(
        base_interpreter_obj.value, interpreter_mod.Interpreter
    ):
        raise RuntimeError(
            f"{lcaml_lexer.Syntax._interpreter_intrinsic} intrinsic contains value not of type PY_OBJ/Interpreter: Illegal state"
        )

    vm_obj: "interpreter_types.Object" = context.get(lcaml_lexer.Syntax._vm_intrinsic)
    if vm_obj is None:
        raise RuntimeError(
            f"{lcaml_lexer.Syntax._vm_intrinsic} intrinsic is not set: Illegal state"
        )
    if vm_obj.type != interpreter_types.DType.PY_OBJ or not isinstance(
        vm_obj.value, interpreter_mod.interpreter_vm.InterpreterVM
    ):
        raise RuntimeError(
            f"{lcaml_lexer.Syntax._vm_intrinsic} intrinsic contains value not of type PY_OBJ/InterpreterVM: Illegal state"
        )

    vm: "interpreter_mod.interpreter_vm.InterpreterVM" = vm_obj.value
    base_interpreter: "interpreter_mod.Interpreter" = base_interpreter_obj.value

    file_interpreter = interpreter_mod.Interpreter(
        code,
        syntax,
        filepath,
        base_interpreter.line_callbacks,
        base_interpreter.next_step_callbacks,
        base_interpreter.enable_vm_callbacks,
        vm,
    )
    result = file_interpreter.execute(context.copy())
    return (
        result
        if result is not None
        else interpreter_types.Object(interpreter_types.DType.UNIT, None)
    )


@pyffi.raw(name="import_glob")
def l_import_glob(context, args):
    if len(args) not in (1, 2):
        raise ValueError("import takes 1 or 2 arguments: filepath, Optional[syntax]")
    if not args[0].type == interpreter_types.DType.STRING:
        raise ValueError("argument 1 (filepath) must be of type string")
    dirpath = os.path.join(os.path.dirname(sys.argv[0]), "__modules", args[0].value)
    if not os.path.exists(dirpath):
        return interpreter_types.Object(interpreter_types.DType.INT, 1)
    filepath = os.path.join(dirpath, "module.lml")
    args = (interpreter_types.Object(interpreter_types.DType.STRING, filepath),) + (
        (args[1],) if len(args) == 2 else ()
    )
    return l_import_lcaml.execute(context, args)


@pyffi.raw(name="import_py")
def l_import_py(context, args):
    if len(args) != 1:
        raise ValueError("import takes 1 argument: filepath")
    if not args[0].type == interpreter_types.DType.STRING:
        raise ValueError("argument 1 (filepath) must be of type string")
    path = args[0].value
    if not os.path.exists(path):
        raise ValueError(f"Invalid path: path {path} doesn't exist")
    path_without_ext, ext = os.path.splitext(path)
    ext = ext[1:]  # remove .
    if ext != "py":
        raise ValueError(
            f"Invalid path: file path points to must have .py extension, but has .{ext}"
        )
    if "." in path_without_ext:
        raise ValueError(
            "Disallowed path: path must not contain any dots (except before extension of file)"
        )
    python_path = (
        path_without_ext.replace("\\\\", ".").replace("\\", ".").replace("/", ".")
    )
    g = {}
    exec(f"import {python_path} as mod", g)
    return g["mod"].module(
        context
    )  # no need to copy context here, module will not change it


@pyffi.raw(name="fuse")
def l_fuse(context, args):
    """fuses 2 tables together"""
    if len(args) not in (2, 3):
        raise ValueError("fuse takes 2 or 3 arguments: table1, table2, [bool inplace]")
    if not all(arg.type == interpreter_types.DType.TABLE for arg in args):
        raise ValueError("arguments must be of type table")
    table1, table2 = args[0].value, args[1].value
    inplace = False
    if len(args) == 3:
        if args[2].type != interpreter_types.DType.BOOL:
            raise ValueError("argument 3 (inplace) must be of type bool")
        inplace = args[2].value
    if inplace:
        data = table1.fields
    else:
        data = table1.fields.copy()
    data.update(table2.fields)
    if inplace:
        return table1
    return interpreter_types.Object(interpreter_types.DType.TABLE, data)


@pyffi.interface(name="exit")
def l_exit(code: int = 0):
    sys.exit(code)


@pyffi.interface(name="sleep")
def l_sleep(seconds):
    if not isinstance(seconds, (int, float)):
        raise ValueError("argument 1 (seconds) must be of type float or int")
    time.sleep(seconds)


@pyffi.interface(name="panic")
def l_panic(msg):
    if not isinstance(msg, str):
        raise TypeError("argument 1 (msg) of panic must be of type str")
    raise RuntimeError("panic: " + msg)


@pyffi.interface(name="ord")
def l_ord(c):
    if not isinstance(c, str):
        raise TypeError("argument 1 (c) must be of type string")
    if len(c) > 1:
        raise ValueError(
            "ord expects a single-character string, but got multi-character string"
        )
    return ord(c)


@pyffi.interface(name="chr")
def l_chr(o):
    if not isinstance(o, int):
        raise TypeError("argument 1 (o) must be of type int")
    return chr(o)


@pyffi.interface(name="time")
def l_time():
    return time.time()


@pyffi.raw(name="breakpoint")
def l_breakpoint(context, _):
    lcaml_debugger.debugger_main(context)
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.raw(name="jit")
def l_jit(context, args):
    if len(args) != 1:
        raise ValueError("jit expects 1 argument: func")
    func_obj = args[0]
    if func_obj.type != interpreter_types.DType.FUNCTION:
        raise TypeError(
            f"first argument of `jit` must be a function, but is of type {interpreter_types.DType.name(func_obj.type)}"
        )
    func: "lcaml_expression.Function" = func_obj.value
    new_func = deepcopy(func)
    new_func.force_jit = True
    return new_func


@pyffi.raw(name="is_defined")
def l_is_defined(context, args):
    if len(args) != 1:
        raise ValueError("is_defined expects 1 argument: var_name")
    var_name_obj = args[0]
    if var_name_obj.type != interpreter_types.DType.STRING:
        raise TypeError(
            f"first argument of `jit` must be a function, but is of type {interpreter_types.DType.name(var_name_obj.type)}"
        )
    var_name: str = var_name_obj.value
    return var_name in context


@pyffi.interface(name="abs")
def l_abs(x):
    return abs(x)


@pyffi.raw(name="py_hasattr")
def l_py_hasattr(context, args):
    if len(args) == 2:
        py_obj, attr = args
    else:
        raise RuntimeError(f"expected 2 arguments, got {len(args)}")
    if attr.type != interpreter_types.DType.STRING:
        raise TypeError(f"expected attribute to be string, but got {type(attr.value)}")
    return interpreter_types.Object(
        interpreter_types.DType.BOOL, hasattr(py_obj.value, attr.value)
    )


@pyffi.raw(name="py_getattr")
def l_py_getattr(context, args):
    if len(args) == 2:
        py_obj, attr, default_value = *args, interpreter_types.Object(
            interpreter_types.DType.UNIT, None
        )
    elif len(args) == 3:
        py_obj, attr, default_value = args
    else:
        raise RuntimeError(f"expected 2 or 3 arguments, got {len(args)}")
    if attr.type != interpreter_types.DType.STRING:
        raise TypeError(f"expected attribute to be string, but got {type(attr.value)}")
    return pyffi._python_to_lcaml(
        getattr(py_obj.value, attr.value, default_value.value),
        context.get(lcaml_lexer.Syntax._vm_intrinsic),
    )


@pyffi.raw(name="py_setattr")
def l_py_setattr(context, args):
    if len(args) == 3:
        py_obj, attr, value = args
    else:
        raise RuntimeError(f"expected 3 arguments, got {len(args.value)}")
    if attr.type != interpreter_types.DType.STRING:
        raise TypeError(f"expected attribute to be string, but got {type(attr.value)}")
    if py_obj.type in (
        interpreter_types.DType.INT,
        interpreter_types.DType.FLOAT,
        interpreter_types.DType.STRING,
        interpreter_types.DType.UNIT,
        interpreter_types.DType.BOOL,
        interpreter_types.DType.STRUCT_TYPE,
        interpreter_types.DType.LIST,
        interpreter_types.DType.TABLE,
    ):
        raise TypeError(
            f"cannot use py_setattr on primitive builtin types in python (list, int, ...): object has type {interpreter_types.DType.name(py_obj.type)}, which is disallowed"
        )
    setattr(py_obj.value, attr.value, value.value)
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.raw(name="py_getattr_exec")
def l_py_getattr_exec(context, args):
    if len(args) == 2:
        py_obj, attr = args
    else:
        raise RuntimeError(f"expected 2 arguments, got {len(args)}")
    if attr.type != interpreter_types.DType.STRING:
        raise TypeError(f"expected attribute to be string, but got {type(attr.value)}")
    g = {"py_obj_value": py_obj.value}
    exec(
        f"x = py_obj_value.{attr.value}",
        g,
    )
    return pyffi._python_to_lcaml(
        g["x"],
        context.get(lcaml_lexer.Syntax._vm_intrinsic),
    )


@pyffi.raw(name="py_setattr_exec")
def l_py_setattr_exec(context, args):
    if len(args) == 3:
        py_obj, attr, value = args
    else:
        raise RuntimeError(f"expected 3 arguments, got {len(args.value)}")
    if attr.type != interpreter_types.DType.STRING:
        raise TypeError(f"expected attribute to be string, but got {type(attr.value)}")
    if py_obj.type in (
        interpreter_types.DType.INT,
        interpreter_types.DType.FLOAT,
        interpreter_types.DType.STRING,
        interpreter_types.DType.UNIT,
        interpreter_types.DType.BOOL,
        interpreter_types.DType.STRUCT_TYPE,
        interpreter_types.DType.LIST,
        interpreter_types.DType.TABLE,
    ):
        raise TypeError(
            f"cannot use py_setattr on primitive builtin types in python (list, int, ...): object has type {interpreter_types.DType.name(py_obj.type)}, which is disallowed"
        )
    exec(
        f"py_obj_value.{attr.value} = value_value",
        {"py_obj_value": py_obj.value, "value_value": value.value},
    )
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.interface(name="py_exec")
def l_py_exec(py_code, py_globs):
    if not isinstance(py_code, str):
        raise TypeError(f"expected py_code to be str, got {type(py_code)}")
    if not isinstance(py_globs, dict):
        raise TypeError(f"expected py_globs to be dict, got {type(py_globs)}")
    exec(py_code, py_globs)
    return py_globs


@pyffi.raw(name="exec")
def l_exec(context, args):
    if len(args) != 2:
        raise RuntimeError(f"expected 2 args (l_code, l_globs), but got {len(args)}")
    l_code, l_globs = args
    if l_code.type != interpreter_types.DType.STRING:
        raise TypeError(
            f"expected l_code to be string, got {interpreter_types.DType.name(l_code.type)}"
        )
    if l_globs.type != interpreter_types.DType.TABLE:
        raise TypeError(
            f"expected l_globs to be table, got {interpreter_types.DType.name(l_globs.type)}"
        )
    interpreter = interpreter_mod.Interpreter(l_code.value)
    result = interpreter.execute(l_globs.value.fields)
    return (
        result
        if result is not None
        else interpreter_types.Object(interpreter_types.DType.UNIT, None)
    )


@pyffi.raw(name="locals")
def l_locals(context, args):
    return interpreter_types.Object(
        interpreter_types.DType.TABLE, lcaml_expression.Table(context)
    )


@pyffi.raw(name="id")
def l_id(context, args):
    if len(args) != 1:
        raise RuntimeError(f"expected 1 arg of any type, but got {len(args)}")
    return interpreter_types.Object(interpreter_types.DType.INT, id(args[0]))


@pyffi.raw(name="update_bounds")
def l_update_bounds(context, args):
    if len(args) != 1:
        raise RuntimeError("expected 1 argument")
    func_obj = args[0]
    if func_obj.type != interpreter_types.DType.FUNCTION:
        raise TypeError(
            f"expected argument of type function, got {interpreter_types.DType.name(func_obj.type)}"
        )
    func = func_obj.value
    intersecting_keys = func.bounds.keys() & context.keys()
    for key in intersecting_keys:
        if func.bounds.get(key) is None:
            func.bounds[key] = context[key]
    return func_obj


@pyffi.pymodule
def module(context):
    exports = {
        "print": l_print,
        "println": l_println,
        "input": l_input,
        "isinstance": l_is_instance,
        "islike": l_is_like,
        "nl": interpreter_types.Object(interpreter_types.DType.STRING, "\n"),
        "__compiled": interpreter_types.Object(
            interpreter_types.DType.BOOL, False
        ),  # this intrinsic signals that program is run by interpreter
        "float": l_float,
        "int": l_int,
        "string": l_string,
        "bool": l_bool,
        "set": l_set,
        "list": l_list,
        "join": l_join,
        "get": l_get,
        "len": l_len,
        "slice": l_slice,
        "keys": l_keys,
        "values": l_values,
        "append": l_append,
        "insert": l_insert,
        "pop": l_pop,
        "import_lcaml": l_import_lcaml,
        "import_glob": l_import_glob,
        "import_py": l_import_py,
        "fuse": l_fuse,
        "exit": l_exit,
        "sleep": l_sleep,
        "panic": l_panic,
        "ord": l_ord,
        "chr": l_chr,
        "time": l_time,
        "breakpoint": l_breakpoint,
        "jit": l_jit,
        "is_defined": l_is_defined,
        "abs": l_abs,
        "py_hasattr": l_py_hasattr,
        "py_getattr": l_py_getattr,
        "py_setattr": l_py_setattr,
        "py_getattr_exec": l_py_getattr_exec,
        "py_setattr_exec": l_py_setattr_exec,
        "py_exec": l_py_exec,
        "exec": l_exec,
        "locals": l_locals,
        "id": l_id,
        "copy": l_copy,
        "deep_copy": l_deep_copy,
        "update_bounds": l_update_bounds,
    }
    return exports
