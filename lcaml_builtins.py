import json
import time
import sys
import os
import interpreter
import lcaml_expression
import interpreter_types
import lcaml_lexer
import pyffi


@pyffi.interface(name="print")
def l_print(*args):
    print(*args, sep="", end="")


@pyffi.interface(name="println")
def l_println(*args):
    print(*args, sep="")


l_input = pyffi.interface(input, name="input")


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
        if index >= len(iterable.values):
            raise IndexError(f"index {index} out of range")
        iterable.values[index] = value
    else:
        iterable.fields[key] = value
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.raw(name="get")
def l_get(context, args):
    if len(args) != 2:
        raise ValueError("get takes 2 arguments: table, key")
    if args[0].type not in (
            interpreter_types.DType.TABLE,
            interpreter_types.DType.LIST,
    ):
        raise ValueError("argument 1 (table) must be of type table")
    iterable, key = args[0].value, args[1]
    if args[0].type == interpreter_types.DType.LIST:
        if key.type != interpreter_types.DType.INT:
            raise TypeError("argument 2 (key) must be of type int")
        index = key.value
        if index >= len(iterable.values):
            raise IndexError(f"index {index} out of range")
        return iterable.values[index]
    elif key not in iterable.fields:
        return interpreter_types.Object(lcaml_expression.DType.UNIT, None)
    return iterable.fields[key]


@pyffi.raw(name="len")
def l_len(context, args):
    if len(args) != 1:
        raise ValueError("len takes 1 argument: iterable")
    if args[0].type not in (
            interpreter_types.DType.TABLE,
            interpreter_types.DType.LIST,
    ):
        raise ValueError("argument 1 (table) must be of type table or list")
    iterable = args[0].value
    if args[0].type == interpreter_types.DType.LIST:
        return interpreter_types.Object(
            interpreter_types.DType.INT, len(iterable.values)
        )
    return interpreter_types.Object(interpreter_types.DType.INT, len(iterable.fields))


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
    list_.values.append(value)
    return interpreter_types.Object(interpreter_types.DType.UNIT, None)


@pyffi.raw(name="pop")
def l_pop(context, args):
    if len(args) not in (1, 2):
        raise ValueError("append takes 2 arguments: list, index")
    if args[0].type != interpreter_types.DType.LIST:
        raise ValueError("argument 1 (list) must be of type list")
    list_ = args[0].value
    if len(args) == 1:
        ret = list_.values.pop()
    else:
        if args[1].type != interpreter_types.DType.INT:
            raise ValueError("argument 2 (index) must be of type int")
        idx = args[1].value
        ret = list_.values.pop(idx)
    if not isinstance(ret, interpreter_types.Object):
        raise RuntimeError("internal error")
    return ret


@pyffi.raw(name="import")
def l_import(context, args):
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
    file_interpreter = interpreter.Interpreter(code, syntax)
    result = file_interpreter.execute(context)
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
    args = (interpreter_types.Object(interpreter_types.DType.STRING, filepath),)
    return l_import.execute(context, args)


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


LML_EXPORTS = {
    "print": l_print,
    "println": l_println,
    "input": l_input,
    "isinstance": l_is_instance,
    "islike": l_is_like,
    "nl": interpreter_types.Object(interpreter_types.DType.STRING, "\n"),
    "float": l_float,
    "int": l_int,
    "string": l_string,
    "bool": l_bool,
    "set": l_set,
    "get": l_get,
    "keys": l_keys,
    "values": l_values,
    "append": l_append,
    "pop": l_pop,
    "import": l_import,
    "import_glob": l_import_glob,
    "fuse": l_fuse,
    "exit": l_exit
}
