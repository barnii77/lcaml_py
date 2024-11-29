import functools
from typing import Callable, Any, Optional, Dict
from . import extern_python as extern_python
from .interpreter_types import DType, Object
from . import lcaml_expression as lcaml_expression
from . import parser_types as parser_types
from . import interpreter_vm as interpreter_vm_mod
from .lcaml_lexer import Syntax

# if hasattr(func, COMPYLA_IS_EXTERN_MAGIC_ATTRIBUTE_NAME) => func is pyffi function
COMPYLA_IS_EXTERN_MAGIC_ATTRIBUTE_NAME = "_e6c50da35e8f9284c183e69b"
COMPILE_WITH_CONTEXT_LEAKING = True


def _lcaml_to_python(
    lcaml_obj, interpreter_vm=None, _prev_values: Optional[Dict[int, Any]] = None
):
    if lcaml_obj is None:
        return None

    if _prev_values is None:
        _prev_values = {}

    # this is a mechanism to allow conversion of cyclic/self-referential objects
    if id(lcaml_obj.value) in _prev_values:
        return _prev_values[id(lcaml_obj.value)]

    if lcaml_obj.type == DType.INT:
        return int(lcaml_obj.value)
    elif lcaml_obj.type == DType.FLOAT:
        return float(lcaml_obj.value)
    elif lcaml_obj.type == DType.STRING:
        return str(lcaml_obj.value)
    elif lcaml_obj.type == DType.BOOL:
        return bool(lcaml_obj.value)
    elif lcaml_obj.type == DType.TABLE:
        # Convert LCaml Table to Python dict
        table = lcaml_obj.value
        out_dict = {}
        _prev_values[id(lcaml_obj.value)] = out_dict
        for key, val in table.fields.items():
            out_dict[key] = _lcaml_to_python(Object(val.type, val), interpreter_vm, _prev_values)
        return out_dict
    elif lcaml_obj.type == DType.UNIT:
        return None
    elif lcaml_obj.type == DType.EXTERN_PYTHON:
        return lcaml_obj.value
    elif lcaml_obj.type == DType.FUNCTION:
        if interpreter_vm is None:
            raise ValueError("interpreter_vm argument is required for function objects")

        def vm_wrapper(*args):
            args = [
                lcaml_expression.ObjectFakeAst(_python_to_lcaml(arg, interpreter_vm)) for arg in args
            ]
            func_call = lcaml_expression.FunctionCall(lcaml_obj, args)
            return _lcaml_to_python(func_call.resolve(interpreter_vm.context))

        return vm_wrapper
    elif lcaml_obj.type == DType.STRUCT_TYPE:
        # Convert LCaml StructType to Python set of field names
        struct_type = lcaml_obj.value
        return {field.name for field in struct_type.fields}
    elif lcaml_obj.type == DType.PY_OBJ:
        return lcaml_obj.value
    elif lcaml_obj.type == DType.LIST:
        lcaml_list = lcaml_obj.value
        out_list = []
        _prev_values[id(lcaml_obj.value)] = out_list
        for item in lcaml_list:
            out_list.append(_lcaml_to_python(item, interpreter_vm, _prev_values))
        return out_list
    else:
        raise TypeError("Unsupported LCaml type")


def _python_to_lcaml(
    py_obj,
    interpreter_vm=None,
    wrap_extern_py=True,
    _prev_values: Optional[Dict[int, Any]] = None,
):
    if py_obj is None:
        ret_obj = Object(DType.UNIT, None)

    if _prev_values is None:
        _prev_values = {}

    # this is a mechanism to allow conversion of cyclic/self-referential objects
    if id(py_obj) in _prev_values:
        return _prev_values[id(py_obj)]

    # NOTE: isinstance(True, int) returns True (and maybe some other weird stuff), therefore, use exact types
    if type(py_obj) is int:
        ret_obj = Object(DType.INT, py_obj)
    elif type(py_obj) is float:
        ret_obj = Object(DType.FLOAT, py_obj)
    elif type(py_obj) is str:
        ret_obj = Object(DType.STRING, py_obj)
    elif type(py_obj) is bool:
        ret_obj = Object(DType.BOOL, py_obj)
    elif type(py_obj) is dict:
        # Convert Python dict to LCaml Table
        fields = {}
        ret_obj = Object(DType.TABLE, lcaml_expression.Table(fields))
        _prev_values[id(py_obj)] = ret_obj
        for key, val in py_obj.items():
            fields[key] = _python_to_lcaml(val, interpreter_vm, wrap_extern_py, _prev_values)
    elif hasattr(py_obj, "__call__"):
        ret_obj = Object(
            DType.EXTERN_PYTHON,
            interface(py_obj, interpreter_vm) if wrap_extern_py else py_obj,
        )
    elif type(py_obj) is set and all(isinstance(item, str) for item in py_obj):
        # Convert Python list of field names to LCaml StructType
        fields = [parser_types.AstIdentifier(field) for field in py_obj]
        ret_obj = Object(DType.STRUCT_TYPE, lcaml_expression.StructType(fields))
    elif type(py_obj) is list:
        inner = []
        ret_obj = Object(DType.LIST, inner)
        _prev_values[id(py_obj)] = ret_obj
        for item in py_obj:
            inner.append(_python_to_lcaml(item, interpreter_vm, wrap_extern_py, _prev_values))
    else:
        ret_obj = Object(DType.PY_OBJ, py_obj)
    return ret_obj


def interface(
    _func=None,
    interpreter_vm=None,
    name: str = "PyFFI_Interface",
) -> Callable:
    """
    A high-level interface for functions that take a list of arguments with python object types and return a python object.
    LCaml objects cannot be mutated directly with this approach.
    This interface is easier to work with than the raw interface, but is less powerful.
    For most use cases, this is the best choice and provides the necessary functionality.
    E.g. almost all of the old lcaml_builtins.py classes (look them up in the old git commits if you want) can be expressed using this interface.

    Args:
        func (function[...]): function to wrap; takes a list of arguments (python types) and returns a python object
        interpreter_vm (InterpreterVM): None | InterpreterVM; required only if the function takes LCaml functions as arguments
            Because these LCaml functions may potentially

    Returns:
        Wrapper | function[func] -> Wrapper: function-wrapping object that provides the needed interface to the LCaml interpreter

    """
    # NOTE: lcaml doesn't have keyword arguments, so we ignore them here

    def decorator(func):
        class Wrapper(extern_python.ExternPython):
            def __init__(self):
                setattr(self, COMPYLA_IS_EXTERN_MAGIC_ATTRIBUTE_NAME, 0)

            @staticmethod
            def execute(context, args):
                _interpreter_vm = interpreter_vm
                if _interpreter_vm is None and Syntax._vm_intrinsic in context:
                    _interpreter_vm = context[Syntax._vm_intrinsic].value
                if _interpreter_vm is not None and not isinstance(
                    _interpreter_vm, interpreter_vm_mod.InterpreterVM
                ):
                    raise TypeError(
                        f"{Syntax._vm_intrinsic} was overwritten with illegal value `{_interpreter_vm}`: intrinsic value must be an InterpreterVM"
                    )
                _prev_values = {}
                py_args = [_lcaml_to_python(arg, _interpreter_vm, _prev_values) for arg in args]
                result = func(*py_args)
                return _python_to_lcaml(result, _interpreter_vm)

            def __str__(self):
                return name

            @staticmethod
            def __call__(*args):  # _ is context
                if COMPILE_WITH_CONTEXT_LEAKING:
                    args = args[1:]
                return func(*args)

        return Wrapper()

    return decorator if _func is None else decorator(_func)


def raw(
    _func=None,
    name: str = "PyFFI_Raw",
) -> Callable:
    """
    A raw interface is a function that takes a context and a list of arguments with lcaml object types.
    The raw interface gives the programmer full power to do basically anything with the args and interpreter state.
    However, it is also harder to work with.
    Use with caution.

    Args:
        func[context: Dict, args: List]: function to wrap; takes interpreter state (context) and a list of arguments (raw LCaml types)

    Returns:
        Wrapper | function[func] -> Wrapper: function-wrapping object that provides the needed interface to the LCaml interpreter

    """
    # NOTE: lcaml doesn't have keyword arguments, so we ignore them here

    def decorator(func):
        class Wrapper(extern_python.ExternPython):
            def __init__(self):
                setattr(self, COMPYLA_IS_EXTERN_MAGIC_ATTRIBUTE_NAME, 0)

            @staticmethod
            def execute(context, args):
                return func(context, args)

            def __str__(self):
                return name

            def __call__(self, *args):
                context = {}
                if COMPILE_WITH_CONTEXT_LEAKING:
                    context, *args = args
                return func(context, args)

        return Wrapper()

    return decorator if _func is None else decorator(_func)


def pymodule(func):
    @functools.wraps(func)
    def wrapper(context):
        exports = func(context)
        if not context.get("__compiled"):
            return _python_to_lcaml(exports, wrap_extern_py=False)
        return exports

    return wrapper
