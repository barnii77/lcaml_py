from typing import Callable

from . import extern_python as extern_python
from .interpreter_types import DType, Object
from . import interpreter_vm as lcaml_vm
from . import lcaml_expression as lcaml_expression
from . import parser_types as parser_types
from . import token_type as token_type


def _lcaml_to_python(lcaml_obj, interpreter_vm=None):
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
        return {
            key: _lcaml_to_python(Object(val.type, val), interpreter_vm)
            for key, val in table.fields.items()
        }
    elif lcaml_obj.type == DType.UNIT:
        return None
    elif lcaml_obj.type == DType.EXTERN_PYTHON:
        return lcaml_obj.value
    elif lcaml_obj.type == DType.FUNCTION:
        # this is a bit more complex
        # return an InterpreterVM Object wrapper that takes args and executes the function and links with the
        # current VM context with the provided arguments

        if interpreter_vm is None:
            raise ValueError("interpreter_vm argument is required for function objects")
        lcaml_func = lcaml_obj.value

        def vm_wrapper(*args):
            func_call = lcaml_expression.FunctionCall(lcaml_func, list(args))
            new_vm = lcaml_vm.InterpreterVM(func_call)
            new_vm.variables = interpreter_vm.variables
            return new_vm.execute()

        return vm_wrapper
    elif lcaml_obj.type == DType.STRUCT_TYPE:
        # Convert LCaml StructType to Python set of field names
        struct_type = lcaml_obj.value
        return {field.name for field in struct_type.fields}
    elif lcaml_obj.type == DType.PY_OBJ:
        return lcaml_obj.value
    elif lcaml_obj.type == DType.LIST:
        lcaml_list = lcaml_obj.value
        return [_lcaml_to_python(item, interpreter_vm) for item in lcaml_list.values]
    else:
        raise TypeError("Unsupported LCaml type")


def _python_to_lcaml(py_obj, interpreter_vm=None):
    # NOTE: isinstance(True, int) returns True (and maybe some other weird stuff), therefore, use exact types
    if type(py_obj) == int:
        return Object(DType.INT, py_obj)
    elif py_obj is None:
        return Object(DType.UNIT, None)
    elif type(py_obj) == float:
        return Object(DType.FLOAT, py_obj)
    elif type(py_obj) == str:
        return Object(DType.STRING, py_obj)
    elif type(py_obj) == bool:
        return Object(DType.BOOL, py_obj)
    elif type(py_obj) == dict:
        # Convert Python dict to LCaml Table
        fields = {
            parser_types.AstIdentifier(
                token_type.Token(token_type.TokenKind._PHANTOM, key)
            ): _python_to_lcaml(val, interpreter_vm)
            for key, val in py_obj.items()
        }  # Assuming all values are strings
        return Object(DType.TABLE, lcaml_expression.Table(fields))
    elif hasattr(py_obj, "__call__"):
        return Object(DType.EXTERN_PYTHON, interface(py_obj, interpreter_vm))
    elif type(py_obj) == set and all(isinstance(item, str) for item in py_obj):
        # Convert Python list of field names to LCaml StructType
        fields = [parser_types.AstIdentifier(field) for field in py_obj]
        return Object(DType.STRUCT_TYPE, lcaml_expression.StructType(fields))
    elif type(py_obj) == list:
        inner = [_python_to_lcaml(item) for item in py_obj]
        return Object(DType.LIST, lcaml_expression.LList(inner))
    else:
        return Object(DType.PY_OBJ, py_obj)


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
            @staticmethod
            def execute(context, args):
                py_args = [_lcaml_to_python(arg, interpreter_vm) for arg in args]
                result = func(*py_args)
                return _python_to_lcaml(result)

            def __str__(self):
                return name

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
            @staticmethod
            def execute(context, args):
                return func(context, args)

            def __str__(self):
                return name

        return Wrapper()

    return decorator if _func is None else decorator(_func)
