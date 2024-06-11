import extern_python
from interpreter_types import Object, DType
import interpreter_vm as lcaml_vm
import lcaml_expression


def lcaml_to_python(lcaml_obj, interpreter_vm=None):
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
        return {key: lcaml_to_python(Object(val.type, val)) for key, val in table.fields.items()}
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
    else:
        raise TypeError("Unsupported LCaml type")


def python_to_lcaml(py_obj):
    if isinstance(py_obj, int):
        return Object(DType.INT, py_obj)
    elif py_obj is None:
        return Object(DType.UNIT, None)
    elif isinstance(py_obj, float):
        return Object(DType.FLOAT, py_obj)
    elif isinstance(py_obj, str):
        return Object(DType.STRING, py_obj)
    elif isinstance(py_obj, bool):
        return Object(DType.BOOL, py_obj)
    elif isinstance(py_obj, dict):
        # Convert Python dict to LCaml Table
        fields = {key: Object(DType.STRING, val) for key, val in py_obj.items()}  # Assuming all values are strings
        return Object(DType.TABLE, lcaml_expression.Table(fields))
    elif hasattr(py_obj, "__call__"):
        # function

        class Wrapper(extern_python.ExternPython):
            def execute(self, context, args):
                return py_obj(*args)

            def __str__(self):
                return "PyFFIFunction"

        return Object(DType.EXTERN_PYTHON, Wrapper())
    elif isinstance(py_obj, set) and all(isinstance(item, str) for item in py_obj):
        # Convert Python list of field names to LCaml StructType
        fields = [lcaml_expression.AstIdentifier(field) for field in py_obj]
        return Object(DType.STRUCT_TYPE, lcaml_expression.StructType(fields))
    else:
        raise TypeError("Unsupported Python type")


def pyffi(func, interpreter_vm=None):
    # NOTE: lcaml doesn't have keyword arguments, so we ignore them here
    class Wrapper(extern_python.ExternPython):
        def execute(self, context, args):
            py_args = [lcaml_to_python(arg, interpreter_vm) for arg in args]
            result = func(*py_args)
            return python_to_lcaml(result)

        def __str__(self):
            return "PyFFI"

    return Wrapper()
