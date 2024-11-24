import os
import traceback
from copy import deepcopy
from timeit import timeit
import core


def get_lcaml_traceback(exc: Exception) -> str:
    tb_lines = []
    tb_lines.append("LCaml Traceback (most recent call last):")
    if not hasattr(exc, "__lcaml_traceback_info"):
        print("Unable to construct lcaml traceback.")
    else:
        tb_info = getattr(exc, "__lcaml_traceback_info")
        code_lines = None
        for loc in reversed(tb_info):
            if isinstance(loc, core.interpreter.Interpreter):
                tb_lines.append(f"In file {loc.vm.file}:\n")
                code_lines = loc.code.splitlines()
            elif isinstance(loc, core.interpreter_vm.InterpreterVM):
                tb_lines.append(f"On line {loc.statement_line}:")
                tb_lines.append(
                    core.lcaml_utils.get_marked_code_snippet(
                        code_lines, loc.statement_line - 1, 3
                    )
                    if code_lines is not None
                    else "<code unavailable>"
                )
                tb_lines.append("")
            elif isinstance(loc, int):
                tb_lines.append(f"On line {loc}:")
                tb_lines.append(
                    core.lcaml_utils.get_marked_code_snippet(code_lines, loc - 1, 3)
                    if code_lines is not None
                    else "<code unavailable>"
                )
                tb_lines.append("")
            elif (
                isinstance(loc, tuple)
                and len(loc) == 2
                and isinstance(loc[0], str)
                and isinstance(loc[1], str)
            ):
                file, code = loc
                tb_lines.append(f"In file {file}:\n")
                code_lines = code.splitlines()
            elif isinstance(loc, str):
                tb_lines.append("Note: " + loc + "\n")
            else:
                raise TypeError("Invalid traceback entry encountered.")

    # this chaos below improves error output format by trying to parse `repr(exc)` to some degree
    r = repr(exc)
    exc_name = ""
    while r[0].isalnum():
        exc_name += r[0]
        r = r[1:]
    if r.startswith("('") or r.startswith('("'):
        r = r[2:]
    elif r.startswith("("):
        r = r[1:]
    if r.endswith("')") or r.endswith('")'):
        r = r[:-2]
    elif r.endswith(")"):
        r = r[:-1]
    if r:
        r = ": " + r
    out = exc_name + r
    if out:
        out = "Raised " + out
    tb_lines.append(out)
    return "\n".join(tb_lines)


@core.pyffi.interface(name="print")
def l_print(*args):
    return


@core.pyffi.interface(name="println")
def l_println(*args):
    return


@core.pyffi.raw(name="jit")
def l_jit(context, args):
    if len(args) != 1:
        raise ValueError(f"jit expects 1 argument: func, but got {len(args)} arguments")
    func_obj = args[0]
    if func_obj.type != core.interpreter_types.DType.FUNCTION:
        raise TypeError(
            f"first argument of `jit` must be a function, but is of type {core.interpreter_types.DType.name(func_obj.type)}"
        )
    func: "core.lcaml_expression.Function" = func_obj.value
    func.force_jit = True
    return func


def flatten_object(obj, parent_key="", sep=".", seen=None):
    """
    Recursively flattens any object into a dictionary, detecting and avoiding circular references.

    :param obj: The object to flatten (can be any type).
    :param parent_key: The base key path (used for recursion).
    :param sep: Separator for key paths.
    :param seen: A set of seen object IDs to detect circular references.
    :return: A flat dictionary where keys represent paths to values in the original object.
    """
    if seen is None:
        seen = set()

    items = {}
    obj_id = id(obj)
    if obj_id in seen:  # Detect circular references
        return items
    seen.add(obj_id)

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_object(v, new_key, sep, seen))
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_object(v, new_key, sep, seen))
    elif hasattr(obj, "__dict__"):  # For objects with attributes
        for k, v in vars(obj).items():
            if k.startswith("__"):
                continue
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_object(v, new_key, sep, seen))
    items[parent_key] = obj  # Base case: primitive value or unhandled type

    return items


def run():
    interpreter.vm = core.interpreter_vm.InterpreterVM(
        interpreter.ast,
        parent=interpreter,
        line_callbacks=interpreter.line_callbacks,
        next_step_callbacks=interpreter.next_step_callbacks,
        file=file,
        _causes_traceback_entry=True,
        _enable_vm_callbacks=interpreter.enable_vm_callbacks,
    )
    interpreter.next_step_callbacks = []
    interpreter.line_callbacks = {}
    interpreter.ast = deepcopy(ast)
    variables = core.interpreter.get_builtins()
    variables["print"] = core.pyffi._python_to_lcaml(l_print, wrap_extern_py=False)
    variables["println"] = core.pyffi._python_to_lcaml(l_println, wrap_extern_py=False)
    variables["jit"] = core.pyffi._python_to_lcaml(
        l_jit, interpreter.vm, wrap_extern_py=False
    )
    if num_runs == 1:
        print("----------------")
        result = interpreter.execute(variables)
        print("----------------")
        print("interpreter returned: ", result)
    else:
        result = interpreter.execute(variables)
    if EXTRACT_JIT_CACHE and n[0] == 0:
        n[0] = 1
        flat = flatten_object(interpreter.ast)
        flat_ast = flatten_object(ast)
        for kc in flat:
            oc, oa = flat[kc], flat_ast[kc]
            if isinstance(oa, core.lcaml_expression.Function):
                assert isinstance(oc, core.lcaml_expression.Function)
                oa.jit_cache = oc.jit_cache


if __name__ == "__main__":
    core.lcaml_expression.initialize_llvmlite()
    num_runs = 1000
    core.lcaml_expression.SUPPRESS_JIT = False
    EXTRACT_JIT_CACHE = True
    folder = "tests/end_to_end"
    folder_py_baseline = "tests/end_to_end_py_baseline"
    if folder.endswith("/"):
        folder = folder[:-1]
    if folder_py_baseline.endswith("/"):
        folder_py_baseline = folder_py_baseline[:-1]
    for file in os.listdir(folder):
        if "breakpoint" in file:
            continue
        if not file.endswith(".lml") or file.startswith("_"):
            continue
        with open(f"{folder}/{file}", "r") as f:
            code = f.read()
        print(f"Running test: {file}")
        if num_runs == 1:
            print()
        interpreter = core.interpreter.Interpreter(code, file=file)
        ast = deepcopy(interpreter.ast)
        n = [0]
        try:
            time_taken = timeit(run, number=num_runs)
        except Exception as e:
            print(get_lcaml_traceback(e))
        else:
            print(
                f"Time taken to run {num_runs} times: {time_taken} seconds [average {time_taken / num_runs}]"
            )

        py_file = os.path.splitext(file)[0] + ".py"
        if not os.path.exists(f"{folder_py_baseline}/{py_file}"):
            print("\n----------------\n")
            continue
        with open(f"{folder_py_baseline}/{py_file}", "r") as f:
            py_code = f.read()
        py_bc = compile(py_code, py_file, "exec")
        try:
            time_taken = timeit(lambda: exec(py_bc), number=num_runs)
        except Exception:
            traceback.format_exc()
        else:
            print(
                f"Python took {time_taken} seconds [average {time_taken / num_runs}] to run {num_runs} times"
            )
            print("\n----------------\n")
