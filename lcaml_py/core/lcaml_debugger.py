import os
import sys
from typing import Union, List
from .lcaml_utils import get_marked_code_snippet
from . import interpreter as interpreter_mod
from . import interpreter_types
from . import lcaml_lexer
from . import pyffi


@pyffi.interface(name="print")
def l_print(*args):
    print(*args, sep="")


def parse_breakpoint_location(vm, params):
    if params:
        p = params[0].strip()
        if p.isdigit():
            file, line = vm.file, int(p)
        elif ":" in p:
            file, line = p.split(":")
            if not line.isdigit():
                print("Invalid line number: not a positive integer")
                return None
        else:
            print("Invalid format for breakpoint location")
            return None
    else:
        file, line = vm.file, vm.statement_line
    try:
        point = f"{os.path.abspath(file)}:{line}"
    except Exception:
        print(f"Invalid filepath `{file}`")
        return None
    return point


def get_vm_and_interpreter_from_context(context):
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

    interpreter_obj: "interpreter_types.Object" = context.get(
        lcaml_lexer.Syntax._interpreter_intrinsic
    )
    if interpreter_obj is None:
        raise RuntimeError(
            f"{lcaml_lexer.Syntax._interpreter_intrinsic} intrinsic is not set: Illegal state"
        )
    if interpreter_obj.type != interpreter_types.DType.PY_OBJ or not isinstance(
        interpreter_obj.value, interpreter_mod.Interpreter
    ):
        raise RuntimeError(
            f"{lcaml_lexer.Syntax._interpreter_intrinsic} intrinsic contains value not of type PY_OBJ/Interpreter: Illegal state"
        )

    vm: "interpreter_mod.interpreter_vm.InterpreterVM" = vm_obj.value
    interpreter: "interpreter_mod.Interpreter" = interpreter_obj.value
    return vm, interpreter


def weighted_levenshtein(s1, s2):
    # Create a matrix to store distances
    len_s1 = len(s1) + 1
    len_s2 = len(s2) + 1
    dp = [[0] * len_s2 for _ in range(len_s1)]
    # Initialize the matrix
    for i in range(len_s1):
        dp[i][0] = i  # Deletion cost
    for j in range(len_s2):
        dp[0][j] = j  # Insertion cost
    # Compute the distances
    for i in range(1, len_s1):
        for j in range(1, len_s2):
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Insertion
                dp[i][j - 1] + 2,  # Deletion
                dp[i - 1][j - 1] + 100 * int(s1[i - 1] != s2[j - 1]),  # Substitution
            )
    return dp[-1][-1]


def cmd_next_handler(vm: "interpreter_mod.interpreter_vm.InterpreterVM", *_):
    vm.local_next_step_callbacks.append(lambda v: debugger_main(v.context))


def cmd_step_handler(vm: "interpreter_mod.interpreter_vm.InterpreterVM", *_):
    vm.next_step_callbacks.append(lambda v: debugger_main(v.context))


def cmd_continue_handler(*_):
    pass


def cmd_list_handler(
    vm: "interpreter_mod.interpreter_vm.InterpreterVM",
    params,
    interpreter: "interpreter_mod.Interpreter",
    *_,
):
    if params:
        line = params[0].strip()
        if not line.isdigit():
            print("Parameter of `list [line]` must be a positive integer")
            return True
        line = int(line)
    else:
        line = vm.statement_line
    snippet = get_marked_code_snippet(interpreter.code.splitlines(), line - 1, 11)
    print(snippet)
    return True


def cmd_break_handler(
    vm: "interpreter_mod.interpreter_vm.InterpreterVM",
    params,
    *_,
):
    point = parse_breakpoint_location(vm, params)
    if point:
        vm.line_callbacks[point] = lambda v: debugger_main(v.context)
    return True


def cmd_lcaml_handler(
    vm: "interpreter_mod.interpreter_vm.InterpreterVM",
    lc: str,
    *_,
):
    if lc.isspace():
        print("Missing lcaml code to execute")
        return True
    if not lc.endswith(";"):
        lc += ";"
    new_interpreter = interpreter_mod.Interpreter(lc, file="<debugger lcaml command>")
    ret = new_interpreter.execute(vm.context)
    if ret is not None:
        l_print.execute(None, (ret,))
    return True


def cmd_print_handler(vm: "interpreter_mod.interpreter_vm.InterpreterVM", params, *_):
    if not params:
        print("print requires exactly one parameter: the name of the variable to print")
        return True
    var = params[0]
    if var in vm.context:
        l_print.execute(None, (vm.context[var],))
    else:
        print("<undefined>")
    return True


def cmd_clear_handler(
    vm: "interpreter_mod.interpreter_vm.InterpreterVM",
    params,
    *_,
):
    if not params:
        vm.line_callbacks.clear()
        return True
    point = parse_breakpoint_location(vm, params)
    if point:
        if point in vm.line_callbacks:
            vm.line_callbacks.pop(point)
        else:
            print("No breakpoint at", point)
    return True


def get_call_stack_traceback(call_stack) -> str:
    tb_lines = []
    tb_lines.append("LCaml Traceback (most recent call last):")
    code_lines = None
    for loc in reversed(call_stack):
        if isinstance(loc, interpreter_mod.Interpreter):
            tb_lines.append(f"In file {loc.vm.file}:\n")
            code_lines = loc.code.splitlines()
        elif isinstance(loc, interpreter_mod.interpreter_vm.InterpreterVM):
            tb_lines.append(f"On line {loc.statement_line}:")
            tb_lines.append(
                get_marked_code_snippet(code_lines, loc.statement_line - 1, 3)
                if code_lines is not None
                else "<code unavailable>"
            )
            tb_lines.append("")
        else:
            raise TypeError("Invalid traceback entry encountered.")
    return "\n".join(tb_lines)


def cmd_backtrace_handler(vm: "interpreter_mod.interpreter_vm.InterpreterVM", *_):
    call_stack: List[
        Union[
            "interpreter_mod.interpreter_vm.InterpreterVM",
            "interpreter_mod.Interpreter",
        ]
    ] = [vm]
    cur_vm = vm
    while cur_vm.parent is not None:
        if (
            isinstance(cur_vm, interpreter_mod.interpreter_vm.InterpreterVM)
            and cur_vm._causes_traceback_entry
            or isinstance(cur_vm, interpreter_mod.Interpreter)
        ):
            call_stack.append(cur_vm.parent)
        cur_vm = cur_vm.parent
    tb = get_call_stack_traceback(call_stack)
    print(tb)
    return True


def cmd_help_handler(context, params, *_):
    if len(params) > 1:
        print("Usage: `help [command]`; Examples: `help`, `help list`")
    if len(params) == 0:
        print(HELP_MESSAGE)
    else:
        name = params[0]
        if name not in commands:
            print(f"Command {name} does not exist. Use `help` for more information.")
            return True
        h = cmd_helpers.get(name)
        if h is None:
            print(f"Command {name} does not have a help entry.")
            return True
        print(h)
    return True


def cmd_quit_handler(*_):
    sys.exit(0)


SIMILARITY_DETECTION_TYPO_CUTOFF = 4
commands = {
    "next": cmd_next_handler,
    "step": cmd_step_handler,
    "continue": cmd_continue_handler,
    "list": cmd_list_handler,
    "breakpoint": cmd_break_handler,
    "lcaml": cmd_lcaml_handler,
    "print": cmd_print_handler,
    "clear": cmd_clear_handler,
    "backtrace": cmd_backtrace_handler,
    "traceback": cmd_backtrace_handler,
    "quit": cmd_quit_handler,
    "exit": cmd_quit_handler,
    "help": cmd_help_handler,
}

NEWLINE = "\n"
HELP_MESSAGE = f"""
This is the LCaml debugger. It is kind of like Pdb (python's builtin debugger), just simpler. Use `help {{command}}` to get information about some command.
Here are the currently supported commands:
{NEWLINE.join(('- ' + cmd for cmd in commands))}
The debugger also supports abbreviations. There are some predefined ones.
If a command is not recognized, the debugger will try to find the most similar existing command if the similarity is great enough.
"""

cmd_helpers = {
    "next": "Step over. Executes until the next line of code is hit or the current scope ends.\nIn the latter case, it halts again in the parent scope.",
    "step": "Step into. Halts at the next statement to be executed.\nIf the current statement calls a function, it will step into the function's code and halt there.",
    "continue": "Continue execution normally until a breakpoint is hit or the breakpoint builtin is executed again.",
    "list": "List a snippet of code around the line the interpreter has been halted at.\nIf a parameter is passed (`list 10`), it will list around the specified line instead of the current line.",
    "breakpoint": "Set a breakpoint on a specific line. You may also specify the file the breakpoint applies to.\nThe file defaults to the currently executing file. Examples: `breakpoint 58`, `breakpoint hello.lml:21`",
    "lcaml": "Execute some lcaml code in the current context. Arbitrary code can be specified after the word `lcaml`.\nExamples: `lcaml println x`, `lcaml let y = 5; while (y > 0) {let y = y - 1;}; println (var_in_context + y);`",
    "print": "Print the value of some variable. Examples: `print x`",
    "clear": "Delete some breakpoint. If no breakpoint is specified, delete all breakpoints.\nExamples: `clear`, `clear 58`, `clear hello.lml:21`",
    "backtrace": "Prints the callstack",
    "traceback": "Equivalent to `backtrace`",
    "quit": "Terminate the process.",
    "exit": "Equivalent to `quit`",
    "help": "Print this message.",
}

aliases = {
    "c": "continue",
    "cont": "continue",
    "con": "continue",
    "b": "breakpoint",
    "break": "breakpoint",
    "br": "breakpoint",
    "lc": "lcaml",
    "p": "print",
    "cl": "clear",
    "bt": "backtrace",
    "tb": "traceback",
    "h": "help",
    "q": "quit",
}

command_pass_raw_args = {"lcaml": True}


def debugger_main(context):
    vm, interpreter = get_vm_and_interpreter_from_context(context)
    rerun = True
    print(
        get_marked_code_snippet(
            interpreter.code.splitlines(), vm.statement_line - 1, window_size=1
        )
    )
    if not vm._enable_vm_callbacks:
        print(
            "Warning: vm callbacks are disabled. This limits the functionality of the debugger."
        )
    while rerun:
        try:
            x = input("(Ldb) ")
            if not x or x.isspace():
                continue
            cmd, *params = x.split()
            lvs = {c: weighted_levenshtein(c, cmd) for c in commands}
            min_lvs = min(lvs, key=lambda k: lvs[k])
            if (
                lvs[min_lvs] >= SIMILARITY_DETECTION_TYPO_CUTOFF
                and cmd not in aliases
                and cmd not in commands
            ):
                print(f"Command {cmd} not found")
                continue
            cmd = (
                cmd if cmd in commands else aliases[cmd] if cmd in aliases else min_lvs
            )
            if command_pass_raw_args.get(cmd, False):
                params = " ".join(params)
            else:
                params = tuple(filter(lambda x: x, params))
            handler = commands[cmd]
            rerun = handler(vm, params, interpreter, context)
        except Exception as e:
            print("Internal debugger error:", str(e))
