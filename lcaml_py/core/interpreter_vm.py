import itertools
from typing import Optional, Callable, Dict, List, Union
from inspect import signature
from . import parser_types as parser_types
from .lcaml_lexer import Syntax
from .interpreter_types import Object, DType
from . import interpreter


class InterpreterVM:
    """
    This class represents the interpreter virtual machine.
    """

    def __init__(
        self,
        ast,
        context: Optional[dict[str, "Object"]] = None,
        parent: Optional[Union["InterpreterVM", "interpreter.Interpreter"]] = None,
        line_callbacks: Optional[Dict[str, Callable]] = None,
        next_step_callbacks: Optional[List[Callable]] = None,
        file: str = "",
        _causes_traceback_entry: bool = False,
        _enable_vm_callbacks: bool = True,
    ):
        if context is None:
            context = {}
        self.file = file
        self.context = context
        self.ast = ast
        self.return_value: Optional[Object] = None
        self.parent = parent
        self._enable_vm_callbacks = _enable_vm_callbacks
        # used for tracebacks and debugger
        self.vm_object = Object(DType.PY_OBJ, self)
        self.statement_idx = 0
        self.statement_line = -1
        self.line_callbacks = {} if line_callbacks is None else line_callbacks
        self.next_step_callbacks = (
            [] if next_step_callbacks is None else next_step_callbacks
        )
        self.local_next_step_callbacks = []
        self._causes_traceback_entry = _causes_traceback_entry

        if not _enable_vm_callbacks:

            def set_vm_callback():
                self.context[Syntax._vm_intrinsic] = self.vm_object

            self._vm_step = set_vm_callback

    def _propagate_local_next_step_callbacks(self):
        # no more statements, control will return to parent => propagate up local_next_step_callbacks to parent
        if self._enable_vm_callbacks and self.parent:
            if isinstance(self.parent, InterpreterVM):
                self.parent.local_next_step_callbacks.extend(
                    self.local_next_step_callbacks
                )
            elif isinstance(
                self.parent.parent, InterpreterVM
            ):  # should always be True (except if parent.parent is None)
                self.parent.parent.local_next_step_callbacks.extend(
                    self.local_next_step_callbacks
                )

    def _invoke_callback(self, cb):
        assert callable(cb) and signature(cb), "Callback must be callable"
        assert len(
            signature(cb).parameters.keys()
        ), "Callback must take 1 argument (arg = vm_instance)"
        cb(self)

    def _vm_step(self):
        self.context[Syntax._vm_intrinsic] = self.vm_object
        cb = self.line_callbacks.get(f"{self.file}:{self.statement_line}")
        if cb is not None:
            self._invoke_callback(cb)
        next_step_callbacks, local_next_step_callbacks = (
            self.next_step_callbacks.copy(),
            self.local_next_step_callbacks.copy(),
        )
        self.next_step_callbacks.clear()
        self.local_next_step_callbacks.clear()
        for cb in itertools.chain(next_step_callbacks, local_next_step_callbacks):
            self._invoke_callback(cb)

    def _execute(self):
        for i, statement in enumerate(self.ast.statements):
            self.statement_idx = i
            self.statement_line = statement.line
            self._vm_step()

            if statement.type == parser_types.AstStatementType.ASSIGNMENT:
                assignment = statement.value
                identifier: parser_types.AstIdentifier = assignment.identifier
                value: Object = assignment.value.resolve(self.context)
                self.context[identifier.name] = value

            elif statement.type == parser_types.AstStatementType.RETURN:
                expression = statement.value.value
                self.return_value = expression.resolve(self.context)
                self._propagate_local_next_step_callbacks()
                return

            elif statement.type == parser_types.AstStatementType.CONTROL_FLOW:
                control_flow = statement.value
                for branch in control_flow.branches:
                    self.statement_line = branch.line
                    self._vm_step()
                    if branch.condition.resolve(self.context):
                        interpreter_vm = InterpreterVM(
                            branch.body,
                            self.context,
                            self,
                            self.line_callbacks,
                            self.next_step_callbacks,
                            self.file,
                            self._causes_traceback_entry,
                            self._enable_vm_callbacks,
                        )
                        _causes_traceback_entry, self._causes_traceback_entry = (
                            self._causes_traceback_entry,
                            False,
                        )
                        interpreter_vm.execute()
                        self._causes_traceback_entry = _causes_traceback_entry
                        if interpreter_vm.return_value is not None:
                            self.return_value = interpreter_vm.return_value
                            self._propagate_local_next_step_callbacks()
                            return
                        break

            elif statement.type == parser_types.AstStatementType.WHILE_LOOP:
                while_loop = statement.value
                while while_loop.condition.resolve(self.context):
                    self._vm_step()
                    interpreter_vm = InterpreterVM(
                        while_loop.body,
                        self.context,
                        self,
                        self.line_callbacks,
                        self.next_step_callbacks,
                        self.file,
                        self._causes_traceback_entry,
                        self._enable_vm_callbacks,
                    )
                    _causes_traceback_entry, self._causes_traceback_entry = (
                        self._causes_traceback_entry,
                        False,
                    )
                    interpreter_vm.execute()
                    self._causes_traceback_entry = _causes_traceback_entry
                    if interpreter_vm.return_value is not None:
                        self.return_value = interpreter_vm.return_value
                        self._propagate_local_next_step_callbacks()
                        return

            elif statement.type == parser_types.AstStatementType.EXPRESSION:
                statement.value.expression.resolve(self.context)

            else:
                raise ValueError(f"Unknown statement type {statement.type}")

        self._propagate_local_next_step_callbacks()

    def execute(self):
        try:
            self._execute()
        except Exception as e:
            if self._causes_traceback_entry:
                if not hasattr(e, "__lcaml_traceback_info"):
                    setattr(e, "__lcaml_traceback_info", [])
                getattr(e, "__lcaml_traceback_info").append(self)
            raise e
