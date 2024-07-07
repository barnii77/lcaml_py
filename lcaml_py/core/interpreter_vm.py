from typing import Optional
from . import parser_types as parser_types
from .interpreter_types import Object


class InterpreterVM:
    """
    This class represents the interpreter virtual machine.

    Attributes:
        ast: AST to interpret
        variables: variables to use in the interpreter
        return_value: return value of the interpreter (None if no return statement was executed)

    """

    def __init__(self, ast, variables: dict['parser_types.AstIdentifier', 'Object'] = None):
        if variables is None:
            variables = {}
        self.variables = variables
        self.ast = ast
        self.return_value: Optional[Object] = None

    def execute(self):
        for statement in self.ast.statements:
            if statement.type == parser_types.AstStatementType.ASSIGNMENT:
                assert (
                    type(statement.value) == parser_types.AstAssignment
                ), "Bug: statement.value is not parser_types.AstAssignment"
                assignment = statement.value
                identifier: parser_types.AstIdentifier = assignment.identifier
                value: Object = assignment.value.resolve(self.variables)
                self.variables[identifier] = value

            elif statement.type == parser_types.AstStatementType.RETURN:
                assert (
                    type(statement.value) == parser_types.AstReturn
                ), "Bug: statement.value is not parser_types.AstReturn"
                expression = statement.value.value
                self.return_value = expression.resolve(self.variables)
                return

            elif statement.type == parser_types.AstStatementType.CONTROL_FLOW:
                control_flow = statement.value
                for branch in control_flow.branches:
                    if branch.condition.resolve(self.variables):
                        interpreter_vm = InterpreterVM(branch.body, self.variables)
                        interpreter_vm.execute()
                        if interpreter_vm.return_value is not None:
                            self.return_value = interpreter_vm.return_value
                            return
                        break

            else:
                raise ValueError("Unknown statement type " + statement.type)
