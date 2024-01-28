from parser_types import AstIdentifier, AstStatementType, AstAssignment, AstReturn
from interpreter_types import Object


class InterpreterVM:
    """
    This class represents the interpreter virtual machine.

    Attributes:
        ast: AST to interpret
        variables: variables to use in the interpreter

    """

    def __init__(self, ast, variables: dict[AstIdentifier, Object]):
        self.variables = variables
        self.ast = ast
        self.return_value = None

    def execute(self):
        for statement in self.ast.statements:
            if statement.type == AstStatementType.ASSIGNMENT:
                assert (
                    type(statement.value) == AstAssignment
                ), "Bug: statement.value is not AstAssignment"
                assignment = statement.value
                identifier: AstIdentifier = assignment.identifier
                value: Object = assignment.value.resolve(self.variables)
                self.variables[identifier] = value

            elif statement.type == AstStatementType.RETURN:
                assert (
                    type(statement.value) == AstReturn
                ), "Bug: statement.value is not AstReturn"
                expression = statement.value.value
                self.return_value = expression.resolve(self.variables)
                return

            else:
                raise ValueError("Unknown statement type " + statement.type)
