import lcaml_parser
from lcaml_parser import Ast
from interpreter_types import Object
from expression_evaluator import ExpressionEvaluator


class InterpreterVM:
    """
    This class represents the interpreter virtual machine.

    Attributes:
        ast: AST to interpret
        variables: variables to use in the interpreter

    """

    def __init__(self, ast: Ast, variables: dict[str, Object]):
        self.variables = variables
        self.ast = ast

    def execute(self):
        for statement in self.ast.statements:
            if statement.type == lcaml_parser.AstStatementType.ASSIGNMENT:
                assert (
                    type(statement.value) == lcaml_parser.AstAssignment
                ), "Bug: statement.value is not AstAssignment"
                assignment = statement.value
                identifier: str = assignment.identifier.name
                value: Object = ExpressionEvaluator(assignment.value)(self.variables)
                self.variables[identifier] = value

            elif statement.type == lcaml_parser.AstStatementType.RETURN:
                assert (
                    type(statement.value) == lcaml_parser.AstReturn
                ), "Bug: statement.value is not AstReturn"
                return_value = ExpressionEvaluator(statement.value.value)(
                    self.variables
                )
                return return_value

            else:
                raise ValueError("Unknown statement type " + statement.type)
