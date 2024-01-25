from interpreter_types import Object
import lcaml_parser


class ExpressionEvaluator:
    """
    This class evaluates an expression.
    """

    def __init__(self, expression: parser.AstExpression):
        self.expression = expression

    def __call__(self, variables: dict[str, Object]) -> Object:
        return eval(str(self.expression), {"__builtins__": None}, {"variables": variables})