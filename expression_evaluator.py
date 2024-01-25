from interpreter_types import Object, DType
import lcaml_parser


class ExpressionEvaluator:
    """
    This class evaluates an expression.
    """

    def __init__(self, expression: lcaml_parser.AstExpression):
        self.expression = expression

    def __call__(self, variables: dict[str, Object]) -> Object:
        result = eval(
            str(self.expression), {"__builtins__": None}, {"variables": variables}
        )
        result_type = type(result)
        if result_type == int:
            return Object(result, DType.INT)
        elif result_type == float:
            return Object(result, DType.FLOAT)
        elif result_type == bool:
            return Object(result, DType.BOOL)
        elif result_type == str:
            return Object(result, DType.STRING)
        else:
            raise ValueError(f"Unknown type {result_type}")
