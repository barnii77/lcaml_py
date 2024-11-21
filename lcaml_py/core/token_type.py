# NOTE: Cannot change to integer because these values are used
# NOTE: to extract the token type from the syntax object
class TokenKind:
    _PHANTOM = "<phantom>"
    LET = "let"
    RETURN = "return_keyword"
    STRUCT = "struct_keyword"
    IF = "if_keyword"
    ELSE_IF = "else_if_keyword"
    ELSE = "else_keyword"
    WHILE = "while_keyword"
    IDENTIFIER = "identifier"
    FUNCTION_ARGS = "function_args"
    UNIT_TYPE = "unit_type"
    INTEGER = "integer"
    FLOATING_POINT = "floating_point"
    STRING_LITERAL = "string_literal"
    BOOLEAN = "boolean"
    EQUALS = "equals"
    SEMICOLON = "semicolon"
    DOT = "dot"
    COLON = "colon"
    COMMA = "comma"
    COMMENT = "comment"
    OPERATOR = "operator"
    LPAREN = "lparen"
    RPAREN = "rparen"
    LSQUARE = "lsquare"
    RSQUARE = "rsquare"
    LCURLY = "lcurly"
    RCURLY = "rcurly"
    _builtin_types = [
        UNIT_TYPE,
        INTEGER,
        FLOATING_POINT,
        STRING_LITERAL,
        BOOLEAN,
    ]


class Token:
    """
    This class represents a token found in the code.

    Attributes:
        type: type of token as string
        value: value of token as string

    """

    def __init__(self, type: str, value: str, line: int = -1):
        self.type = type
        self.value = value
        self.line = line

    def __str__(self):
        return "Token(" + self.type + ", " + self.value + ")"

    def __repr__(self):
        return "Token(" + self.type + ", " + self.value + ")"

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.type == other.type and self.value == other.value
