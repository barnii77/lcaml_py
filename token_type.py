# NOTE: Cannot change to integer because these values are used
# NOTE: to extract the token type from the syntax object
class TokenKind:
    LET = "let"
    IDENTIFIER = "identifier"
    INTEGER = "integer"
    FLOATING_POINT = "floating_point"
    STRING_LITERAL = "string_literal"
    BOOLEAN = "boolean"
    EQUALS = "equals"
    SEMICOLON = "semicolon"
    COMMENT = "comment"
    OPERATOR = "operator"
    LPAREN = "lparen"
    RPAREN = "rparen"
    LSQUARE = "lsquare"
    RSQUARE = "rsquare"
    LCURLY = "lcurly"
    RCURLY = "rcurly"


class Token:
    """
    This class represents a token found in the code.

    Attributes:
        type: type of token as string
        value: value of token as string

    """

    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

    def __str__(self):
        return "Token(" + self.type + ", " + self.value + ")"

    def __repr__(self):
        return "Token(" + self.type + ", " + self.value + ")"

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.type == other.type and self.value == other.value
