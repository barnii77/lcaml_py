from token_type import Token, TokenKind
from ast_related import AstRelated


class AstIdentifier(AstRelated):
    """

    Attributes:
        name: name of identifier

    """

    def __init__(self, name_token: Token):
        if name_token.type != TokenKind.IDENTIFIER:
            raise ValueError("Token must be an identifier")
        self.name = name_token.value

    def __str__(self):
        return "AstIdentifier(" + self.name + ")"

    def __eq__(self, other):
        if not isinstance(other, AstIdentifier):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class AstStatementType:
    ASSIGNMENT = 0
    RETURN = 1
    CONTROL_FLOW = 2


class AstAssignment(AstRelated):
    """

    Attributes:
        identifier: identifier to write to
        value: (Expression) value to assign

    """

    def __init__(self, identifier: AstIdentifier, value):
        self.identifier = identifier
        self.value = value

    def __str__(self):
        return "AstAssignment(" + str(self.identifier) + ", " + str(self.value) + ")"


class AstReturn(AstRelated):
    """

    Attributes:
        value: (Expression) value to return

    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "AstReturn(" + str(self.value) + ")"


class AstControlFlow(AstRelated):
    """
    Attributes:
        conditions: (List[Expression]) list of conditions
    """
    def __init__(self, conditions: list):
        self.conditions = conditions

    def __str__(self):
        return "AstControlFlow(" + str(self.conditions) + ")"
