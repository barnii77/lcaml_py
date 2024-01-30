class OperationKind:
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4
    NOT = 5
    EQ = 6
    NEQ = 7
    LT = 8
    GT = 9
    FLIP = 10
    LTE = 11
    GTE = 12
    OR = 13
    AND = 14
    BITOR = 15
    BITAND = 16

    _unary = [NOT, FLIP]
