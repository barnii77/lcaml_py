class OperationKind:
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    IDIV = 4
    MOD = 5
    NOT = 6
    EQ = 7
    NEQ = 8
    LT = 9
    GT = 10
    FLIP = 11
    LTE = 12
    GTE = 13
    OR = 14
    AND = 15
    BITOR = 16
    BITAND = 17
    POW = 18
    LSH = 19
    RSH = 20
    BITXOR = 21

    _unary = [NOT, FLIP]
