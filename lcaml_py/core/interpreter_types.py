from .operation_kind import OperationKind
from .resolvable import Resolvable
from typing import Any


class DType:
    """
    This class represents a data type enum.
    """
    ty = int

    INT: ty = 0
    FLOAT: ty = 1
    STRING: ty = 2
    BOOL: ty = 3
    UNIT: ty = 4
    FUNCTION: ty = 5
    STRUCT_TYPE: ty = 6
    TABLE: ty = 7
    EXTERN_PYTHON: ty = 8
    PY_OBJ: ty = 9
    LIST: ty = 10

    @staticmethod
    def name(code: int):
        name_map = {
            DType.INT: "Int",
            DType.FLOAT: "Float",
            DType.STRING: "String",
            DType.BOOL: "Bool",
            DType.UNIT: "UnitType",
            DType.FUNCTION: "Function",
            DType.STRUCT_TYPE: "StructType",
            DType.TABLE: "Table",
            DType.EXTERN_PYTHON: "ExternPython",
            DType.PY_OBJ: "PyObj",
            DType.LIST: "List",
        }
        if code in name_map:
            return name_map[code]
        raise ValueError(f"Unknown type {code}")

    def __repr__(self):
        return str(self)

    # rules for what type to return when performing an operation
    # on two objects
    # NOTE: if type is not in there, operation is unsupported
    _operation_result_rules = {
        OperationKind.ADD: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
            STRING: {
                STRING: STRING,
            },
            LIST: {
                LIST: LIST,
            },
        },
        OperationKind.SUB: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
        },
        OperationKind.MUL: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
                STRING: STRING,
                LIST: LIST,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
            STRING: {
                INT: STRING,
            },
            LIST: {
                INT: LIST,
            },
        },
        OperationKind.DIV: {
            INT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
        },
        OperationKind.IDIV: {
            INT: {
                INT: INT,
                FLOAT: INT,
            },
            FLOAT: {
                INT: INT,
                FLOAT: INT,
            },
        },
        OperationKind.POW: {
            INT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
        },
        OperationKind.MOD: {
            INT: {
                INT: INT,
                FLOAT: FLOAT,
            },
            FLOAT: {
                INT: FLOAT,
                FLOAT: FLOAT,
            },
        },
        OperationKind.EQ: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            STRING: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            BOOL: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            FUNCTION: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            STRUCT_TYPE: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            TABLE: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            UNIT: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            EXTERN_PYTHON: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            PY_OBJ: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            LIST: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
        },
        OperationKind.NEQ: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            STRING: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            BOOL: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            FUNCTION: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            STRUCT_TYPE: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            TABLE: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            UNIT: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            EXTERN_PYTHON: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            PY_OBJ: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
            LIST: {
                INT: BOOL,
                FLOAT: BOOL,
                STRING: BOOL,
                BOOL: BOOL,
                FUNCTION: BOOL,
                STRUCT_TYPE: BOOL,
                TABLE: BOOL,
                UNIT: BOOL,
                EXTERN_PYTHON: BOOL,
                PY_OBJ: BOOL,
                LIST: BOOL,
            },
        },
        OperationKind.LT: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.GT: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.LTE: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.GTE: {
            INT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
            FLOAT: {
                INT: BOOL,
                FLOAT: BOOL,
            },
        },
        OperationKind.OR: {
            BOOL: {
                BOOL: BOOL,
            },
        },
        OperationKind.AND: {
            BOOL: {
                BOOL: BOOL,
            },
        },
        OperationKind.BITOR: {
            INT: {
                INT: INT,
            },
        },
        OperationKind.BITAND: {
            INT: {
                INT: INT,
            },
        },
        OperationKind.LSH: {
            INT: {
                INT: INT,
            },
        },
        OperationKind.RSH: {
            INT: {
                INT: INT,
            },
        },
        OperationKind.BITXOR: {
            INT: {
                INT: INT,
            },
        },
        OperationKind.FLIP: {
            INT: INT,
        },
        OperationKind.NOT: {
            BOOL: BOOL,
        },
    }


class Object(Resolvable):
    """
    This class represents an object in the interpreter.

    Attributes:
        type: (DType) The type of the object (int, float, string, etc.)
        value: (Any) The value of the object.
    """

    def __init__(self, type: "DType.ty", value: Any):
        if isinstance(value, Object):
            value = value.value
        self.type = type
        self.value = value

    def resolve(self, context):
        return self

    def __hash__(self):
        return hash((self.type, self.value))

    def __str__(self):
        return f"Object::{DType.name(self.type)}(" + str(self.value) + ")"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return self.type == other.type and self.value == other.value

    def __bool__(self):
        return bool(self.value)

    def get(self, ident: "AstIdentifier"):
        return self.value.get(ident)

    def pow(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.POW, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value**other.value)

    def add(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.ADD, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value + other.value)

    def sub(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.SUB, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value - other.value)

    def mul(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.MUL, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value * other.value)

    def div(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.DIV, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value / other.value)

    def idiv(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.DIV, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value // other.value)

    def mod(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.MOD, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value % other.value)

    def eq(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.EQ, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value == other.value)

    def neq(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.NEQ, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value != other.value)

    def lt(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.LT, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value < other.value)

    def gt(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.GT, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value > other.value)

    def lte(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.LTE, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        result = self.value <= other.value
        return Object(return_type, self.value <= other.value)

    def gte(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.GTE, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value >= other.value)

    def bool_or(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.OR, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value or other.value)

    def bool_and(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.AND, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value and other.value)

    def bool_not(self):
        return_type = DType._operation_result_rules.get(OperationKind.NOT, {}).get(
            self.type
        )
        if return_type is None:
            raise TypeError(f"Unsupported operation for {self.type}")
        return Object(return_type, not self.value)

    def flip(self):
        return_type = DType._operation_result_rules.get(OperationKind.FLIP, {}).get(
            self.type
        )
        if return_type is None:
            raise TypeError(f"Unsupported operation for type {self.type}")
        return Object(return_type, ~self.value)

    def bitor(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.BITOR, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value | other.value)

    def bitand(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.BITAND, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value & other.value)

    def bitxor(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.BITXOR, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value ^ other.value)

    def lsh(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.LSH, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value << other.value)

    def rsh(self, other):
        if not isinstance(other, Object):
            raise TypeError(f"Expected type Object, got {type(other)}")

        return_type = (
            DType._operation_result_rules.get(OperationKind.RSH, {})
            .get(self.type, {})
            .get(other.type)
        )
        if return_type is None:
            raise TypeError(
                f"Unsupported operation between {self.type} and {other.type}"
            )
        return Object(return_type, self.value >> other.value)
