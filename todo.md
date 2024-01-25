# TODO

- begin by parsing all the individual expr that make up the operations
- this is recursive, because parentheses count as such exprs
- also, function calls / defs etc count as such
- Expression can contain Closure, FunctionCall, Operation, Variable or Constant
- Operation contains 2 Expressions and an Operator:
```py
class Expression:
    def __init__(self, type: int, value: ):
        self.type = type
        self.value = value


class Operation:
    def __init__(self, left: Expression, right: Expression, optype: int):  # optype in enum
        self.left = left
        self.right = right
        self.type = optype
```
- Add operation type for operations that captures left and right side
- When multiple ops after one another, go from left to right
- if parentheses, parse inner first (to recursive Ops)
```rs
enum Operation {
    Add(i32, i32),
    Sub(i32, i32),
    Mul(i32, i32),
    Div(i32, i32),
    Mod(i32, i32),
}

impl Operation {
    fn compute(&self) -> Result<i32, OperationError> {
        match self {
            Operation::Add(x, y) => Ok(x + y),
            Operation::Sub(x, y) => Ok(x - y),
            Operation::Mul(x, y) => Ok(x * y),
            Operation::Div(x, y) => {
                if y == 0 {
                    Err(OperationError::ZeroDivisionError)
                } else {
                    Ok(x / y)
                }
            }
            Operation::Mod(x, y) => {
                if y == 0 {
                    Err(OperationError::ZeroDivisionError)
                } else {
                    Ok(x % y)
                }
            }
        }
    }
}
```

```py
class OperationType:
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4


class Operation:
    def __init__(self, type: int, x: int, y: int):
        self.type = type
        self.x = x
        self.y = y

    def compute(self):
        if self.type == OperationType.ADD:
            return self.x + self.y
        elif self.type = OperationType.SUB:
            return self.x - self.y
        ...
```
