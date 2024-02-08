# Description
An interpreter written in python for a custom language LCaml.
LCaml is an incomplete, terribly slow, lua-style functional programming language.
Note that I did only little prep for this project (know basic stepos of interpreter) and did not read a book or somethign for this project, I kinda just did what seemed the most obvious to me. Therefor, I might have made some unusual decisions.

# LCaml language concept
LCaml is an esoteric interpreted programming language written primarily to troll my programming teacher in school.
This is also one of the main reasons to write the interpreter in python as opposed to a more suitable language.
However, I might come back to LCaml in the future and write an interpreter in Rust, maybe even a compiler.

# Components of the interpreter
Lexer: CODE -> Tokens

Parser: Tokens -> AST

IR generator (?): AST -> IR

Interpreter (.rs?): AST | IR -> EXECUTE

# TODO

- Algebraic data types

- Use ParseError and LexError instead of ValueError

- (??) Loops (As functions????)
- (?) Implicit returns (like Rust)
