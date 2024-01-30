# Description
An interpreter written in python for a custom language LCaml.

# LCaml language concept
LCaml is an isoteric interpreted programming language written primarily to troll my programming teacher in school.
This is also one of the main reasons to write the interpreter in python as opposed to a more suitable language.
However, I might come back to LCaml in the future and write an interpreter in Rust, maybe even a compiler.

# Components of the interpreter
Lexer: CODE -> Tokens

Parser: Tokens -> AST

IR generator (?): AST -> IR

Interpreter: AST | IR -> EXECUTE

# Resources
[CodePulse](https://www.youtube.com/watch?v=Eythq9848Fg&list=PLZQftyCk7_SdoVexSmwy_tBgs7P0b97yD)

# TODO

- figure out and fix why ~x is -x instead of -x - 1

- Currently, when a function references values that are not passed as args, if the values go out of scope, the function will not work anymore
That means: Wrap everything in an object type and implement reference counting!! (Whoohooo! Yeaaahhhhh!! LOL pls kill me now)

- Use ParseError and LexError instead of ValueError

- Implement control flow
- (?) Implicit returns (like Rust)
