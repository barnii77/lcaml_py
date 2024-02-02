# Description
An interpreter written in python for a custom language LCaml.
Note that I did only little prep for this project (know basic stepos of interpreter) and did not read a book or somethign for this project, I kinda just did what seemed the most obvious to me. Therefor, I might have made some unusual decisions.

# LCaml language concept
LCaml is an esoteric interpreted programming language written primarily to troll my programming teacher in school.
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

- Currently, when a function references values that are not passed as args, if the values go out of scope, the function will not work anymore
    - This means: collect local symbols (all identifiers = functions + variables) and return them
    - Everything needs to collect all the symbols of anything it contains (eg: function containing control flow containing expression) and bubble it up (so to speak)
    - Functions need to also save a dict of IDs and Nones inside themselves as well as return them as a list (the dict is just for avoiding copies later)
    - When a function is instantiated in the program, the values of these symbols need to be resolved
    - FunctionCalls then need to use those and first create context containing globals, then update it with the values of the bound values in that dict, then finally the locals (args)

- Use ParseError and LexError instead of ValueError

- (??) Loops (As functions????)
- (?) Implicit returns (like Rust)
