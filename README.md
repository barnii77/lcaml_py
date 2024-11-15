# Description
An interpreter written in python for a custom language LCaml.
LCaml is an incomplete, terribly slow, lua-style functional programming language.
Note that I did only little prep for this project (know basic steps of interpreter) and did not read a book or something for this project, I kinda just did what seemed the most obvious to me. Therefore, I might have made some unusual decisions.

# LCaml language concept
LCaml is an esoteric interpreted programming language written primarily to troll my programming teacher in school.
This is also one of the main reasons to write the interpreter in python as opposed to a more suitable language.
However, I might come back to LCaml in the future and write an interpreter in Rust, maybe even a compiler.

# LCaml foundation (lol)
The arch nemesis of the rust foundation, the LCaml foundation can be found here: https://discord.gg/bPNDJq9zmg

# Usage
## Interpreter
To run an lcaml program, just run `python -m lcaml_py.interpreter lcaml_file.lml` in your terminal. This assumes you have python installed. The project does not have any requirements that are not part of the python standard library, therefore there is no requirements.txt file.
Optionally, you can also provide a second filepath to a json file. This file will be used to overwrite the syntax of the program you are writing. Notice that if you import any files in the file you are executing, you will need to explicitly specify the syntax file to use for the imported file as a second argument to the import builtin. This argument is optional and the interpreter will, if not provided, use the default lcaml syntax.
So, to summarize
`python -m lcaml_py.interpreter program.lml -s syntax.json` to run a program using the lcaml interpreter. Note that the -s argument is optional.

## LTP-Compiler
To use the LCaml-to-Python compiler (that transpiles to python), use `python -m lcaml_py.compyla program.lml other_file.lml third_file.lml -s syntax.json`, where the -s argument is optional.
The compiler will create a `build/compiled_lcaml` folder if one doesn't already exist, remove any files potentially in there (rm -rf build/compiled_lcaml/*) and dump in all lcaml files with the same name (but .py extension).

# Syntax file format
The syntax json file format is the following:
```json
{
    "field_name": "field_pattern",
    "second_field_name": ["second_field_pattern", group_int]
}
```

Here, "field_name" has to be replaced with the field you want to configure, the names of which can be found in the Syntax class in the lcaml_lexer.py file.
Note that you cannot set the _this_keyword.
Here are some of the most important ones:
| name | description | default |
|------|-------------|---------|
| let | the let keyword that every assignment statement has to start with | "let\s" |
| return_keyword | - | "return\s" |
| struct_keyword | - | "struct(?![a-zA-Z0-9_])" |
| if_keyword | - | "if\s" |
| else_if_keyword | - | "else\s+if\s" |
| else_keyword | - | "else\s" |

You can set any value to either a pattern (string) or a list of [pattern (string), group (integer)] where the integer represents what regex group of the pattern to pick extract and use as the token value. The group defaults to 0.
Note that if you provide a list, you have to provide the group. It is only set for you if you only provide a string.

For an **example**, see tests/end_to_end/custom_syntax.json

# Components of the interpreter and compyla
```
Code            -> |     |                               |-> |Compyla| -> Emit Python
Syntax Settings -> |Lexer| -> Tokens -> |Parser| -> AST -|                                                                                    yes  |-> |Type analysis| -> |Compile with LLVM| -> Compiled function -> |Call comp func| -> output -> |Convert to LCaml object| ---|
                                                         |                                                                                         |        |                   |                                                                                                |
                                                         |-> |Execution Engine| -> Execute -> ... -> Function call -> Jit Compiler enabled? -------|        |  \*error          |  \*error                          *only simple code can be JIT-compiled, error = uncompilable  |
                                                                A                                                                                  |       \\/                 \\/                                                                                               |
                                                                |                                                                             no   |-> |Spawn new execution engine| -> output -----------------------------------------------------------------------------------|
                                                                |________________________________________________________________________________________________________________________________________________________________________________________________________________|
```
