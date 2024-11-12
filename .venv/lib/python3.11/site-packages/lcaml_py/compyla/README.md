# lcaml_compyla
LCaml compyla (pronounced "LCaml compiler") is a compiler that transpiles lcaml to python. Thus the name *lcaml* *com*piler to *py*thon *la*nguange. It is based on the lcaml_py interpreter implementation and uses the same parser (that it imports from lcaml_py). It uses that parser to parse a source file and throws the AST into the python transpiler.

# python transpiler
The python transpiler module of this project stores the mappings from lcaml to python builtins and is able to lower any lcaml code without external dependencies to python. It will create a `build` folder and recreates the same file structure that you are compiling (you pass all files that shall be compiled to the compiler and it creates all folders it finds those files in, with the same structure, in the build folder).

# why?
+ Why not? (No def not because lcaml is slow as hell no of course not)
+ Opportunity to call stupid name that actually fits the project
