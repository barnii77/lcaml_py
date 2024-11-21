import re
from typing import List, Dict, Callable
from .token_type import Token


class LexError(Exception):
    """
    This exception is raised when the lexer cannot find a matching pattern to form a token.
    """

    pass


class Syntax:
    """
    This class defines the syntax of the language by containing named regex patterns.
    """
    _this_intrinsic = r"__this"
    _vm_intrinsic = r"__vm"
    _interpreter_intrinsic = r"__interpreter"

    def __init__(self, **kwargs):
        # non-syntax-pattern stuff
        self._extract_fn_args: Callable = lambda args_str: [
            a for a in args_str.strip("|").split(" ") if a
        ]

        # keywords
        self.let = r"let\s"
        self.return_keyword = r"return\s"
        self.struct_keyword = r"struct(?![a-zA-Z0-9_])"
        self.if_keyword = r"if"
        self.else_if_keyword = r"else\s+if"
        self.else_keyword = r"else"
        self.while_keyword = "while"

        # types
        self.unit_type = r"\(\)"
        self.floating_point = r"-?[0-9]+\.[0-9]+"  # be careful - define this before int so it first checks this
        self.integer = r"-?[0-9]+"
        self._true = r"true"
        self.boolean = r"true|false"
        self.string_literal = r"\"(.*?)\"", 1
        self.comment = r"--.*\n"

        # identifiers and builtins
        self.identifier = r"[a-zA-Z_][a-zA-Z0-9_]*"
        self.function_args = (
            f"\\|\\s*({self.identifier}\\s*)*{self.identifier}\\s*\\|"
        )

        # operators
        operators = (
            "==",
            "!=",
            "!",
            "~",
            "**",
            "*",
            "//",
            "/",
            "%",
            "+",
            "<<",
            ">>",
            "-",
            "<=",
            ">=",
            "<",
            ">",
            "||",
            "&&",
            "|",
            "&",
            "^",
        )
        self.operator = "|".join("".join(f"\\{c}" for c in op) for op in operators)

        # symbols
        self.dot = r"\."
        self.comma = r","
        self.equals = r"="
        self.semicolon = r";"
        self.colon = r":"
        self.lparen = r"\("
        self.rparen = r"\)"
        self.lsquare = r"\["
        self.rsquare = r"\]"
        self.lcurly = r"\{"
        self.rcurly = r"\}"

        self.set_custom(kwargs)
        self.compiled_patterns = self.get_compiled_patterns()

    def set_custom(self, kwargs: Dict[str, str]):
        """
        This function is used by the initializer to set the custom syntax patterns.

        Args:
            kwargs: the kwargs passed to the init function

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def patterns(self):
        """

        Returns:
            Iterable[Tuple[str, str]]: A iterable of tuples of the form [(name, pattern), ...] that excludes fields starting with _ (_fieldname)

        """
        return filter(lambda i: not i[0].startswith("_"), vars(self).items())

    def get_compiled_patterns(self) -> dict:
        result = {}
        for k, pattern_info in self.patterns():
            if not isinstance(pattern_info, (tuple, list)):
                if not isinstance(pattern_info, str):
                    raise TypeError(f"Invalid pattern_info type: {type(pattern_info)}")
                pattern_info = (pattern_info, 0)
            pattern, group = pattern_info
            result[k] = re.compile(f"^\\s*({pattern})"), group + 1
        return result


class TokenList(list):
    def __init__(self):
        super().__init__()
        # this is needed
        self.__file = "<unknown>"


class Lexer:
    """

    Attributes:
        code (str): The code to lex
        syntax (Syntax): The syntax to use for lexing
        num_symbols (int): The number of symbols in the code
        tokens (list[Token]): The tokens found in the code (when called multiple times, will not recompute)

    """

    def __init__(self, code: str, syntax: Syntax, _file: str = "<unknown>"):
        self.code = code
        self.syntax = syntax
        self.num_symbols = len(code)
        self.tokens = TokenList()
        self.tokens.__file = _file

    def __call__(self) -> List[Token]:
        """

        Returns:
            list[Token]: A list of tokens

        Raises:
            LexError: If no matching pattern is found

        """
        if self.tokens:  # if already lexed, just return the tokens
            return self.tokens

        line = 1
        code = self.code

        while code.strip() != "":
            # match all the patterns in the syntax
            for kind, pattern_info in self.syntax.compiled_patterns.items():
                pattern, group = pattern_info
                m = pattern.match(code)
                if m:
                    break
            else:
                raise LexError(f"Syntax Error on line {line}: No matching pattern for ```{code[:20]}.....```")

            # save the match as a token
            token_value = m.group(group)
            self.tokens.append(Token(kind, token_value, line))

            # increment position
            token_len = m.end(1)
            line += code[:token_len].count("\n")
            while token_len < len(code) and code[token_len].isspace():
                if code[token_len] == "\n":
                    line += 1
                token_len += 1
            code = code[token_len:]

        return self.tokens
