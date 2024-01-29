import re
from typing import List, Dict
from token_type import Token


class LexError(Exception):
    """
    This exception is raised when the lexer cannot find a matching pattern to form a token.
    """

    pass


class Syntax:
    """
    This class defines the syntax of the language by containing named regex patterns.

    Attributes:
        let:
        identifier:
        integer:
        floating_point:
        string_literal:
        equals:
        semicolon:
        comment:
        operator:

    """

    def __init__(self, **kwargs):
        self.let = r"let\s"
        self.identifier = r"[a-zA-Z_][a-zA-Z0-9_]*"
        self.floating_point = r"[0-9]+\.[0-9]+"  # be careful - define this before int so it first checks this
        self.integer = r"[0-9]+"
        self.boolean = r"true|false"
        self.string_literal = r"\".*\""
        self.equals = r"="
        self.semicolon = r";"
        self.comment = r"--.*\n"
        operators = (
            "+",
            "-",
            "*",
            "/",
            "%",
            "!",
            "==",
            "!=",
            "<",
            ">",
            "~",
            "<=",
            ">=",
            "||",
            "&&",
            "|",
            "&",
        )
        self.operator = "|".join("".join(f"\\{c}" for c in op) for op in operators)
        self.lparen = r"\("
        self.rparen = r"\)"
        self.lsquare = r"\["
        self.rsquare = r"\]"
        self.lcurly = r"\{"
        self.rcurly = r"\}"
        self.bar = r"\|"
        self.comma = r","

        self.set_custom(kwargs)

    def set_custom(self, kwargs: Dict[str, str]):
        """
        This function is used by the initializer to set the custom syntax patterns.

        Args:
            kwargs: the kwargs passed to the init function

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def patterns(self):
        return vars(self).items()


class Lexer:
    """

    Attributes:
        code (str): The code to lex
        syntax (Syntax): The syntax to use for lexing
        num_symbols (int): The number of symbols in the code
        tokens (list[Token]): The tokens found in the code (when called multiple times, will not recompute)

    """

    def __init__(self, code: str, syntax: Syntax):
        self.code = code
        self.syntax = syntax
        # self.state = LexState()
        self.num_symbols = len(code)
        self.tokens = []

    def __call__(self) -> List[Token]:
        """

        Returns:
            list[Token]: A list of tokens

        Raises:
            LexError: If no matching pattern is found

        """
        if self.tokens:  # if already lexed, just return the tokens
            return self.tokens

        code = self.code

        while code.strip() != "":
            # match all the patterns in the syntax
            for kind, pattern in self.syntax.patterns():
                pattern = re.compile(
                    f"^\\s*({pattern}).*"
                )  # assert pattern is at the start of the string
                m = pattern.match(code)
                if m:
                    break
            else:
                raise LexError("No matching pattern for " + code)

            # save the match as a token
            token_value = m.group(1).strip()
            self.tokens.append(Token(kind, token_value))

            # increment position
            token_len = m.end(1)
            code = code[token_len:]

        return self.tokens


if __name__ == "__main__":
    code = """
    let x = 10; -- x y z
    let y = 20;
    let z = x + y + zahl;
    """
    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    for token in tokens:
        print(token)
