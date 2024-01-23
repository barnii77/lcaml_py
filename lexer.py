import re


class LexError(Exception):
    pass


class Token:
    """A token is a tuple of (type, value)"""
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

    def __str__(self):
        return "Token(" + self.type + ", " + self.value + ")"

    def __repr__(self):
        return "Token(" + self.type + ", " + self.value + ")"


class Syntax:
    """Contains a bunch of regexes for matching tokens"""
    def __init__(self):
        self.let = r"let\s"
        self.identifier = r"[a-zA-Z_][a-zA-Z0-9_]*"
        self.integer = r"[0-9]+"
        self.floating_point = r"[0-9]+\.[0-9]+"
        self.string_literal = r"\".*\""
        self.equals = r"="
        self.semicolon = r";"
        self.comment = r"--.*\n"
        operators = ('+', '-', '*', '/', '%', '!', '==', '!=', '<', '>', '~', '<=', '>=', '||', '&&', '|', '&')
        self.operator = '|'.join(f"\\{op}" for op in operators)

    def patterns(self):
        return vars(self).items()
        # return [self.let, self.identifier, self.integer, self.floating_point, self.string_literal]


# class LexState:
#     any: bool = False
#     string_literal: bool = False
#     integer: bool = False
#     floating_point: bool = False
#     identifier: bool = False
#     symbol: bool = False
#     comment: bool = False
#
#     position: int = 0
#
#     def __init__(self):
#         self.any = True
#         self.string_literal = False
#         self.integer = False
#         self.floating_point = False
#         self.identifier = False
#         self.symbol = False
#         self.comment = False
#         self.position = 0


class Lexer:
    def __init__(self, code: str, syntax: Syntax):
        self.code = code
        self.syntax = syntax
        # self.state = LexState()
        self.num_symbols = len(code)
        self.position = 0
        self.tokens = []

    def __call__(self):
        code = self.code[self.position:]

        while self.position < self.num_symbols and code.strip() != "":
            # match all the patterns in the syntax
            for kind, pattern in self.syntax.patterns():
                whitespaces = r"\s*"
                pattern = re.compile(f"^{whitespaces}({pattern}){whitespaces}.*")  # assert pattern is at the start of the string
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
            self.position += token_len
            code = code[token_len:]

        return self.tokens


if __name__ == "__main__":
    code = """
    let x = 10; -- x y z
    let y = 20;
    let z = x + y;
    """
    syntax = Syntax()
    lexer = Lexer(code, syntax)
    tokens = lexer()
    for token in tokens:
        print(token)