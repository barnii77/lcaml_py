import random
import string
from typing import List
from .token_type import TokenKind, Token


LCAML_RECURSION_LIMIT = 2**31 - 1
_uuid = 0
NAME_GEN_N_RAND_CHARS = 24
TokenStream = List[Token]


def clip(x, lower, upper):
    return max(lower, min(upper, x))


def get_marked_code_snippet(code_lines, marked_line_idx, window_size=1):
    if isinstance(code_lines, str):
        code_lines = code_lines.splitlines()
    window_size = (window_size - 1) / 2
    printed_line_nums = list(
        range(
            clip(round(marked_line_idx - window_size + 0.1), 0, len(code_lines) - 1),
            clip(round(marked_line_idx + window_size + 0.1), 0, len(code_lines) - 1)
            + 1,
        )
    )
    max_line_num_len = max(map(lambda x: len(str(x + 1)), printed_line_nums))
    return "\n".join(
        map(
            lambda line: f"{('->' if line == marked_line_idx else ''):<3}{str(line + 1).rjust(max_line_num_len)}| {code_lines[line]}",
            printed_line_nums,
        )
    )


def expect_only_expression(to_python_return: tuple[str, str, str]) -> str:
    pre_insert, expr, post_insert = to_python_return
    if pre_insert or post_insert:
        raise ValueError("Expected only expression")
    return expr


def indent(string: str, level: int = 1):
    return "\n".join("    " * level + line for line in string.split("\n"))


def get_unique_name():
    global _uuid
    _uuid += 1
    return (
        "_"
        + "".join(random.choices(string.hexdigits, k=NAME_GEN_N_RAND_CHARS))
        + str(_uuid)
    )


def unreachable():
    raise Exception("unreachable")


def is_token_pair(token1: Token, token2: Token):
    return (
        (token1.type == TokenKind.LPAREN and token2.type == TokenKind.RPAREN)
        or (token1.type == TokenKind.LSQUARE and token2.type == TokenKind.RSQUARE)
        or (token1.type == TokenKind.LCURLY and token2.type == TokenKind.RCURLY)
    )


def split_at_context_end(
    stream: TokenStream, terminating_token: Token
) -> tuple[TokenStream, TokenStream]:
    context_stack = []
    terminating_idx = 0
    while terminating_idx < len(stream):
        token = stream[terminating_idx]

        # add token if it starts a context
        if token.type in (
            TokenKind.LPAREN,
            TokenKind.LSQUARE,
            TokenKind.LCURLY,
        ):
            context_stack.append(token)

        # remove token if it ends a context
        if context_stack and is_token_pair(context_stack[-1], token):
            context_stack.pop()
        # break if terminating token found and no inner contexts
        elif not context_stack and token == terminating_token:  # no inner contexts
            break

        terminating_idx += 1

    body_stream, remaining_stream = (
        stream[:terminating_idx],
        stream[terminating_idx:],  # skip RCURLY
    )

    return body_stream, remaining_stream


class PhantomType:
    """
    A phantom type that will always be equal to everything.
    """

    def __init__(self, *_, **__):
        pass

    def __eq__(self, _):
        return True

    def __neq__(self, _):
        return False

    def __repr__(self):
        return "PhantomType()"

    def __radd__(self, other):
        if not isinstance(other, str):
            raise TypeError("Can only add PhantomType to string")
        return other + "PhantomType()"


class EqualsAny:
    def __init__(self, *args):
        self.args = args

    def __eq__(self, other):
        return other in self.args
