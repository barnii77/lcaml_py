import hashlib
from typing import List
from core.token_type import TokenKind, Token


LCAML_RECURSION_LIMIT = 10**7
_uuid = 0
NAME_GEN_N_HASH_DIGITS = 24
TokenStream = List[Token]


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
        + hashlib.md5(str(_uuid).encode("utf-8")).hexdigest()[:NAME_GEN_N_HASH_DIGITS]
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
