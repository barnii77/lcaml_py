from parser_types import AstIdentifier
from interpreter_types import Object


class Gettable:
    def get(self, ident: AstIdentifier) -> Object:
        raise NotImplementedError()
