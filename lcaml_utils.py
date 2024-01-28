def unreachable():
    raise Exception("unreachable")


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
