############ b ############


def b_factory():
    class __DEPS:
        pass

    from typing import Union
    import abc

    class ModuleDef:
        class B:
            def __init__(self):
                x = Union[int, str]
                __DEPS.a = abc.ABCMeta()
                print(x, __DEPS.a)

    return ModuleDef, __DEPS


############ a ############


def a_factory():
    class __DEPS:
        pass

    import abc
    from typing import Union, List, Dict

    class ModuleDef:
        class X:
            def __init__(self):
                self.x = 1
                x2 = Union[int, str]
                print(self.x, __DEPS.b, __DEPS.c.C, __DEPS.c.D, x2)

    return ModuleDef, __DEPS


############ c ############


def c_factory():
    class __DEPS:
        pass

    class ModuleDef:
        class C:
            x = 3

        class D:
            y = 4

    return ModuleDef, __DEPS


b, b_deps = b_factory()
a, a_deps = a_factory()
c, c_deps = c_factory()

### Now fuse ###

b_deps.a = a

a_deps.b = b
a_deps.c = c


### Your turn from here ###
