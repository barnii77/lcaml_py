############ b ############


def b_factory():
    class __DEPS:
        pass

    import abc
    from typing import Union

    class ModuleDef:
        
        
        
        x = Union[int, str]
        __DEPS.a = abc.ABCMeta()
        print(x, __DEPS.a)

    return ModuleDef, __DEPS


############ a ############


def a_factory():
    class __DEPS:
        pass

    import abc
    from typing import Union

    class ModuleDef:
        x2 = Union[int, str]
        a2 = abc.ABCMeta()
        print(x, __DEPS.b)

    return ModuleDef, __DEPS


############ c ############


def c_factory():
    class __DEPS:
        pass

    

    class ModuleDef:
        x = 3

    return ModuleDef, __DEPS


b, b_deps = b_factory()
a, a_deps = a_factory()
c, c_deps = c_factory()

### Now fuse ###

b_deps.a = a

a_deps.b = b
a_deps.c = c



### Your turn from here ###