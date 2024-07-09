from lcaml_py.core import pyffi

glob_counter = 1


@pyffi.interface
def test_pyffi(n: int):
    global glob_counter
    print("wow call to python with arg", n)
    print("this was call nr", glob_counter, "to python")
    glob_counter += 1
    return n + 99


@pyffi.interface
def test_pyffi_ext(_):
    # returns a python function
    def inner(n):
        print("lcaml just called a python function that was returned from a pyffi")
        return n + 10

    return inner


@pyffi.interface
def append_to_list(lst: list, item):
    print("lcaml called a python function that appends to a list")
    lst.append(item)
    return lst


def module():
    return {
        "test_pyffi": test_pyffi,
        "test_pyffi_ext": test_pyffi_ext,
        "append_to_list": append_to_list,
    }
