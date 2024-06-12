import pyffi


@pyffi.interface
def test_pyffi(n: int):
    print("wow call to python", n)
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


LML_EXPORTS = {
    "test_pyffi": test_pyffi,
    "test_pyffi_ext": test_pyffi_ext,
    "append_to_list": append_to_list,
}
