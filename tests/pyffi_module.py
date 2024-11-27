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


@pyffi.interface
def call_lcaml_callback(lcb):
    return lcb(3)


@pyffi.interface
def use_cyclic_ds(cds):
    print("cyclic_ds['table']", cds["table"])
    out = {"y": {"z": -2}, **cds}
    y = out["y"]
    y["r"] = out
    return out


@pyffi.pymodule
def module(context):
    d = {
        "test_pyffi": test_pyffi,
        "test_pyffi_ext": test_pyffi_ext,
        "append_to_list": append_to_list,
        "call_lcaml_callback": call_lcaml_callback,
        "use_cyclic_ds": use_cyclic_ds,
    }
    return d
