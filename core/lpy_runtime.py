def lpy_get(thing, key):
    return thing[key] if key in thing else None


def lpy_set(thing, key, value):
    thing[key] = value


def lpy_keys(thing):
    return thing.keys()


def lpy_values(thing):
    return thing.values()


def lpy_fuse(thing1, thing2):
    return {**thing1, **thing2}


def lpy_print(*args):
    print(*args, sep="", end="")


def lpy_println(*args):
    print(*args, sep="")


def lpy_input(prompt):
    return input(prompt)


def lpy_is_instance(table, struct_type):
    if not isinstance(table, dict):
        raise TypeError(f"Expected table to be a dict, got {type(table)}")
    if not isinstance(struct_type, set):
        raise TypeError(
            f"Expected struct_type to be a StructType, got {type(struct_type)}"
        )
    return set(table.keys()) == struct_type


def lpy_string(x):
    try:
        return str(x)
    except ValueError:
        pass
