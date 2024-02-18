import re


def replace_imports_with_namespaces(code):
    import_from_pattern = r"^from\s+(\S+)\s+import\s+(.*?)\s*$"
    import_pattern = r"^import\s+(.*?)\s*$"
    imported_symbols = []
    # Find "from ... import" statements and store them in the list
    from_imports = re.findall(import_from_pattern, code, flags=re.MULTILINE)
    from_imported_symbols_with_namespaces = []
    for module, imported in from_imports:
        if "," in imported:
            # Handle multiple imports from the same module
            for symbol in imported.split(","):
                symbol = symbol.strip()
                imported_symbols.append(f"{module}.{symbol}")
                from_imported_symbols_with_namespaces.append((symbol, module))
        else:
            imported_symbols.append(f"{module}.{imported.strip()}")
    # Find "import" statements and store them in the list
    imports = re.findall(import_pattern, code, flags=re.MULTILINE)
    for imported in imports:
        if "," in imported:
            # Handle multiple imports
            for symbol in imported.split(","):
                symbol = symbol.strip()
                imported_symbols.append(symbol)
        else:
            imported_symbols.append(imported.strip())
    # Replace "from ... import" statements
    code = re.sub(
        import_from_pattern, lambda m: replace_import_from(m), code, flags=re.MULTILINE
    )
    # Replace "import" statements
    code = add_deps(code, imported_symbols, from_imported_symbols_with_namespaces)
    return code


def replace_import_from(m):
    module = m.group(1)
    imports = m.group(2).split(",")
    namespace = module
    replacements = [f"{namespace}.{imp.strip()}" for imp in imports]
    return f"import {', '.join(replacements)}"


def add_deps(code, deps, from_imported_symbols_with_namespaces):
    """
    Function that replaces all uses of names from imported modules with the namespace __DEPS.{namespace}.{other_stuff}

    Args:
        code: (str) Python code

    """
    for dep in deps:
        code = code.replace(dep, f"__DEPS.{dep}")
    lines = []
    for line in code.splitlines():
        if not line.startswith('import'):
            for symbol, module in from_imported_symbols_with_namespaces:
                line = line.replace(symbol, f"__DEPS.{module}.{symbol}")
        lines.append(line)
    code = "\n".join(lines)
    return code


if __name__ == "__main__":
    # Example usage:
    input_code = """
from module import func1, func2
import module2
x = func1()
y = func2()
z = module2.func3()
"""
    output_code = replace_imports_with_namespaces(input_code)
    print(output_code)
