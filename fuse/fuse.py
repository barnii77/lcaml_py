import argparse
import os
import re
import sys
from graphviz import Digraph
import fuse_utils

parser = argparse.ArgumentParser()
parser.add_argument("--fuse", action="store_true", help="Fuse Python files into one")
parser.add_argument(
    "--remove-all-namespaces", action="store_true", help="Fuse Python files into one"
)
parser.add_argument(
    "--files", type=str, default=None, help="Ordered list of files separated by spaces"
)
parser.add_argument(
    "--deps", action="store_true", help="Analyze dependencies"
)
parser.add_argument(
    "--print", action="store_true", help="Print the dependencies of Python files"
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="Create a graph visualization of the dependencies",
)
parser.add_argument("-o", "--output", type=str, help="Output file name. If using --visualize, the file should not have .pdf extension.")
args = parser.parse_args()
if args.output.endswith(".pdf"):
    args.output = args.output[:-4]


def indent(code):
    return "\n".join(map(lambda line: " " * 4 + line, code.split("\n")))


def refactor_imports(content, files):
    illegal_import_as_pattern = r"import \S+ as \S+|from \S+ import \S+ as \S+"
    illegal_import_as_re = re.compile(illegal_import_as_pattern)
    illegal_import_as_statements = illegal_import_as_re.findall(content)
    if illegal_import_as_statements:
        raise ValueError("import .. as .. is not supported")
    import_re = re.compile(r"import \S+|from \S+ import \S+")
    import_statements = import_re.findall(content)
    imports = [imp for sublist in import_statements for imp in sublist if imp]
    illegal_imports = [imp for imp in imports if imp + ".py" in files]
    legal_imports = [imp for imp in imports if imp not in illegal_imports]
    legal_import_statements = set()
    for i in legal_imports:
        ni = f"import {i}"
        fi = f"from {i} import "
        for s in import_statements:
            if s.startswith(ni) or s.startswith(fi):
                legal_import_statements.add(s)
    for imp in legal_import_statements:
        content = content.replace(imp, "")
    content = fuse_utils.replace_imports_with_namespaces(content)
    return content, "\n".join(legal_import_statements)


def read_files(files):
    contents = []
    for file in files:
        if args.print:
            print(file)
        with open(file, "r") as f:
            # contents.append(f'################# {file} ##################')
            contents.append(f.read())
    return contents


def dependency_graph(contents, files):
    import_re = re.compile(r"import (\S+)|from (\S+) import")
    dependencies = {}
    for content, file in zip(contents, files):
        imports = import_re.findall(content)
        imports = [imp for sublist in imports for imp in sublist if imp]
        imports = [imp for imp in imports if imp + ".py" in files]
        dependencies[file[:-3]] = imports
    return dependencies


if args.fuse:
    if args.files is None:
        dir = os.listdir()
        files = [f for f in dir if not os.path.isdir(f) and f.endswith(".py")]
    else:
        files = args.files.split()
        files = [f if f.endswith(".py") else f + ".py" for f in files]

    if args.output in files:
        print(
            "Output file cannot be one of the input files, ignoring output file as input."
        )
        files.remove(args.output)
    if sys.argv[0] in files:
        print(
            "Fuse script cannot be one of the input files, ignoring fuse script as input."
        )
        files.remove(sys.argv[0])
    contents = read_files(files)
    modnames = [file[:-3] for file in files]
    codes, import_stmt_groups = zip(
        *list(map(lambda c: refactor_imports(c, files), contents))
    )
    fused_code = ""
    for c, i, f in zip(codes, import_stmt_groups, modnames):
        factory_function = f"""
def {f}_factory():
    class __DEPS:
        pass

{indent(i)}

    class ModuleDef:
{indent(indent(c))}

    return ModuleDef, __DEPS
"""
        fused_code += f"############ {f} ############\n\n" + factory_function + "\n\n"

    for f in modnames:
        fused_code += f"{f}, {f}_deps = {f}_factory()\n"

    fused_code += "\n### Now fuse ###\n\n"

    dependencies = dependency_graph(contents, files)
    for mod, deps in dependencies.items():
        for dep in deps:
            fused_code += f"{mod}_deps.{dep} = {dep}\n"
        fused_code += "\n"

    fused_code += "\n### Your turn from here ###"

    if args.output:
        with open(args.output, "w") as f:
            f.write(fused_code)
    if args.print:
        print(fused_code)

elif args.deps:
    python_files = [f for f in os.listdir() if f.endswith(".py")]
    contents = read_files(python_files)
    dependencies = dependency_graph(contents, python_files)

    if args.visualize:
        if args.output.endswith(".pdf"):
            print("Warning: output file should not have .pdf extension")
        # g = Digraph("G", filename=args.output, format="pdf")

        # for file, deps in dependencies.items():
        #     for dep in deps:
        #         g.edge(file, dep)
        # g.render()
        fontsize = '24'
        g = Digraph("G", filename=args.output, format='pdf', graph_attr={'bgcolor': 'transparent', 'fontname': 'Arial', 'fontsize': fontsize, 'style': 'filled', 'fillcolor': 'white'})
        g.attr(label="Dependency Graph", labelloc="t", fontsize=fontsize)

        import_counts = {file: 0 for file in dependencies.keys()}  # how many things a file imports
        export_counts = {file: 0 for file in dependencies.keys()}  # how often a file is imported

        for file, deps in dependencies.items():
            for dep in deps:
                export_counts[dep] += 1
                import_counts[file] += 1
                g.edge(file, dep, fontsize=fontsize)

        for file in dependencies.keys():
            if import_counts[file] == 0 and export_counts[file] == 0:  # standalone file
                g.node(file, shape='ellipse', style='filled', fillcolor='lightgray', fontname='Helvetica', fontsize=fontsize, color='black')
            elif import_counts[file] == 0:  # file does not import anything
                g.node(file, shape='rect', style='filled', fillcolor='lightyellow', fontname='Helvetica', fontsize=fontsize, color='black')
            elif export_counts[file] == 0:  # file is never imported
                g.node(file, shape='circle', style='filled', fillcolor='red', fontname='Helvetica', fontsize=fontsize, color='black')
            elif import_counts[file] >= export_counts[file]:  # file imports more than it is imported itself
                g.node(file, shape='circle', style='filled', fillcolor='lightgreen', fontname='Helvetica', fontsize=fontsize, color='black')
            else:  # file is imported more than it imports itself
                g.node(file, shape='rect', style='filled', fillcolor='lightblue', fontname='Helvetica', fontsize=fontsize, color='black')
        g.render()
        os.remove(args.output)
    if args.print:
        for file, deps in dependencies.items():
            print(f"{file}: {deps}")
