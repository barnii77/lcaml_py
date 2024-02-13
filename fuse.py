import argparse
import os
import re
import sys
from graphviz import Digraph

parser = argparse.ArgumentParser()
parser.add_argument('--fuse', action='store_true', help='Fuse Python files into one')
parser.add_argument('--remove-all-namespaces', action='store_true', help='Fuse Python files into one')
parser.add_argument('--files', type=str, default=None, help='Ordered list of files separated by spaces')
parser.add_argument('--deps', action='store_true', help='Print dependencies of Python files')
parser.add_argument('--print', action='store_true', help='Print the dependencies of Python files')
parser.add_argument('--visualize', action='store_true', help='Create a graph visualization of the dependencies')
parser.add_argument('-o', '--output', type=str, help='Output file name')
args = parser.parse_args()
if args.output.endswith('.pdf'):
    args.output = args.output[:-4]


def remove_imports(content, files):
    import_re = re.compile(r'import (\S+)|from (\S+) import (\S+)')
    imports = import_re.findall(content)
    imports = [imp for sublist in imports for imp in sublist if imp]
    imports = [imp for imp in imports if imp + '.py' in files]
    for imp in imports:
        if args.remove_all_namespaces:
            content = content.replace(f'import {imp}', '')
            content = content.replace(f'from {imp} import', '')
            content = content.replace(f'{imp}.', '')
    return content


def concatenate_files(files):
    contents = []
    for file in files:
        print(file)
        with open(file, 'r') as f:
            contents.append(f'################# {file} ##################')
            contents.append(f.read())
    return '\n\n'.join(contents)


if args.fuse:
    if args.files is None:
        dir = os.listdir()
        files = [f for f in dir if not os.path.isdir(f) and f.endswith('.py')]
    else:
        files = args.files.split()
        files = [f if f.endswith('.py') else f + '.py' for f in files]

    if args.output in files:
        print("Output file cannot be one of the input files, ignoring output file as input.")
        files.remove(args.output)
    if sys.argv[0] in files:
        print("Fuse script cannot be one of the input files, ignoring fuse script as input.")
        files.remove(sys.argv[0])
    fused = concatenate_files(files)
    fused = remove_imports(fused, files)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(fused)
    if args.print:
        print(fused)

if args.deps:

    python_files = [f for f in os.listdir() if f.endswith('.py')]
    import_re = re.compile(r'import (\S+)|from (\S+) import')
    dependencies = {}

    for file in python_files:
        with open(file, 'r') as f:
            content = f.read()

        imports = import_re.findall(content)
        imports = [imp for sublist in imports for imp in sublist if imp]
        imports = [imp for imp in imports if imp + '.py' in python_files]
        dependencies[file[:-3]] = imports

    if args.visualize:
        g = Digraph('G', filename=args.output, format='pdf')

        for file, deps in dependencies.items():
            for dep in deps:
                g.edge(file, dep)
        g.render()
        os.remove(args.output)
    if args.print:
        for file, deps in dependencies.items():
            print(f'{file}: {deps}')
