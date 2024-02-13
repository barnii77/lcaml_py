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


def indent(code):
	return '\n'.join(map(lambda line: ' ' * 4 + line, code.split('\n')))


def refactor_imports(content, files):
    # TODO replace all from imports sith normal ones
    import_re = re.compile(r'import (\S+)|from (\S+) import (\S+)')
    import_statements = import_re.findall(content)
    imports = [imp for sublist in import_statements for imp in sublist if imp]
    illegal_imports = [imp for imp in imports if imp + '.py' in files]
    legal_imports = [imp for imp in imports if imp not in illegal_imports]
    for imp in import_statements:
        content = content.replace(imp, '')
    for imp in illegal_imports:
    	content = content.replace(f'{imp}.', '__DEPS.{imp}.')
    legal_import_statements = set()
    for i in legal_imports:
    	ni = f'import {i}'
    	fi = f'from {i} import '
    	for s in import_statements:
    		if s.startswith(ni) or s.startswith(fi):
    			legal_import_statements.add(s)
    return content, '\n'.join(legal_import_statements)


def read_files(files):
    contents = []
    for file in files:
        print(file)
        with open(file, 'r') as f:
            #contents.append(f'################# {file} ##################')
            contents.append(f.read())
    return contents


def dependency_graph(contents, files):
	import_re = re.compile(r'import (\S+)|from (\S+) import')
    dependencies = {}
    for content in contents:
        imports = import_re.findall(content)
        imports = [imp for sublist in imports for imp in sublist if imp]
        imports = [imp for imp in imports if imp + '.py' in python_files]
        dependencies[file[:-3]] = imports
    return dependencies


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
    contents = read_files(files)
    modnames = [files[:-3] for file in files]
    codes, import_stmt_groups = zip(*list(map(lambda c:  refactor_imports(c, files), contents)))
    fused_code = ''
    for c, i, f in zip(codes, import_statement_groups, modnames):
    	factory_function = f"""
def {f}_factory():
	class __DEPS:
		pass

{indent(i)}
	
	class ModuleDef:
{indent(indent(c))}

	return ModuleDef, __DEPS
"""
		fused_code += f'############ {f} ############\n\n' + factory_function + '\n\n'
    
    for f in modnames:
    	fused_code += f'{f}, {f}_deps = {f}_factory()\n'
    
    fused_code += '\n### Now fuse ###\n\n'
    
    dependencies = dependency_graph(contents, files)
    for mod, deps in dependencies.items():
    	for dep in deps:
    		fused_code += f'{mod}_deps.{dep} = {dep}\n'
    	fused_code += '\n'
    
    fused_code += '\n### Your turn from here ###'
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(fused)
    if args.print:
        print(fused)

if args.deps:

    python_files = [f for f in os.listdir() if f.endswith('.py')]
    contents = read_files(python_files)
    dependencies = dependency_graph(contents, python_files)

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
