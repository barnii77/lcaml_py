import ast
import autopep8


def remove_type_annotations_and_format(code):
    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)

    # Function to remove type annotations from AST nodes
    def remove_annotations(node):
        if isinstance(node, (ast.AnnAssign, ast.arg)):
            node.annotation = None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
        elif isinstance(node, ast.ClassDef):
            node.bases = []

    # Traverse the AST and remove type annotations
    for node in ast.walk(tree):
        remove_annotations(node)

    # Convert the modified AST back to code
    modified_code = ast.unparse(tree)

    # Format the modified code using autopep8
    formatted_code = autopep8.fix_code(modified_code)

    return formatted_code


# Example usage
input_code = """
def greet(name: str) -> None:
     print("Hello, " + name)
"""

formatted_code = remove_type_annotations_and_format(input_code)
print(formatted_code)
