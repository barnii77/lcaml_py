import ast
import random
import string


def refactor_python_code(input_code):
    # Parse the input Python code
    tree = ast.parse(input_code)

    # Function to generate random identifier names
    def random_identifier():
        return "".join(random.choices(string.ascii_lowercase, k=60))

    # Function to recursively traverse the abstract syntax tree and rename identifiers
    def rename_identifiers(node):
        if isinstance(node, ast.Name):
            node.id = random_identifier()
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.FunctionDef):
                child_node.name = random_identifier()
            elif isinstance(child_node, ast.ClassDef):
                child_node.name = random_identifier()
            elif isinstance(child_node, ast.Name):
                child_node.id = random_identifier()

    # Rename identifiers in the parsed tree
    rename_identifiers(tree)

    # Generate the refactored code from the modified tree
    refactored_code = ast.unparse(tree)
    return refactored_code


# Example usage:
if __name__ == "__main__":
    original_code = """
def greet(name):
    print("Hello, " + name)

greet("World")
"""

    refactored_code = refactor_python_code(original_code)
    print(refactored_code)
