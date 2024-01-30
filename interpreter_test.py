from interpreter import Interpreter


with open('lcaml_test_code.lml', 'r') as f:
    code = f.read()
interpreter = Interpreter(code)
result = interpreter.execute()
print(result)
