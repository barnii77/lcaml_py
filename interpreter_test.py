from timeit import timeit
from interpreter import Interpreter


# def python_run():
#     def factorial(n):
#         if n <= 1:
#             return 1
#         else:
#             return n * factorial(n - 1)

#     return factorial(5)


def run():
    interpreter.vm.variables = {}
    result = interpreter.execute()
    print("interpreter returned: ", result)


if __name__ == '__main__':
    with open('lcaml_test_code.lml', 'r') as f:
        code = f.read()
    interpreter = Interpreter(code)
    num_runs = 1  # 0000
    time_taken = timeit(run, number=num_runs)
    # time_taken_python = timeit(python_run, number=num_runs)
    print(f"Time taken to run {num_runs} times: {time_taken} seconds [average {time_taken / num_runs}]")
    # print(f"Python took {time_taken_python} seconds [average {time_taken_python / num_runs}]")
