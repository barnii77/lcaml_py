import os
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
    for file in os.listdir("tests/end_to_end"):
        with open(f'tests/end_to_end/{file}', 'r') as f:
            code = f.read()
        print(f"Running test: {file}")
        print()
        interpreter = Interpreter(code)
        num_runs = 1  # 000
        time_taken = timeit(run, number=num_runs)
        # time_taken_python = timeit(python_run, number=num_runs)
        print(f"Time taken to run {num_runs} times: {time_taken} seconds [average {time_taken / num_runs}]")
        # print(f"Python took {time_taken_python} seconds [average {time_taken_python / num_runs}]")
        print("\n----------------\n")
