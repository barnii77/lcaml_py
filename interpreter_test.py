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
    with open('lcaml_test_code.lml', 'r') as f:
        code = f.read()
    interpreter = Interpreter(code)
    result = interpreter.execute()
    print(result)


if __name__ == '__main__':
    num_runs = 10000
    time_taken = timeit(run, number=num_runs)
    # time_taken_python = timeit(python_run, number=num_runs)
    print(f"Time taken to run {num_runs} times: {time_taken} seconds [average {time_taken / num_runs}]")
    # print(f"Python took {time_taken_python} seconds [average {time_taken_python / num_runs}]")
