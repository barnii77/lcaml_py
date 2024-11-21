import os
import sys
from timeit import timeit
from lcaml_py.core.interpreter import Interpreter


def run():
    interpreter.vm.context = {}
    result = interpreter.execute()
    print("interpreter returned: ", result)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "tests/end_to_end"
    if folder.endswith("/"):
        folder = folder[:-1]
    for file in os.listdir(folder):
        if not file.endswith(".lml") or file.startswith("_"):
            continue
        with open(f'{folder}/{file}', 'r') as f:
            code = f.read()
        print(f"Running test: {file}")
        print()
        interpreter = Interpreter(code, file)
        num_runs = 1  # 000
        time_taken = timeit(run, number=num_runs)
        # time_taken_python = timeit(python_run, number=num_runs)
        print(f"Time taken to run {num_runs} times: {time_taken} seconds [average {time_taken / num_runs}]")
        # print(f"Python took {time_taken_python} seconds [average {time_taken_python / num_runs}]")
        print("\n----------------\n")
