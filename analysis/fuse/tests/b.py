import a
import abc
from typing import Union


class B:
    def __init__(self):
        x = Union[int, str]
        a = abc.ABCMeta()
        print(x, a)
