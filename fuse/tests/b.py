import a
import abc
from typing import Union

x = Union[int, str]
a = abc.ABCMeta()
print(x, a)