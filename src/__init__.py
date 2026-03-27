# src/__init__.py

from .variable import Variable
from .function import setup_operators
from .config import no_grad
from .utils import get_dot_graph, plot_dot_graph
from .parameter import Parameter
from . import layer

# Variable 运算符绑定
setup_operators()
