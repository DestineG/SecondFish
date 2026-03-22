# src/operators.py

from .variable import Variable
from .function import add, mul

def setup_operators():
    # 动态将函数绑定到 Variable 的魔法方法上
    Variable.__add__ = add
    Variable.__mul__ = mul

# 执行注册
setup_operators()