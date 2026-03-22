# src/__init__.py

from .variable import Variable
from .function import add, sub, rsub, mul, div, rdiv, neg, pow_

def setup_operators():
    # 动态将函数绑定到 Variable 的魔法方法上
    # 调用方式
    # var/other: op(var, other)
    # other/var: rop(var, other)
    Variable.__add__ = add          # Variable + other
    Variable.__radd__ = add         # other + Variable
    Variable.__sub__ = sub          # Variable - other
    Variable.__rsub__ = rsub        # other - Variable
    Variable.__mul__ = mul          # Variable * other
    Variable.__rmul__ = mul         # other * Variable
    Variable.__truediv__ = div      # Variable / other
    Variable.__rtruediv__ = rdiv    # other / Variable
    Variable.__neg__ = neg          # -Variable
    Variable.__pow__ = pow_          # Variable ** other


# 执行注册
setup_operators()