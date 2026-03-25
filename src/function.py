# function.py

import numpy as np
import weakref

from .variable import Variable
from .config import Config

def as_array(x):
    '''将标量转换为 np.ndarray 类型
    '''
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs) -> Variable | list[Variable]:
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]

        y = self.forward(*xs)

        if not isinstance(y, tuple):
            y = (y,)
        outputs = [Variable(as_array(y_i)) for y_i in y]

        if Config.enable_backprop:
            # 用于拓扑排序，生成数越大，越靠近输出端
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gy, gy

def add(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)

class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gy, -gy

def sub(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs
        return gy * x1, gy * x0

def mul(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 / x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)

class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy

def neg(x: Variable) -> Variable:
    return Neg()(x)

class Pow(Function):
    def __init__(self, c: float):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** self.c

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        c = self.c
        gx = gy * c * x ** (c - 1)
        return gx

def pow_(x: Variable, c: float) -> Variable:
    return Pow(c)(x)

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

if __name__ == "__main__":
    x0 = Variable(np.array(np.pi/4), name="x0")
    # x1 = Variable(np.array(2.0))
    y0 = sin(x0)
    y0.backward()
    print(f"y0={y0.data}, x0.grad={x0.grad}")