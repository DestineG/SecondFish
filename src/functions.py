import numpy as np

from .function import Function
from .variable import Variable

class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x: Variable) -> Variable:
    return Sin()(x)

class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * (-sin(x))
        return gx

def cos(x: Variable) -> Variable:
    return Cos()(x)