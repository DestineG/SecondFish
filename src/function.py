# function.py

import numpy as np

from .variable import Variable, as_array

class Function:
    def __call__(self, *inputs) -> Variable | list[Variable]:
        xs = [x.data for x in inputs]

        y = self.forward(*xs)

        if not isinstance(y, tuple):
            y = (y,)
        outputs = [Variable(as_array(y_i)) for y_i in y]
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

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

def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x: Variable) -> Variable:
    return Square()(x)

if __name__ == "__main__":
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(2.0))
    y0 = add(x0, x1)
    y2 = add(y0, x0)
    y1 = square(y2)
    y1.backward()
    print(x0.grad)
    print(x1.grad)
