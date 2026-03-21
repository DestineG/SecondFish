# function.py

import numpy as np

from variable import Variable

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def grad(self, input: Variable, epsilon=1e-5) -> Variable:
        '''
        grad = (f(x + h) - f(x - h)) / (2 * h)
        '''
        x0 = Variable(input.data - epsilon)
        x1 = Variable(input.data + epsilon)
        y1 = self(x0)
        y2 = self(x1)
        grad = (y2.data - y1.data) / (2 * epsilon)
        return Variable(grad)

    @classmethod
    def test(cls):
        x = Variable(np.array([3.0, 4.0]))
        f = cls()
        y = f(x)
        print("#" * 10, f" Testing {cls.__name__}... ", "#" * 10)
        print(f"Input: {x.data}")
        print(f"Output: {y.data}")
        print(f"Gradient: {f.grad(x).data}")

class Square(Function):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = input ** 2
        return output

class Exp(Function):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.exp(input)
        return output

if __name__ == "__main__":
    Square.test()
    Exp.test()