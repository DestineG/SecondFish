# function.py

import numpy as np

from variable import Variable

class Function:
    def __call__(self, input: Variable) -> Variable:
        # Variable -> np.ndarray
        x = input.data

        # result of forward()
        y = self.forward(x)

        # result of grad()
        grad = self.grad(x)
        input.set_grad(grad)

        return Variable(y)

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def grad(self, input: np.ndarray, epsilon=1e-5) -> np.ndarray:
        '''
        grad = (f(x + h) - f(x - h)) / (2 * h)
        '''
        x0 = input - epsilon
        x1 = input + epsilon
        y1 = self.forward(x0)
        y2 = self.forward(x1)
        grad = (y2 - y1) / (2 * epsilon)
        return grad

    @classmethod
    def test(cls):
        x = Variable(np.array([3.0, 4.0]))
        f = cls()
        y = f(x)
        print("#" * 10, f" Testing {cls.__name__}... ", "#" * 10)
        print(f"Input: {x.data}")
        print(f"Output: {y.data}")
        print(f"Gradient: {x.get_grad()}")

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