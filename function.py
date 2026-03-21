# function.py

import numpy as np

from variable import Variable

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

    def grad(self, input):
        '''
        grad = (f(x + h) - f(x - h)) / (2 * h)
        '''
        x = input.data
        x1 = x - 0.00001
        x2 = x + 0.00001
        y1 = self.forward(x1)
        y2 = self.forward(x2)
        return (y2 - y1) / (x2 - x1)

    @classmethod
    def test(cls):
        x = Variable(np.array([3.0, 4.0]))
        f = cls()
        y = f(x)
        print("#" * 10, f" Testing {cls.__name__}... ", "#" * 10)
        print(f"Input: {x.data}")
        print(f"Output: {y.data}")
        print(f"Gradient: {f.grad(x)}")

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

if __name__ == "__main__":
    Square.test()
    Exp.test()