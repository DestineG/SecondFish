# function.py

import numpy as np

from variable import Variable

class Function:
    def __call__(self, input: Variable) -> Variable:
        # Variable -> np.ndarray
        x = input.data

        # result of forward()
        y = self.forward(x)

        # record the input variable for backward()
        self.input = input

        return Variable(y)

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, input: np.ndarray) -> np.ndarray:
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
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.exp(input)
        return output

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array([0.5, 0.7]))
    a = A(x)
    b = B(a)
    y = C(b)
    y.set_grad(np.array([1.0, 1.0]))
    b.set_grad(C.backward(y.get_grad()))
    a.set_grad(B.backward(b.get_grad()))
    x.set_grad(A.backward(a.get_grad()))
    print(f"Input: {x.data}")
    print(f"Output: {y.data}")
    print(f"Gradient: {x.get_grad()}")