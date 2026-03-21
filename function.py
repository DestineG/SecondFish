# function.py

import numpy as np

from variable import Variable

class Function:
    def __call__(self, input: Variable) -> Variable:
        # Variable -> np.ndarray
        x = input.data

        # result of forward()
        y = self.forward(x)
        output = Variable(y)

        # record the input variable for backward()
        self.input = input
        self.output = output

        # record the creator function for backward()
        output.set_creator(self)

        return output

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, input: np.ndarray):
        raise NotImplementedError()

class Square(Function):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = input ** 2
        return output
    
    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = 2 * x * gy
        self.input.set_grad(gx)
        if self.input.get_creator() is not None:
            self.input.get_creator().backward(gx)

class Exp(Function):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.exp(input)
        return output

    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = np.exp(x) * gy
        self.input.set_grad(gx)
        if self.input.get_creator() is not None:
            self.input.get_creator().backward(gx)

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array([0.5, 0.7]))
    a = A(x)
    b = B(a)
    y = C(b)
    y.backward()
    print(x.get_grad())
    print(a.get_grad())
    print(b.get_grad())