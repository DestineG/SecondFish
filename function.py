# function.py

import numpy as np

from variable import Variable, as_array

class Function:
    def __call__(self, input: Variable) -> Variable:
        # Variable -> np.ndarray
        x = input.data

        # result of forward()
        y = self.forward(x)
        # 0 维的 np.ndarray 输入可能会得到 0 维的 np 标量
        # 因此将 np 标量转换为 np.ndarray 类型
        output = Variable(as_array(y))

        # record the input variable for backward()
        self.input = input
        self.output = output

        # record the creator function for backward()
        output.set_creator(self)

        return output

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class Square(Function):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = input ** 2
        return output
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        self.input.set_grad(gx)
        return gx


class Exp(Function):
    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.exp(input)
        return output

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        self.input.set_grad(gx)
        return gx

def square(x: Variable) -> Variable:
    return Square()(x)

def exp(x: Variable) -> Variable:
    return Exp()(x)

if __name__ == "__main__":
    x = Variable(np.array([0.5, 0.7]))
    a = square(x)
    b = exp(a)
    y = square(b)
    y.backward()
    print(x.get_grad())
    print(a.get_grad())
    print(b.get_grad())