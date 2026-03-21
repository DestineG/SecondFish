# variable.py

import numpy as np

class Variable:
    def __init__(self, data):
        # 仅支持 np.ndarray 类型的数据
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.grad = None
        self.creator = None

    def set_grad(self, grad):
        self.grad = grad
    
    def get_grad(self):
        return self.grad

    def clear_grad(self):
        self.grad = None
    
    def set_creator(self, func):
        self.creator = func
    
    def get_creator(self):
        return self.creator
    
    def clear_creator(self):
        self.creator = None
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.get_creator()]
        while funcs:
            creater = funcs.pop()
            input = creater.input
            output = creater.output
            grad = creater.backward(output.grad)
            input.set_grad(grad)
            if input.get_creator() is not None:
                funcs.append(input.get_creator())

    @classmethod
    def test(cls):
        data = np.array(1.0)
        x = cls(data)
        print(x.data)

        x.data = np.array(2.0)
        print(x.data)

def as_array(x):
    '''将 np 标量转换为 np.ndarray 类型
    '''
    if np.isscalar(x):
        return np.array(x)
    return x

if __name__ == "__main__":
    Variable.test()