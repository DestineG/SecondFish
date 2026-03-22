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
        self.generation = 0

    def set_grad(self, grad):
        self.grad = grad
    
    def get_grad(self):
        return self.grad

    def clear_grad(self):
        self.grad = None
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def get_creator(self):
        return self.creator
    
    def clear_creator(self):
        self.creator = None
    
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_set = set()  # 避免重复添加函数
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.get_creator())
        while funcs:
            creater = funcs.pop()
            gys = [output().grad for output in creater.outputs]   # 获取输出的梯度(调用weakref返回变量)
            gxs = creater.backward(*gys)                        # 计算输入的梯度
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(creater.inputs, gxs):
                if x.grad is None:                              # 如果 x.grad 还没有值，则直接设置
                    x.grad = gx
                else:                                           # 否则进行累加
                    x.grad = x.grad + gx
                if x.get_creator() is not None:
                    add_func(x.get_creator())
            
            if not retain_grad:
                for y in creater.outputs:
                    y().grad = None

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