# layer.py

import os
import numpy as np
import weakref

from .parameter import Parameter
from . import functions as F

class Layer:
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        '''如果设置的属性是 Parameter | Layer | 两者的子类，则将其名称添加到 _params 集合中'''
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
    
    def cleargrads(self):
        for param in self.params():
            param.clear_grad()
    
    def _flatten_params(self, params_dict, parent_key=""):
        '''递归地将所有参数展平到一个字典中，键为层级路径，值为参数对象'''
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, parent_key=key)
            else:
                params_dict[key] = obj
    
    def save_weights(self, path):
        '''将所有参数保存到一个 .npz 文件中，键为层级路径，值为参数的 numpy 数组'''
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        try:
            np.savez(path, **array_dict)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise e
    
    def load_weights(self, path):
        '''从一个 .npz 文件中加载参数，文件中的键应与 save_weights 中保存的键一致'''
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file '{path}' not found.")
        params_dict = {}
        self._flatten_params(params_dict)
        try:
            with np.load(path) as data:
                for key, param in params_dict.items():
                    if key in data:
                        param.data = data[key]
                    else:
                        print(f"Warning: Key '{key}' not found in weight file. Skipping this parameter.")
        except Exception as e:
            raise e

class Linear(Layer):
    def __init__(self, input_size, output_size, bias=True,dtype=np.float32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.W = Parameter(None, name="W")
        if bias:
            self.b = Parameter(np.zeros(self.output_size, dtype=self.dtype), name="b")
        else:
            self.b = None
        self._init_W()
    
    def _init_W(self, xp=np):
        input_size, output_size = self.input_size, self.output_size
        W_data = xp.random.randn(input_size, output_size).astype(self.dtype) * np.sqrt(2.0 / input_size)
        self.W.data = W_data
    
    def forward(self, x):
        y = F.linear(x, self.W, self.b)
        return y