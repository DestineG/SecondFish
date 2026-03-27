import numpy as np

from .function import Function, as_variable

class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * (-sin(x))
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)

class exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy * y
        return gx

def exp(x):
    return exp()(x)

class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy / x
        return gx

def log(x):
    return Log()(x)

# ===============================================================
# 激活函数
# ===============================================================

class ReLU(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy * (x > 0)
        return gx

def relu(x):
    return ReLU()(x)

class LeakyReLU(Function):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = gy * np.where(x > 0, 1, self.alpha)
        return gx

def leaky_relu(x, alpha=0.2):
    return LeakyReLU(alpha)(x)

class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

class Softmax(Function):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=self.axis, keepdims=True)  # 数值稳定化: Softmax(x) == Softmax(x - c)
        exp_x = np.exp(x)
        y = exp_x / exp_x.sum(axis=self.axis, keepdims=True)
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy - (gy * y).sum(axis=self.axis, keepdims=True)
        gx *= y
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

class LogSoftmax(Function):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=self.axis, keepdims=True)  # 数值稳定化: LogSoftmax(x) == LogSoftmax(x - c)
        log_sum_exp = np.log(np.exp(x).sum(axis=self.axis, keepdims=True))
        y = x - log_sum_exp
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy - np.exp(y) * (gy.sum(axis=self.axis, keepdims=True))
        return gx

def logsoftmax(x, axis=1):
    return LogSoftmax(axis)(x)

class Mean(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        # 计算参与均值计算的元素个数
        if self.axis is None:
            self.numel = x.size
        else:
            # 处理多轴情况或单轴情况
            axes = self.axis if isinstance(self.axis, (list, tuple)) else (self.axis,)
            self.numel = 1
            for ax in axes:
                self.numel *= x.shape[ax]
        
        y = x.mean(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        # mean 的导数是 1/N
        # 先按 Sum 的逻辑还原形状，再乘以系数
        from .utils import reshape_sum_backward
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx / self.numel

def mean(x, axis=None, keepdims=False):
    return Mean(axis, keepdims)(x)

# ===============================================================
# normalize | batch_normalization
# ===============================================================

class Normalize(Function):
    def __init__(self, axis=-1):
        self.axis = axis
        self.eps = 1e-8

    def forward(self, x):
        mu = x.mean(axis=self.axis, keepdims=True)
        var = x.var(axis=self.axis, keepdims=True)
        std = np.sqrt(var + self.eps)
        
        self.std = std
        y = (x - mu) / std
        return y

    def backward(self, gy):
        maen_gy = mean(gy, axis=self.axis, keepdims=True)
        gx = (gy - maen_gy) / self.std
        return gx

def normalize(x, axis=-1):
    return Normalize(axis)(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        from .utils import sum_to
        self.x_shape = x.shape
        y = sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    '''用 sum 对输入 x 进行 rehsape 至 shape
    '''
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        from .utils import reshape_sum_backward
        # grad 形状匹配
        gy = reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        
        # grad 广播
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)
        return gx, gw

def matmul(x, W):
    return MatMul()(x, W)

# ===============================================================
# max | min | clip
# ===============================================================

from .utils import max_backward_shape

class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()
        shape = max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond

def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)

class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y

def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

# ===============================================================
# 损失函数
# ===============================================================

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * 2 * diff / len(diff)
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        y = softmax(x)
        self.y = y
        self.t = t
        loss = mean(-log(y[np.arange(len(t)), t]))
        return loss

    def backward(self, gy):
        x, t = self.inputs
        y = self.y
        batch_size = len(t)
        gx = y.copy()
        gx[np.arange(batch_size), t] -= 1
        gx *= gy / batch_size
        return gx, None

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y

# ===============================================================
# 算子 linear | conv | deconv | pooling
# ===============================================================

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        if b.data is not None:
            gb = sum_to(gy, b.shape)
        else:
            gb = None
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

