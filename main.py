# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable, plot_dot_graph
import src.functions as F


def check_grad(name, x0, x1, y):
    print(f"--- Testing {name} ---")
    print(f"Shapes: x0:{x0.shape}, x1:{x1.shape} -> y:{y.shape}")
    y.backward()
    print(f"y: \n{y.data}")
    print(f"x0.grad: \n{x0.grad}")
    print(f"x1.grad: \n{x1.grad}")
    print("\n")

def test_broadcasting():
    # 1. 加法广播：(2, 2) + (1, 2)
    # x1 会在第0维被复制
    x0 = Variable(np.array([[10, 20], [30, 40]]))
    x1 = Variable(np.array([[1, 2]]))
    y = x0 + x1 
    check_grad("Addition [[2,2] + [1,2]]", x0, x1, y)

    # 2. 减法广播：(2, 3) - 标量
    # 标量会被复制到所有位置
    x0 = Variable(np.array([[10, 10, 10], [10, 10, 10]]))
    x1 = Variable(np.array([5]))
    y = x0 - x1
    check_grad("Subtraction [[2,3] - scalar]", x0, x1, y)

    # 3. 乘法广播：(2, 3) * (2, 1)
    # x1 会在第1维被拉伸（从1列变成3列）
    x0 = Variable(np.ones((2, 3)))
    x1 = Variable(np.array([[2], [3]]))
    y = x0 * x1
    check_grad("Multiplication [[2,3] * [2,1]]", x0, x1, y)

    # 4. 除法广播：(2, 3) / (3,)
    # x1 会在左侧补维变成 (1, 3)，然后在第0维拉伸
    x0 = Variable(np.array([[10, 20, 30], [10, 20, 30]]))
    x1 = Variable(np.array([2, 2, 2]))
    y = x0 / x1
    check_grad("Division [[2,3] / [3,]]", x0, x1, y)

    # 5. 反向除法测试 (rdiv)：标量 / Variable
    x0 = Variable(np.array([1, 2, 4]))
    y = 8 / x0  # 触发 __rtruediv__
    y.backward()
    print("--- Testing rdiv [8 / [1, 2, 4]] ---")
    print(f"y: {y.data}")
    print(f"x0.grad (should be -8/x^2): {x0.grad}")

if __name__ == "__main__":
    test_broadcasting()