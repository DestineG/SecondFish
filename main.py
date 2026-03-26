# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable, plot_dot_graph
import src.functions as F


def test_matmul():
    print("--- Testing MatMul ---")
    
    # 1. 定义输入
    # x: (2, 3), W: (3, 4) -> y: (2, 4)
    x_data = np.array([[1, 2, 3], 
                       [4, 5, 6]])
    W_data = np.array([[1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 1, 0, 0]])
    
    x = Variable(x_data)
    W = Variable(W_data)
    
    # 2. 前向传播
    y = F.matmul(x, W)
    print(f"y shape: {y.shape}") # 应该是 (2, 4)
    print(f"y data: \n{y.data}")
    
    # 3. 反向传播
    y.backward()
    
    # 4. 验证梯度形状
    print(f"x.grad shape: {x.grad.shape}") # 应该是 (2, 3)
    print(f"W.grad shape: {W.grad.shape}") # 应该是 (3, 4)
    
    print(f"x.grad: \n{x.grad}")
    print(f"W.grad: \n{W.grad}")

    # --- 进阶验证：使用内置的转置方法 ---
    # 假设你已经实现了 Variable.T 或 Variable.transpose()
    print("\n--- Testing MatMul with Transpose property ---")
    x2 = Variable(np.array([[1, 2]])) # (1, 2)
    W2 = Variable(np.array([[3, 4], [5, 6]])) # (2, 2)
    y2 = F.matmul(x2, W2)
    y2.backward()
    print(f"x2.grad: {x2.grad}") # 应该是 [gy*W^T] -> [1,1] * [[3,5],[4,6]] = [7, 11]

if __name__ == "__main__":
    test_matmul()