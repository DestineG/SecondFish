# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable
import src.functions as F

def main():
    # 1. 准备数据
    x_values = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    x = Variable(x_values)
    
    # 2. 计算 y = sin(x)
    y = F.sin(x)
    
    # 3. 计算一阶到三阶导数
    # 注意：为了计算高阶导数，每次 backward 需要清除之前的梯度或支持 create_graph=True
    
    # 一阶导数 (cos x)
    x.clear_grad()
    y.backward(create_graph=True) # 假设你的框架支持高阶导数所需的计算图保持
    dy = x.grad
    
    # 二阶导数 (-sin x)
    x.clear_grad()
    dy.backward(create_graph=True)
    d2y = x.grad
    
    # 三阶导数 (-cos x)
    x.clear_grad()
    d2y.backward()
    d3y = x.grad

    # 4. 绘图
    plt.figure(figsize=(12, 8))
    
    plt.plot(x_values, y.data, label="y = sin(x)", linewidth=2)
    plt.plot(x_values, dy.data, label="y' = cos(x)", linestyle="--")
    plt.plot(x_values, d2y.data, label="y'' = -sin(x)", linestyle=":")
    plt.plot(x_values, d3y.data, label="y''' = -cos(x)", linestyle="-.")

    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5) # 0刻度线
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title("Function and its Derivatives (1st, 2nd, 3rd)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 保存
    save_path = os.path.join(os.getcwd(), "./figures/sin_derivatives.png")
    plt.savefig(save_path)
    print(f"✅ 函数及其一至三阶导数图表已保存至: {save_path}")


if __name__ == "__main__":
    main()