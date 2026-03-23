# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable

def rosenbrock(x0, x1):
    return (1 - x0) ** 2 + 100 * (x1 - x0 ** 2) ** 2

def plot_contour_and_path(path, save_path='optimization_path.png'):
    '''绘制 Rosenbrock 函数的等高线图和优化轨迹'''
    path = np.array(path)
    
    # 准备等高线数据
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    # 计算 Z 值，Rosenbrock 公式
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    plt.figure(figsize=(10, 8))
    
    # 绘制等高线，使用对数刻度以更好地显示平缓区域
    levels = np.logspace(-1, 3, 20)
    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)

    # 绘制轨迹
    plt.plot(path[:, 0], path[:, 1], 'r.-', label='Optimization Path', markersize=2, linewidth=1)
    
    # 标记起点和终点
    plt.plot(path[0, 0], path[0, 1], 'go', label='Start')
    plt.plot(1.0, 1.0, 'bx', label='Target (1.0, 1.0)')
    
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title('Rosenbrock Function Contour and Gradient Descent Path')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

def main():
    save_dir = 'figures'
    os.makedirs(save_dir, exist_ok=True)
    x0 = Variable(np.array(0.0), name="x0")
    x1 = Variable(np.array(2.0), name="x1")
    lr = 0.001
    iters = 10000
    
    path = [] # 用于记录轨迹

    for i in range(iters):
        path.append([float(x0.data), float(x1.data)])
        
        y = rosenbrock(x0, x1)

        x0.clear_grad()
        x1.clear_grad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

        if i % 1000 == 0:
            print(f"iter={i}, x0={x0.data:.4f}, x1={x1.data:.4f}, loss={float(y.data):.4f}")

    # 绘图
    plot_contour_and_path(path, save_path=os.path.join(save_dir, 'rosenbrock_optimization_path.png'))

if __name__ == "__main__":
    main()