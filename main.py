# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable

def rosenbrock(x0, x1):
    return (1 - x0) ** 2 + 100 * (x1 - x0 ** 2) ** 2

def rosenbrock_hessian(x0, x1):
    '''
    计算 Rosenbrock 函数在 (x0, x1) 处的 Hessian 矩阵
    H = [[f_x0x0, f_x0x1],
         [f_x1x0, f_x1x1]]
    '''
    # 这里的输入是数值 (float)
    f_x0x0 = 1200 * (x0**2) - 400 * x1 + 2
    f_x0x1 = -400 * x0
    f_x1x0 = -400 * x0
    f_x1x1 = 200.0
    
    return np.array([
        [f_x0x0, f_x0x1],
        [f_x1x0, f_x1x1]
    ])

def plot_contour_and_path(path, save_path='optimization_path.png', title='Optimization Path'):
    '''绘制 Rosenbrock 函数的等高线图和优化轨迹'''
    path = np.array(path)
    
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    plt.figure(figsize=(10, 8))
    levels = np.logspace(-1, 3, 20)
    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)

    # 绘制轨迹，牛顿法点少，增加 markersize 方便观察
    plt.plot(path[:, 0], path[:, 1], 'r.-', label='Newton Path', markersize=6, linewidth=1.5)
    
    plt.plot(path[0, 0], path[0, 1], 'go', label='Start')
    plt.plot(1.0, 1.0, 'bx', label='Target (1.0, 1.0)')
    
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

def main():
    save_dir = 'figures'
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始点
    x0 = Variable(np.array(0.0), name="x0")
    x1 = Variable(np.array(2.0), name="x1")
    
    # 牛顿法通常不需要很小的学习率，甚至可以直接设为 1.0
    # 但在远离极值点时，设为较小的值可以增加稳定性
    lr = 1.0 
    iters = 10 # 牛顿法收敛极快，10次绰绰有余
    
    path = [] 

    print("Starting Newton's Method Optimization...")
    for i in range(iters):
        path.append([float(x0.data), float(x1.data)])
        
        y = rosenbrock(x0, x1)
        print(f"iter={i}, x0={x0.data:.4f}, x1={x1.data:.4f}, loss={float(y.data):.6f}")

        # 获取一阶导数 (Gradient)
        x0.clear_grad()
        x1.clear_grad()
        y.backward()
        grad = np.array([x0.grad, x1.grad])

        # 获取二阶导数 (Hessian)
        H = rosenbrock_hessian(float(x0.data), float(x1.data))

        # x=[x0, x1], grad=[f_x0, f_x1]^T, H=[[f_x0x0, f_x0x1], [f_x1x0, f_x1x1]]
        # 2 阶taylor 展开: f(x + Δx) ≈ f(x) + grad(x)^T *  Δx + 0.5 * Δx^T * H(x) * Δx
        # df/dΔx = grad(x) + H(x) * Δx = 0 => Δx = -H(x)^-1 * grad(x)
        # 计算牛顿步长: delta = H(x)^-1 * grad(x)
        # 使用 np.linalg.solve 比直接求逆矩阵
        delta = np.linalg.solve(H, grad)

        # 更新参数
        x0.data -= lr * delta[0]
        x1.data -= lr * delta[1]

        if np.all(np.abs(delta) < 1e-7): # 提前收敛
            break

    # 绘图
    plot_contour_and_path(path, 
                          save_path=os.path.join(save_dir, 'newton_optimization_path.png'),
                          title="Rosenbrock Optimization using Newton's Method")

if __name__ == "__main__":
    main()