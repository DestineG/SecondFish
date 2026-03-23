# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable

def rosenbrock(x0, x1):
    return (1 - x0) ** 2 + 100 * (x1 - x0 ** 2) ** 2

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
    
    lr = 1.0 
    iters = 10 
    path = [] 

    print("Starting Automatic Newton's Method Optimization...")
    for i in range(iters):
        path.append([float(x0.data), float(x1.data)])
        
        y = rosenbrock(x0, x1)
        print(f"iter={i}, x0={x0.data:.4f}, x1={x1.data:.4f}, loss={float(y.data):.6f}")

        # ---- 第一步：自动求一阶导 (需开启 create_graph) ----
        x0.clear_grad()
        x1.clear_grad()
        y.backward(create_graph=True) # 关键：开启二阶导支持
        
        # 此时的 grad 是 Variable 对象，不是纯数值
        g0 = x0.grad 
        g1 = x1.grad
        grad_vec = np.array([float(g0.data), float(g1.data)])

        # ---- 第二步：自动求二阶导 (Hessian) ----
        # Hessian 第一行: 对 g0 求导 [f_x0x0, f_x0x1]
        x0.clear_grad()
        x1.clear_grad()
        g0.backward(create_graph=True)
        h00 = float(x0.grad.data)
        h01 = float(x1.grad.data)

        # Hessian 第二行: 对 g1 求导 [f_x1x0, f_x1x1]
        x0.clear_grad()
        x1.clear_grad()
        g1.backward() # 最后一次求导，可以不传 create_graph
        h10 = float(x0.grad.data)
        h11 = float(x1.grad.data)

        H = np.array([[h00, h01], [h10, h11]])

        # ---- 第三步：计算牛顿步长并更新 ----
        try:
            delta = np.linalg.solve(H, grad_vec)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered, falling back to gradient descent.")
            delta = 0.1 * grad_vec # 矩阵不可逆时的简单退化处理

        x0.data -= lr * delta[0]
        x1.data -= lr * delta[1]

        if np.all(np.abs(delta) < 1e-7): 
            break

    # 绘图
    plot_contour_and_path(path, 
                          save_path=os.path.join(save_dir, 'auto_newton_path.png'),
                          title="Auto-Differentiation Newton's Method")

if __name__ == "__main__":
    main()