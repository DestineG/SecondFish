# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable, plot_dot_graph
import src.functions as F

class Linear:
    def __init__(self, input_dim, output_dim, normalize=None, activation=None):
        # 定义模型参数
        self.W = Variable(np.random.randn(input_dim, output_dim) * 0.01)
        self.b = Variable(np.zeros(output_dim))
        self.normalize = normalize
        self.activation = activation


    def predict(self, x):
        y = F.linear(x, self.W, self.b)
        if self.normalize:
            y = self.normalize(y)
        if self.activation:
            y = self.activation(y)
        return y
    
    def clear_grad(self):
        self.W.clear_grad()
        self.b.clear_grad()
    
    def optimize(self, lr):
        self.W.data -= lr * self.W.grad.data
        self.b.data -= lr * self.b.grad.data

class Regression:
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        self.fc = [
            Linear(input_dim, hidden_dim, normalize=None, activation=None),
            Linear(hidden_dim, hidden_dim, normalize=F.normalize, activation=F.tanh),
            Linear(hidden_dim, hidden_dim, normalize=None, activation=F.tanh),
            Linear(hidden_dim, hidden_dim, normalize=F.normalize, activation=F.tanh),
            Linear(hidden_dim, hidden_dim, normalize=None, activation=F.tanh)
        ]
        self.fc_out = Linear(hidden_dim, output_dim)

    def predict(self, x):
        for layer in self.fc:
            x = layer.predict(x)
        return self.fc_out.predict(x)

    def clear_grad(self):
        for layer in self.fc:
            layer.clear_grad()
        self.fc_out.clear_grad()

    def optimize(self, lr):
        for layer in self.fc:
            layer.optimize(lr)
        self.fc_out.optimize(lr)

def target_func(x):
    return np.sin(2 * np.pi * x)

def main():
    # 设置随机种子以保证结果可复现
    np.random.seed(42)

    # 实例化模型与定义超参数
    model = Regression(1, 1)
    lr = 0.01
    iters = 100000

    # 生成训练数据
    factor = 6
    extend = 1
    x_train = np.random.rand(200, 1) * factor
    y_train = target_func(x_train) + np.random.randn(200, 1) * 0.01 # 添加噪声

    print(f"Starting training for {iters} iterations...")
    # 训练循环
    for i in range(iters):
        x = Variable(x_train)
        t = Variable(y_train)

        # 前向传播
        y_pred = model.predict(x)
        
        # 计算均方误差 (MSE)
        loss = F.mean_squared_error(y_pred, t)

        # 反向传播前手动清空梯度
        model.clear_grad()
        
        loss.backward()

        # 参数更新 (SGD)
        model.optimize(lr)

        if i % 1000 == 0:
            print(f"Iteration {i:4d} | Loss: {loss.data:.6f}")

    # 结果可视化
    print("\nTraining complete. Plotting results...")
    
    # 生成平滑的测试曲线
    x_test = np.linspace(0, factor + extend, 300).reshape(300, 1)
    y_test_pred = model.predict(Variable(x_test)).data
    y_test_true = target_func(x_test)

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_test_true, 'g--', label="Target Function (sin)")
    plt.plot(x_test, y_test_pred, 'r-', linewidth=2, label="Regression Result")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function Fitting with DeZero-style Framework")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()